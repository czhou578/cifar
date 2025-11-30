"""
Vision Transformer Hyperparameter Search - Colab-Friendly Version
Run ONE trial at a time, resume later
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchmetrics
from torch.amp import GradScaler, autocast
import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
import math
from datetime import datetime
# from vision_transformer import ViT
import os
import joblib

def get_data_loaders():
    """Prepare data loaders (reusable across trials)"""
    train_transform = transforms.Compose([
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
        transforms.RandomErasing(p=0.25)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    cifar_train_raw = datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
    
    train_size = int(0.9 * len(cifar_train_raw))
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, len(cifar_train_raw)))

    cifar_train = Subset(datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform), train_indices)
    cifar_val = Subset(datasets.CIFAR100(root="./data", train=True, transform=test_transform), val_indices)

    train_loader = DataLoader(cifar_train, batch_size=256, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=6)
    val_loader = DataLoader(cifar_val, batch_size=512, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=6)
    
    return train_loader, val_loader


def train_single_config(hp_config, num_epochs=30):
    """Train model with a specific config - single trial"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model config
    model_config = {
        "patch_size": 4,
        "num_classes": 100,
        "num_channels": 3,
        "num_hidden_layers": 6,
        "hidden_size": 256,
        "image_size": 32,
        "dropout_rate": hp_config["dropout_rate"],
        "num_attent_heads": 8,
        "intermediate_size": 1024,
        "qkv_bias": True,
        "initializer_range": 0.02,
    }
    
    # Initialize model
    model = ViT(model_config).to(device)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders()
    
    # Training setup
    loss_function = nn.CrossEntropyLoss(label_smoothing=hp_config["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=hp_config["learning_rate"], 
        weight_decay=hp_config["weight_decay"]
    )
    
    # Warmup + Cosine schedule
    warmup_epochs = hp_config["warmup_epochs"]
    def get_lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return max(1e-6 / hp_config["learning_rate"], 0.5 * (1 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
    scaler = GradScaler()
    
    train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=100).to(device)
    val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=100).to(device)
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_accuracy.reset()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item()
            train_accuracy.update(outputs.detach(), targets)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_accuracy.compute().item()
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_accuracy.reset()
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(inputs)
                    loss = loss_function(outputs, targets)
                
                val_loss += loss.item()
                val_accuracy.update(outputs, targets)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_accuracy.compute().item()
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
        
        # Log to MLflow
        mlflow.log_metrics({
            "train_loss": avg_train_loss,
            "train_accuracy": train_acc,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
        }, step=epoch)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Best: {best_val_acc:.4f}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return {
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "final_train_accuracy": train_acc,
        "final_val_accuracy": val_acc,
    }


# Pre-defined configs to try one at a time
PRIORITY_CONFIGS = [
    # Config 1: Balanced (Best starting point)
    {
        "name": "config_1_balanced",
        "learning_rate": 1e-3,
        "weight_decay": 0.1,
        "warmup_epochs": 10,
        "dropout_rate": 0.15,
        "label_smoothing": 0.1
    },
    
    # Config 2: Strong regularization
    {
        "name": "config_2_strong_reg",
        "learning_rate": 1e-3,
        "weight_decay": 0.15,
        "warmup_epochs": 10,
        "dropout_rate": 0.2,
        "label_smoothing": 0.1
    },
    
    # Config 3: Fast learning
    {
        "name": "config_3_fast",
        "learning_rate": 2e-3,
        "weight_decay": 0.1,
        "warmup_epochs": 5,
        "dropout_rate": 0.15,
        "label_smoothing": 0.1
    },
    
    # Config 4: Conservative
    {
        "name": "config_4_conservative",
        "learning_rate": 5e-4,
        "weight_decay": 0.05,
        "warmup_epochs": 15,
        "dropout_rate": 0.1,
        "label_smoothing": 0.0
    },
    
    # Config 5: Aggressive
    {
        "name": "config_5_aggressive",
        "learning_rate": 2e-3,
        "weight_decay": 0.15,
        "warmup_epochs": 10,
        "dropout_rate": 0.2,
        "label_smoothing": 0.1
    },
    
    # Config 6: High dropout
    {
        "name": "config_6_high_dropout",
        "learning_rate": 1e-3,
        "weight_decay": 0.1,
        "warmup_epochs": 10,
        "dropout_rate": 0.25,
        "label_smoothing": 0.15
    },
    
    # Config 7: Low regularization
    {
        "name": "config_7_low_reg",
        "learning_rate": 1.5e-3,
        "weight_decay": 0.05,
        "warmup_epochs": 8,
        "dropout_rate": 0.1,
        "label_smoothing": 0.05
    },
    
    # Config 8: Long warmup
    {
        "name": "config_8_long_warmup",
        "learning_rate": 2e-3,
        "weight_decay": 0.1,
        "warmup_epochs": 15,
        "dropout_rate": 0.15,
        "label_smoothing": 0.1
    },
]


def run_single_trial(trial_index=0, num_epochs=30, experiment_name="/Users/colizu2020@gmail.com/cifar-100-vit-manual"):
    """
    Run a SINGLE trial - perfect for Colab free tier
    
    Args:
        trial_index: Which config to run (0-7)
        num_epochs: How many epochs (30 for quick test, 50 for final)
        experiment_name: MLflow experiment name
    """
    
    if trial_index >= len(PRIORITY_CONFIGS):
        print(f"❌ Invalid trial_index {trial_index}. Must be 0-{len(PRIORITY_CONFIGS)-1}")
        return
    
    hp_config = PRIORITY_CONFIGS[trial_index]
    
    print("="*80)
    print(f"RUNNING TRIAL {trial_index + 1}/{len(PRIORITY_CONFIGS)}")
    print("="*80)
    print(f"Config: {hp_config['name']}")
    print(f"Hyperparameters:")
    for key, value in hp_config.items():
        if key != 'name':
            print(f"  {key}: {value}")
    print("="*80)
    print()
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{hp_config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log hyperparameters
        mlflow.log_params({k: v for k, v in hp_config.items() if k != 'name'})
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("trial_index", trial_index)
        
        # Train model
        print(f"Starting training for {num_epochs} epochs...")
        metrics = train_single_config(hp_config, num_epochs=num_epochs)
        
        # Log final metrics
        mlflow.log_metrics({
            "best_val_accuracy": metrics["best_val_accuracy"],
            "best_val_loss": metrics["best_val_loss"],
            "final_train_accuracy": metrics["final_train_accuracy"],
            "final_val_accuracy": metrics["final_val_accuracy"],
        })
        
        run_id = mlflow.active_run().info.run_id
    
    print()
    print("="*80)
    print("TRIAL COMPLETE")
    print("="*80)
    print(f"✅ Best Val Accuracy: {metrics['best_val_accuracy']:.4f}")
    print(f"✅ Best Val Loss: {metrics['best_val_loss']:.4f}")
    print(f"✅ MLflow Run ID: {run_id}")
    print("="*80)
    
    # Save progress
    progress_file = "search_progress.txt"
    with open(progress_file, "a") as f:
        f.write(f"\nTrial {trial_index}: {hp_config['name']}\n")
        f.write(f"Val Acc: {metrics['best_val_accuracy']:.4f}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write("-"*40 + "\n")
    
    print(f"\n📝 Progress saved to {progress_file}")
    print("\n💡 Next steps:")
    print(f"   - Run trial {trial_index + 1} next: run_single_trial(trial_index={trial_index + 1})")
    print(f"   - Check MLflow UI to compare results")
    print(f"   - Total trials remaining: {len(PRIORITY_CONFIGS) - trial_index - 1}")
    
    return metrics


def show_all_configs():
    """Display all available configs"""
    print("\nAVAILABLE CONFIGURATIONS:")
    print("="*80)
    for i, config in enumerate(PRIORITY_CONFIGS):
        print(f"\nTrial {i}: {config['name']}")
        print("-"*40)
        for key, value in config.items():
            if key != 'name':
                print(f"  {key:20s}: {value}")
    print("\n" + "="*80)
    print(f"\nTotal configs: {len(PRIORITY_CONFIGS)}")
    print("\nTo run a specific config:")
    print("  run_single_trial(trial_index=0)  # Run first config")
    print("  run_single_trial(trial_index=1)  # Run second config")
    print("  etc...")

    
# Run single trial
run_single_trial(trial_index=0, num_epochs=30)
