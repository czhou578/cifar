"""
Simple Grid Search with MLflow Tracking
Good for exhaustive search over small parameter spaces
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchmetrics
from torch.amp import GradScaler, autocast
import mlflow
import math
from datetime import datetime
from vision_transformer import ViT
import itertools


def get_data_loaders():
    """Prepare data loaders"""
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

    train_loader = DataLoader(cifar_train, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(cifar_val, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader


def train_with_config(hp_config, num_epochs=30):
    """Train model with specific hyperparameters"""
    
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
    
    model = ViT(model_config).to(device)
    train_loader, val_loader = get_data_loaders()
    
    loss_function = nn.CrossEntropyLoss(label_smoothing=hp_config.get("label_smoothing", 0.1))
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=hp_config["learning_rate"], 
        weight_decay=hp_config["weight_decay"]
    )
    
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
        
        val_acc = val_accuracy.compute().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Log to MLflow
        mlflow.log_metrics({
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
        }, step=epoch)
        
        scheduler.step()
    
    del model
    torch.cuda.empty_cache()
    
    return best_val_acc


def run_grid_search(experiment_name="/Users/colizu2020@gmail.com/cifar-100-vit-gridsearch"):
    """Run grid search over hyperparameter space"""
    
    mlflow.set_experiment(experiment_name)
    
    # Define search space
    search_space = {
        "learning_rate": [5e-4, 1e-3, 2e-3],
        "weight_decay": [0.05, 0.1, 0.15],
        "warmup_epochs": [5, 10, 15],
        "dropout_rate": [0.1, 0.15, 0.2],
        "label_smoothing": [0.0, 0.1]
    }
    
    # Generate all combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    all_configs = []
    
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        all_configs.append(config)
    
    print(f"Total configurations to try: {len(all_configs)}")
    print(f"Estimated time (30 epochs each): ~{len(all_configs) * 45} minutes\n")
    
    # Option to limit search
    max_trials = 15  # Limit to first 15 combinations
    configs_to_run = all_configs[:max_trials]
    
    results = []
    
    with mlflow.start_run(run_name=f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        for i, hp_config in enumerate(configs_to_run, 1):
            print(f"\n{'='*80}")
            print(f"Running config {i}/{len(configs_to_run)}")
            print(f"{'='*80}")
            print(f"Hyperparameters: {hp_config}")
            
            with mlflow.start_run(run_name=f"trial_{i}", nested=True):
                # Log hyperparameters
                mlflow.log_params(hp_config)
                mlflow.log_param("trial_number", i)
                
                try:
                    best_val_acc = train_with_config(hp_config, num_epochs=30)
                    
                    mlflow.log_metric("best_val_accuracy", best_val_acc)
                    
                    results.append({
                        "config": hp_config,
                        "best_val_accuracy": best_val_acc,
                        "trial": i
                    })
                    
                    print(f"✅ Best Val Accuracy: {best_val_acc:.4f}")
                    
                except Exception as e:
                    print(f"❌ Trial failed: {e}")
                    mlflow.log_param("status", "failed")
    
    # Print summary
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    
    results.sort(key=lambda x: x["best_val_accuracy"], reverse=True)
    
    print("\nTOP 5 CONFIGURATIONS:")
    for i, result in enumerate(results[:5], 1):
        print(f"\nRank {i}:")
        print(f"  Val Accuracy: {result['best_val_accuracy']:.4f}")
        print(f"  Config: {result['config']}")
    
    return results


if __name__ == "__main__":
    print("Starting Grid Search...")
    print("This will exhaustively search the hyperparameter space\n")
    
    results = run_grid_search()
    
    print("\n✅ Grid search complete! Check MLflow UI for detailed results")
