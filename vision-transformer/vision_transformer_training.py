
"""
weight decay: 5e-5
cosine init lr: 1e-3, highest is 1e-5
warmup 5 epochs

"""


import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchmetrics
from torch.amp import GradScaler, autocast

# Hyperparameter Search Space
search_space = {
    # Learning rate (most critical)
    "learning_rate": [
        5e-4,   # Conservative - good baseline
        1e-3,   # Standard for ViT
        2e-3,   # Aggressive
        3e-3,   # Very aggressive
    ],
    
    # Weight decay (regularization strength)
    "weight_decay": [
        0.01,   # Light regularization
        0.05,   # Medium (current)
        0.1,    # Strong (recommended for ViT)
        0.15,   # Very strong
    ],
    
    # Warmup epochs
    "warmup_epochs": [
        5,      # Short warmup
        10,     # Medium warmup (recommended)
        15,     # Long warmup
    ],
    
    # Dropout rate (change in config)
    "dropout_rate": [
        0.1,    # Light
        0.15,   # Medium
        0.2,    # Strong
    ],
}

# Priority combinations (try these first)
priority_configs = [
    # Configuration 1: Balanced (Best starting point)
    {"learning_rate": 1e-3, "weight_decay": 0.1, "warmup_epochs": 10, "dropout_rate": 0.15},
    
    # Configuration 2: Strong regularization (For overfitting)
    {"learning_rate": 1e-3, "weight_decay": 0.15, "warmup_epochs": 10, "dropout_rate": 0.2},
    
    # Configuration 3: Fast learning (If training is too slow)
    {"learning_rate": 2e-3, "weight_decay": 0.1, "warmup_epochs": 5, "dropout_rate": 0.15},
    
    # Configuration 4: Conservative (If training is unstable)
    {"learning_rate": 5e-4, "weight_decay": 0.05, "warmup_epochs": 15, "dropout_rate": 0.1},
    
    # Configuration 5: Aggressive (To push performance)
    {"learning_rate": 3e-3, "weight_decay": 0.15, "warmup_epochs": 10, "dropout_rate": 0.2},
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
    transforms.RandomErasing(p=0.5)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

cifar_train_raw = datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
cifar_test_raw = datasets.CIFAR100(root='./data', train=False, download=True, transform=None)

train_size = int(0.9 * len(cifar_train_raw))
val_size = len(cifar_train_raw) - train_size

train_indices = list(range(0, train_size))
val_indices = list(range(train_size, len(cifar_train_raw)))

cifar_train = Subset(datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform), train_indices)
cifar_val = Subset(datasets.CIFAR100(root="./data", train=True, transform=test_transform), val_indices)
cifar_test = datasets.CIFAR100(root="./data", train=False, transform=test_transform)

# DataLoaders
train_loader = DataLoader(
    cifar_train,
    batch_size=256,  # Smaller batch for ViT
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=6
)

val_loader = DataLoader(
    cifar_val,
    batch_size=512,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=6
)

model = ViT(config).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params}")

tracker = ExperimentTracker(experiment_name="/Users/colizu2020@gmail.com/cifar-100-vision-transformer")

num_epochs = 2
loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)

scaler = GradScaler()

train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=100).to(device)
val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=100).to(device)

best_val_loss = float('inf')
patience = 15
patience_counter = 0

with tracker.start_run(f"cifar100_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    tracker.log_params(config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tracker.log_params({
      "total_parameters": total_params,
      "trainable_parameters": trainable_params,
      "model_size_mb": total_params * 4 / (1024 * 1024)
    })

    for epoch in range(num_epochs):
        print(f'Starting Epoch {epoch + 1}')
        model.train()

        current_loss = 0.0
        num_batches = 0
        train_accuracy.reset()

        for i, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(input)
                loss = loss_function(outputs, target)     

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)           

            current_loss += loss.item()
            num_batches += 1
            train_accuracy.update(outputs.detach(), target)        

            if i % 50 == 0:
                print(f"Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = current_loss / num_batches
        train_acc = train_accuracy.compute()

        tracker.log_metrics({
            "train_loss": avg_train_loss,
            "train_accuracy": train_acc.item(),
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]['lr']
        }, step=epoch)        
        
        print(f'Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')

        # Validation
        if (epoch + 1) % 2 == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            val_accuracy.reset()

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        outputs = model(inputs)
                        loss = loss_function(outputs, targets)

                    val_loss += loss.item()
                    val_batches += 1
                    val_accuracy.update(outputs, targets)

            avg_val_loss = val_loss / val_batches
            val_acc = val_accuracy.compute()
            
            print(f'Validation - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}')

            tracker.log_metrics({
              "val_loss": avg_val_loss,
              "val_accuracy": val_acc.item(),
              "epoch": epoch
            }, step=epoch)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                tracker.log_metrics({
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": val_acc.item()
                }, step=epoch)

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss,
                    'config': config
                }, 'best_vit_model.pth')
                print("Model saved!")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    print("Training completed!")

    tracker.log_model(model, artifact_path="model", registered_model_name="workspace.default.cifar100_vt_final")

    if os.path.exists('best_model.pth'):
        tracker.log_artifact('best_model.pth', 'models')
    if os.path.exists('trained_model_gpu.pth'):
        tracker.log_artifact('trained_model_gpu.pth', 'models')

tracker.log_metrics({
    "final_train_loss": avg_train_loss,
    "final_train_accuracy": train_acc.item(),
    "final_val_loss": avg_val_loss if (epoch + 1) % 2 == 0 else None,
    "final_val_accuracy": val_acc.item() if (epoch + 1) % 2 == 0 else None,
    "total_epochs_trained": epoch + 1
})   

print("✅ Experiment logged to MLflow!")
print("Training has completed")