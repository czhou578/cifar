import torch
from torch.utils.data import  DataLoader
from torchvision import datasets, transforms
from torch import nn
from torch.amp import GradScaler, autocast
import torchmetrics
from collections import OrderedDict
from torch.utils.data import Subset
import traceback
from torchvision.transforms import v2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import os


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Mild to avoid over-distortion
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
    transforms.RandomErasing(p=0.5)  # Apply after normalization for consistency
])

test_transform = transforms.Compose([
    transforms.ToTensor(), # Moved ToTensor before Normalize (good practice)
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

# Download both training and test sets
cifar_train_raw = datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
cifar_test_raw = datasets.CIFAR100(root="./data", train=False, download=True, transform=None)

# Split only the training set into train/val
train_size = int(0.9 * len(cifar_train_raw))  # 45,000 training points
val_size = len(cifar_train_raw) - train_size   # 5,000 validation points

train_indices = list(range(0, train_size))
val_indices = list(range(train_size, len(cifar_train_raw)))

# Create datasets
cifar_train = Subset(datasets.CIFAR100(root="./data", train=True, transform=train_transform), train_indices)
cifar_val = Subset(datasets.CIFAR100(root="./data", train=True, transform=test_transform), val_indices)
cifar_test = datasets.CIFAR100(root="./data", train=False, transform=test_transform)  # Use actual test set

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(3, 96, 3, padding=1)),
            ('bn1_1', nn.BatchNorm2d(96)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(96, 96, 3, padding=1)),
            ('bn1_2', nn.BatchNorm2d(96)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2)),
            ('drop1', nn.Dropout(0.25)),

            ('conv2_1', nn.Conv2d(96, 192, 3, padding=1)),
            ('bn2_1', nn.BatchNorm2d(192)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv2d(192, 192, 3, padding=1)),
            ('bn2_2', nn.BatchNorm2d(192)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2)),
            ('drop2', nn.Dropout(0.3)),

            ('conv3_1', nn.Conv2d(192, 384, 3, padding=1)),
            ('bn3_1', nn.BatchNorm2d(384)),
            ('relu3_1', nn.ReLU(inplace=True)),
            ('conv3_2', nn.Conv2d(384, 384, 3, padding=1)),
            ('bn3_2', nn.BatchNorm2d(384)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(2)),
            ('drop3', nn.Dropout(0.4)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(384 * 4 * 4, 2048)),  # Updated input size
            ('bn1', nn.BatchNorm1d(2048)),
            ('relu1', nn.ReLU(inplace=True)),
            ('drop1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(2048, 1024)),
            ('bn2', nn.BatchNorm1d(1024)),
            ('relu2', nn.ReLU(inplace=True)),
            ('drop2', nn.Dropout(0.3)),
            ('fc3', nn.Linear(1024, 100))
        ]))

        # ADD THIS: Proper initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using best practices"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.out_features == 100:  # Final classification layer
                    # Xavier/Glorot for final layer to prevent overconfident predictions
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                else:
                    # He initialization for hidden layers with ReLU
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

train_loader = DataLoader(
    cifar_train,  # Use directly
    batch_size=1024,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=6
)

val_loader = DataLoader(
    cifar_val,  # Use directly
    batch_size=1024,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=6
)

test_loader = DataLoader(
    cifar_test,
    batch_size=1024,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=6
)

cutmix = v2.CutMix(num_classes=100)
mixup = v2.MixUp(num_classes=100)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

num_classes = 100
train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)


mlp = MLP().to(device)
if hasattr(torch, 'compile'):
    mlp = torch.compile(mlp)
    print("Model compiled for faster execution")

# Define the TTA function BEFORE it's used
def create_tta_transforms():
    """Create multiple test-time augmentation transforms"""
    tta_transforms = []

    # Original image (no augmentation)
    tta_transforms.append(transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ]))

    # Horizontal flip
    tta_transforms.append(transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),  # Always flip
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ]))

    # Small rotations
    for angle in [-3, 3]:
        tta_transforms.append(transforms.Compose([
            transforms.RandomRotation(degrees=(angle, angle)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ]))

    # Slight brightness adjustment
    tta_transforms.append(transforms.Compose([
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ]))

    return tta_transforms

def evaluate_test_set_with_tta():
    """Efficient batch-based TTA evaluation"""
    try:
        print("Loading model...")
        loaded_model_state = torch.load('trained_model_gpu.pth', map_location='cpu')

        loaded_mlp = MLP()

        # Handle compiled model state dict
        state_dict = loaded_model_state['model_state_dict']
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            print("Detected compiled model state dict - extracting original weights...")
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key.replace('_orig_mod.', '')
                    cleaned_state_dict[new_key] = value
                else:
                    cleaned_state_dict[key] = value
            state_dict = cleaned_state_dict

        loaded_mlp.load_state_dict(state_dict)
        loaded_mlp.to(device)
        loaded_mlp.eval()

        # Create TTA transforms
        tta_transforms = create_tta_transforms()
        print(f"Using {len(tta_transforms)} TTA variations")

        # Create multiple test datasets with different augmentations
        tta_datasets = []

        for transform in tta_transforms:
            tta_datasets.append(datasets.CIFAR100(root="./data", train=False, transform=transform))

        # Create data loaders for each TTA variation
        tta_loaders = [DataLoader(dataset, batch_size=512, shuffle=False,
                                 num_workers=2, pin_memory=True) for dataset in tta_datasets]

        # Collect predictions from all TTA variations
        all_predictions = []
        all_targets = None

        for idx, loader in enumerate(tta_loaders):
            print(f"Processing TTA variation {idx+1}/{len(tta_loaders)}")
            batch_predictions = []
            batch_targets = []

            with torch.no_grad():
                for i, (test_inputs, test_targets) in enumerate(loader):
                    test_inputs = test_inputs.to(device, non_blocking=True)

                    with autocast(device_type='cuda'):
                        outputs = loaded_mlp(test_inputs)
                        # Convert to probabilities for averaging
                        probs = F.softmax(outputs, dim=1)

                    batch_predictions.append(probs.cpu())

                    # Store targets only once (they're the same for all TTA variations)
                    if all_targets is None:
                        batch_targets.append(test_targets)

                    if i % 10 == 0:
                        print(f"  Batch {i}/{len(loader)}")

            # Concatenate all batches for this TTA variation
            variation_predictions = torch.cat(batch_predictions, dim=0)
            all_predictions.append(variation_predictions)

            # Store targets from first variation only
            if all_targets is None:
                all_targets = torch.cat(batch_targets, dim=0)

        # Average predictions across all TTA variations
        print("Averaging TTA predictions...")
        averaged_predictions = torch.stack(all_predictions).mean(dim=0)

        # Calculate metrics
        predicted_classes = averaged_predictions.argmax(dim=1)
        correct = (predicted_classes == all_targets).sum().item()
        total = len(all_targets)
        accuracy = correct / total

        # Calculate per-class metrics using torchmetrics for consistency
        test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        test_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        test_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')

        # Update metrics with TTA predictions
        test_accuracy.update(averaged_predictions, all_targets)
        test_precision.update(averaged_predictions, all_targets)
        test_recall.update(averaged_predictions, all_targets)
        test_f1.update(averaged_predictions, all_targets)

        # Compute final metrics
        tta_acc = test_accuracy.compute()
        tta_prec = test_precision.compute()
        tta_rec = test_recall.compute()
        tta_f1_score = test_f1.compute()

        print("\n" + "="*50)
        print("=== TTA TEST RESULTS ===")
        print("="*50)
        print(f'TTA Test Accuracy:  {tta_acc:.4f} ({tta_acc*100:.2f}%)')
        print(f'TTA Test Precision: {tta_prec:.4f}')
        print(f'TTA Test Recall:    {tta_rec:.4f}')
        print(f'TTA Test F1-Score:  {tta_f1_score:.4f}')
        print("="*50)

        return tta_acc.item()

    except Exception as e:
        print(f"Error during TTA evaluation: {e}")
        traceback.print_exc()
        return None

# Initialize experiment tracker
tracker = ExperimentTracker(experiment_name="/Users/colizu2020@gmail.com/cifar100-experiments")

# Update your config to match actual training parameters
config = {
    # Learning rate - matching your OneCycleLR max_lr
    "learning_rate": 5e-3,  # Changed from 0.001
    "max_lr": 5e-3,         # Added - matches OneCycleLR
    "base_lr": 3e-3,        # Added - matches optimizer

    # Batch size - already matches
    "batch_size": 1024,

    # Epochs - matching your num_epochs
    "epochs": 3,           # Changed from 10

    # Optimizer settings - matching your actual optimizer
    "optimizer": "AdamW",
    "weight_decay": 5e-4,   # Changed from 0.0005 to match 5e-4

    # Scheduler settings - matching your OneCycleLR
    "scheduler": "OneCycleLR",
    "pct_start": 0.4,
    "div_factor": 12.0,
    "final_div_factor": 400.0,
    "anneal_strategy": "cos",

    # Loss function settings
    "loss_function": "CrossEntropyLoss",
    "label_smoothing": 0.1,

    # Model architecture details
    "model_architecture": "CNN-3Block",
    "conv_channels": [96, 192, 384],
    "classifier_dims": [2048, 1024, 100],
    "dropout_rates": [0.25, 0.3, 0.4, 0.5, 0.3],

    # Data augmentation
    "augmentation": "CutMix+MixUp",
    "cutmix_mixup_prob": "Progressive",  # 100% -> 50% -> 30%

    # Training settings
    "mixed_precision": True,
    "early_stopping_patience": 8,
    "validation_frequency": 2,  # Every 2 epochs

    # Dataset info
    "dataset": "CIFAR-100",
    "num_classes": 100,
    "input_size": [3, 32, 32],
    "train_split": 0.9,
    "val_split": 0.1,

    # Hardware settings
    "device": "cuda",
    "num_workers": 2,
    "pin_memory": True,
    "compile_model": True
}

# Use config for epochs
num_epochs = config["epochs"]

# Use config for loss function
loss_function = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

# Use config for optimizer
optimizer = torch.optim.AdamW(
    mlp.parameters(),
    lr=config["base_lr"],
    weight_decay=config["weight_decay"]
)

# Use config for scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config["max_lr"],
    epochs=config["epochs"],
    steps_per_epoch=len(train_loader),
    pct_start=config["pct_start"],
    anneal_strategy=config["anneal_strategy"],
    div_factor=config["div_factor"],
    final_div_factor=config["final_div_factor"]
)

# Use config for early stopping
patience = config["early_stopping_patience"]

scaler = GradScaler()
best_val_loss = float('inf')
patience_counter = 0

# Start MLflow run and log config - FIXED: Missing closing parenthesis
with tracker.start_run(f"cifar100_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    # Log all hyperparameters
    tracker.log_params(config)

    # Log model summary
    total_params = sum(p.numel() for p in mlp.parameters())
    trainable_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    tracker.log_params({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024)
    })

    for epoch in range(num_epochs):
        print(f'Starting Epoch {epoch+1}')
        mlp.train()

        current_loss = 0.0
        num_batches = 0
        train_accuracy.reset()

        for i, data in enumerate(train_loader):
            inputs, targets = data

            # Progressive augmentation reduction
            if epoch >= 45:
                if torch.rand(1) < 0.3:
                    inputs, targets = cutmix_or_mixup(inputs, targets)
            elif epoch >= 30:
                if torch.rand(1) < 0.5:
                    inputs, targets = cutmix_or_mixup(inputs, targets)
            else:
                inputs, targets = cutmix_or_mixup(inputs, targets)

            inputs, targets = inputs.to(device), targets.to(device)

            with autocast(device_type='cuda'):
                outputs = mlp(inputs)
                loss = loss_function(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            current_loss += loss.item()
            num_batches += 1
            train_accuracy.update(outputs.detach(), targets)

            if i % 50 == 0:
                print(f'Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}')

        avg_train_loss = current_loss / num_batches
        print(f'Epoch {epoch+1} finished')
        print(f'Training - Loss: {avg_train_loss:.4f}')

        # Validation
        if (epoch + 1) % config["validation_frequency"] == 0:
            mlp.eval()
            val_loss = 0.0
            val_batches = 0

            val_accuracy.reset()
            val_precision.reset()
            val_recall.reset()
            val_f1.reset()

            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_targets = val_data
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)

                    val_outputs = mlp(val_inputs)
                    val_batch_loss = loss_function(val_outputs, val_targets)

                    val_loss += val_batch_loss.item()
                    val_batches += 1

                    val_accuracy.update(val_outputs, val_targets)
                    val_precision.update(val_outputs, val_targets)
                    val_recall.update(val_outputs, val_targets)
                    val_f1.update(val_outputs, val_targets)

            avg_val_loss = val_loss / val_batches
            val_acc = val_accuracy.compute()
            val_prec = val_precision.compute()
            val_rec = val_recall.compute()
            val_f1_score = val_f1.compute()

            print(f'Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}')
            print(f'Validation - Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1_score:.4f}')

            # Log metrics
            tracker.log_metrics({
                "train_loss": avg_train_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch
            }, step=epoch)

            tracker.log_metrics({
                "val_loss": avg_val_loss,
                "val_accuracy": val_acc.item(),
                "val_precision": val_prec.item(),
                "val_recall": val_rec.item(),
                "val_f1": val_f1_score.item()
            }, step=epoch)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': mlp.state_dict(),
                    'model_architecture': 'MLP',
                    'num_classes': 100,
                    'input_size': (3, 32, 32),
                    'epoch': epoch,
                    'val_loss': avg_val_loss
                }, 'best_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Model saving and evaluation INSIDE the MLflow context
    print("\n--- Saving Trained Model ---")
    mlp.cpu()

    torch.save({
        'model_state_dict': mlp.state_dict(),
        'model_architecture': 'MLP',
        'num_classes': 100,
        'input_size': (3, 32, 32),
        'epoch': num_epochs,
    }, 'trained_model_gpu.pth')

    print("GPU-trained model saved as 'trained_model_gpu.pth'")

    # Run TTA evaluation
    print("\n=== Running TTA evaluation ===")
    tta_accuracy = evaluate_test_set_with_tta()

    # Log final results
    if tta_accuracy:
        tracker.log_metrics({
            "final_tta_accuracy": tta_accuracy,
        })

        tracker.log_params({
             "evaluation_method": "TTA_only",
             "tta_transforms_count": 5,
             "final_test_accuracy": tta_accuracy # This will be logged as a param if tta_accuracy is a number
         })


        print(f"\n" + "="*50)
        print("=== FINAL TEST RESULTS ===")
        print("="*50)
        print(f'TTA Test Accuracy: {tta_accuracy:.4f} ({tta_accuracy*100:.2f}%)')
        print("="*50)

    else:
        print("⚠️ TTA evaluation failed - skipping final metrics logging")


    # Log model and artifacts
    # Update model name for Unity Catalog
    # Replace 'your_catalog_name' and 'your_schema_name' with your actual Unity Catalog names
    tracker.log_model(mlp, artifact_path="model", model_name="main.default.cifar100_final_model")

    if os.path.exists('best_model.pth'):
        tracker.log_artifact('best_model.pth', 'models')
    if os.path.exists('trained_model_gpu.pth'):
        tracker.log_artifact('trained_model_gpu.pth', 'models')

print("✅ Experiment logged to MLflow!")
print("Training has completed")

if torch.cuda.is_available():
    torch.cuda.empty_cache()