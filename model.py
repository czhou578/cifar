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

num_epochs = 60
loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

base_lr = 4e-3
batch_scale = 1024 / 256  # 4x larger batches
scaled_lr = base_lr * batch_scale**0.5  # Square root scaling

optimizer = torch.optim.AdamW(
    mlp.parameters(),
    lr=3e-3,  # Keep this for now, let OneCycleLR handle it
    weight_decay=5e-4
)

# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=6e-3,                # Reduce from 8e-3 to 6e-3 for stability
#     epochs=num_epochs,          # 50
#     steps_per_epoch=len(train_loader),
#     pct_start=0.35,             # Longer warmup (17 epochs vs 12)
#     anneal_strategy='cos',
#     div_factor=15.0,            # Higher start LR (4e-4 vs 4e-5)
#     final_div_factor=300.0      # Much lower final LR (2e-5 vs 8e-5)
# )

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=5e-3,                # Slightly lower peak LR for stability
    epochs=num_epochs,          # Keep 45 epochs
    steps_per_epoch=len(train_loader),
    pct_start=0.4,              # Increase warmup to 40% (18 epochs)
    anneal_strategy='cos',
    div_factor=12.0,            # Start LR = 5e-3 / 12 = 4.2e-4
    final_div_factor=400.0      # Final LR = 5e-3 / 400 = 1.25e-5
)

# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=scaled_lr,  # ~8e-3 (2x higher for larger batches)
#     epochs=num_epochs,
#     steps_per_epoch=len(train_loader),  # Will be 44 instead of 176
#     pct_start=0.25,
#     anneal_strategy='cos',
#     div_factor=20.0,
#     final_div_factor=100.0
# )

# optimizer = torch.optim.AdamW(
#     mlp.parameters(),
#     lr=3e-3,      # Increase from 1e-3 to 2e-3
#     weight_decay=5e-4  # Reduce from 5e-3 to 1e-3
# )

# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=4e-3,                # Keep proven peak LR
#     epochs=num_epochs,          # 60
#     steps_per_epoch=len(train_loader),
#     pct_start=0.25,             # 25% warmup (15 epochs) - longer peak
#     anneal_strategy='cos',
#     div_factor=20.0,            # Start LR = 2e-4 (higher start)
#     final_div_factor=200.0      # Final LR = 2e-5 (much higher final)
# )

# new_max_lr = 2e-3 * (256 / 128)**0.25

# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=new_max_lr,
#     epochs=num_epochs,
#     steps_per_epoch=len(train_loader),
#     pct_start=0.15,  # Increase from 0.1 to 0.3
#     anneal_strategy='cos'
# )

# base_max_lr = 4e-3  # Higher for faster convergence in 60 epochs
# batch_size = 256
# scaling_factor = (batch_size / 256) ** 0.5  # Linear scaling
# max_lr = base_max_lr * scaling_factor  # = 4e-3

# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=max_lr,              # 4e-3 for 60 epochs
#     epochs=num_epochs,          # 60
#     steps_per_epoch=len(train_loader),
#     pct_start=0.2,              # 20% warmup (12 epochs)
#     anneal_strategy='cos',
#     div_factor=25.0,            # Start LR = max_lr/25 = 1.6e-4
#     final_div_factor=10000.0    # Final LR = max_lr/10000 = 4e-7
# )

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=num_epochs, eta_min=1e-5
# )

scaler = GradScaler()

best_val_loss = float('inf')
patience = 8
patience_counter = 0

for epoch in range(num_epochs):
    print(f'Starting Epoch {epoch+1}')
    mlp.train()

    current_loss = 0.0
    num_batches = 0
    train_accuracy.reset()

    for i, data in enumerate(train_loader):
        inputs, targets = data
        
        # Reduce augmentation intensity in later epochs
        # Progressive augmentation reduction
        if epoch >= 45:
            # 30% chance in final phase
            if torch.rand(1) < 0.3:
                inputs, targets = cutmix_or_mixup(inputs, targets)
        elif epoch >= 30:
            # 50% chance in middle phase  
            if torch.rand(1) < 0.5:
                inputs, targets = cutmix_or_mixup(inputs, targets)
        else:
            # 100% chance in early phase
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

        # Add progress monitoring
        if i % 50 == 0:
            print(f'Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}')

    avg_train_loss = current_loss / num_batches
    # train_acc = train_accuracy.compute()

    print(f'Epoch {epoch+1} finished')
    print(f'Training - Loss: {avg_train_loss:.4f}')
    # print(f'Training - Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}')

    if (epoch + 1) % 2 == 0:
        mlp.eval()
        val_loss = 0.0
        val_batches = 0

        print(f'Epoch {epoch+1} finished')
        print(f'average training loss is {avg_train_loss:.4f}')

        val_accuracy.reset()
        val_precision.reset()
        val_recall.reset()
        val_f1.reset()

        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_targets = val_data
                val_inputs = val_inputs.to(device)  # Convert inputs to FP16
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

        print(f'Epoch {epoch+1} finished')
        print(f'Training - Loss: {avg_train_loss:.4f}')
        # print(f'Training - Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}')
        print(f'Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}')
        print(f'Validation - Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1_score:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
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

print("Training has completed")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n--- Saving Trained Model ---")

# IMPORTANT: Move model to CPU before saving for cross-device compatibility
mlp.cpu()

torch.save({
    'model_state_dict': mlp.state_dict(),
    'model_architecture': 'MLP',
    'num_classes': 100,
    'input_size': (3, 32, 32),
    'epoch': num_epochs,
}, 'trained_model_gpu.pth')

print("GPU-trained model saved as 'trained_model_gpu.pth'")

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

def evaluate_test_set():
    try:
        print("Loading model...")
        loaded_model_state = torch.load('trained_model_gpu.pth', map_location='cpu')
        
        # Recreate the model architecture
        loaded_mlp = MLP()
        
        # Handle compiled model state dict
        state_dict = loaded_model_state['model_state_dict']
        
        # Check if this is a compiled model state dict
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            print("Detected compiled model state dict - extracting original weights...")
            # Remove '_orig_mod.' prefix from all keys
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key.replace('_orig_mod.', '')
                    cleaned_state_dict[new_key] = value
                else:
                    cleaned_state_dict[key] = value
            state_dict = cleaned_state_dict
        
        # Load the cleaned state dictionary
        loaded_mlp.load_state_dict(state_dict)
        print("Model weights loaded successfully")

        # Move to device
        loaded_mlp.to(device)
        loaded_mlp.eval()

        print("Using uncompiled model for evaluation")

        # Initialize metrics
        test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
        test_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
        test_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
        test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)

        print(f"Starting evaluation on {len(test_loader)} batches...")
        
        with torch.no_grad():
            for i, test_data in enumerate(test_loader):
                test_inputs, test_targets = test_data
                test_inputs = test_inputs.to(device, non_blocking=True)
                test_targets = test_targets.to(device, non_blocking=True)

                # Use autocast for consistency with training
                with autocast(device_type='cuda'):
                    test_outputs = loaded_mlp(test_inputs)

                test_accuracy.update(test_outputs, test_targets)
                test_precision.update(test_outputs, test_targets)
                test_recall.update(test_outputs, test_targets)
                test_f1.update(test_outputs, test_targets)

                if i % 20 == 0:
                    print(f"Progress: {i}/{len(test_loader)} batches")

        # Compute final metrics
        test_acc = test_accuracy.compute()
        test_prec = test_precision.compute()
        test_rec = test_recall.compute()
        test_f1_score = test_f1.compute()

        print("\n" + "="*50)
        print("=== FINAL TEST RESULTS ===")
        print("="*50)
        print(f'Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)')
        print(f'Test Precision: {test_prec:.4f}')
        print(f'Test Recall:    {test_rec:.4f}')
        print(f'Test F1-Score:  {test_f1_score:.4f}')
        print("="*50)
        
        return test_acc.item()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        return None

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

print("Training has completed")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n--- Saving Trained Model ---")

# IMPORTANT: Move model to CPU before saving for cross-device compatibility
mlp.cpu()

torch.save({
    'model_state_dict': mlp.state_dict(),
    'model_architecture': 'MLP',
    'num_classes': 100,
    'input_size': (3, 32, 32),
    'epoch': num_epochs,
}, 'trained_model_gpu.pth')

print("GPU-trained model saved as 'trained_model_gpu.pth'")

print("\n=== Running standard evaluation ===")
standard_accuracy = evaluate_test_set()

print("\n=== Running TTA evaluation ===")
tta_accuracy = evaluate_test_set_with_tta()

if standard_accuracy and tta_accuracy:
    improvement = (tta_accuracy - standard_accuracy) * 100
    print(f"\n" + "="*50)
    print("=== COMPARISON RESULTS ===")
    print("="*50)
    print(f'Standard Test Accuracy: {standard_accuracy:.4f} ({standard_accuracy*100:.2f}%)')
    print(f'TTA Test Accuracy:      {tta_accuracy:.4f} ({tta_accuracy*100:.2f}%)')
    print(f'TTA Improvement:        +{improvement:.2f} percentage points')
    print("="*50)
