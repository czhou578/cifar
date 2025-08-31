import torch
from torch.utils.data import  DataLoader
from torchvision import datasets, transforms
from torch import nn
from torch.amp import GradScaler, autocast
import torchmetrics
from collections import OrderedDict
from torch.utils.data import Subset
import traceback

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

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),                              # Increase from 15 to 20
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.15),              # Increase from 0.3 to 0.4
    transforms.RandomGrayscale(p=0.1),                         # Add grayscale augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33))      # Increase from 0.1 to 0.25
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
cifar_train = Subset(datasets.CIFAR100(root="./data", train=True, transform=transform), train_indices)
cifar_val = Subset(datasets.CIFAR100(root="./data", train=True, transform=test_transform), val_indices)
cifar_test = datasets.CIFAR100(root="./data", train=False, transform=test_transform)  # Use actual test set

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(3, 96, 3, padding=1)),      # Increase from 64 to 96
            ('bn1_1', nn.BatchNorm2d(96)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(96, 96, 3, padding=1)),     # Increase from 64 to 96
            ('bn1_2', nn.BatchNorm2d(96)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2)),
            ('drop1', nn.Dropout(0.25)),

            ('conv2_1', nn.Conv2d(96, 192, 3, padding=1)),    # Increase from 128 to 192
            ('bn2_1', nn.BatchNorm2d(192)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv2d(192, 192, 3, padding=1)),   # Increase from 128 to 192
            ('bn2_2', nn.BatchNorm2d(192)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2)),
            ('drop2', nn.Dropout(0.3)),                       # Increase from 0.25 to 0.3
        ]))        

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(192 * 8 * 8, 2048)),    # Change from 128*4*4 to 192*8*8, increase to 2048
            ('bn1', nn.BatchNorm1d(2048)),             # Add BatchNorm
            ('relu1', nn.ReLU(inplace=True)),
            ('drop1', nn.Dropout(0.5)),                # Reduce from 0.7 to 0.5
            ('fc2', nn.Linear(2048, 1024)),            # Increase from 512 to 1024
            ('bn2', nn.BatchNorm1d(1024)),             # Add BatchNorm
            ('relu2', nn.ReLU(inplace=True)),
            ('drop2', nn.Dropout(0.3)),                # Reduce from 0.5 to 0.3
            ('fc3', nn.Linear(1024, 100))
        ]))

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1) # flatten [batch_size, 8*32*32]
        x = self.classifier(x)
        return x

train_loader = DataLoader(
    cifar_train,  # Use directly
    batch_size=256,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

val_loader = DataLoader(
    cifar_val,  # Use directly
    batch_size=256,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)

test_loader = DataLoader(
    cifar_test,
    batch_size=256,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)

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

num_epochs = 80
loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    mlp.parameters(),
    lr=2e-3,      # Increase from 1e-3 to 2e-3
    weight_decay=1e-3  # Reduce from 5e-3 to 1e-3
)

new_max_lr = 2e-3 * (256 / 128)**0.25

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=new_max_lr,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,  # Increase from 0.1 to 0.3
    anneal_strategy='cos'
)

scaler = GradScaler()

best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    print(f'Starting Epoch {epoch+1}')
    mlp.train()

    current_loss = 0.0
    num_batches = 0
    train_accuracy.reset()

    for i, data in enumerate(train_loader):
        inputs, targets = data
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
    train_acc = train_accuracy.compute()

    print(f'Epoch {epoch+1} finished')
    print(f'Training - Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}')

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
        print(f'Training - Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}')
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

# Call the function
evaluate_test_set()