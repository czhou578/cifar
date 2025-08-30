import torch
import torch.nn.utils as utils
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import nn
from torch.amp import GradScaler, autocast
import torch.nn.utils.prune as prune
import torchmetrics
import torch.profiler
from collections import OrderedDict
from torch.utils.data import Subset

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()

# Simplified augmentation
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(2.8),
#     transforms.RandomGrayscale(0.2),
#     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # Stronger color jitter
#     transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
#     transforms.RandAugment(num_ops=2, magnitude=9),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
#     transforms.RandomErasing(p=0.25, scale=(0.02, 0.33))  # Add random erasing
# ])
# Replace your current transform with stronger augmentation
# transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(15),  # Add rotation
#     transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),  # Add color jitter
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
#     transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))  # Add random erasing
# ])

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
        # self.layers = nn.Sequential(OrderedDict([
        #     ('conv1_1', nn.Conv2d(3, 64, 3, padding=1)),
        #     ('bn1_1', nn.BatchNorm2d(64)),
        #     ('relu1_1', nn.ReLU(inplace=True)),
        #     ('conv1_2', nn.Conv2d(64, 64, 3, padding=1)),
        #     ('bn1_2', nn.BatchNorm2d(64)),
        #     ('relu1_2', nn.ReLU(inplace=True)),
        #     ('pool1', nn.MaxPool2d(2)),
        #     ('drop1', nn.Dropout(0.25)),

        #     ('conv2_1', nn.Conv2d(64, 128, 3, padding=1)),
        #     ('bn2_1', nn.BatchNorm2d(128)),
        #     ('relu2_1', nn.ReLU(inplace=True)),
        #     ('conv2_2', nn.Conv2d(128, 128, 3, padding=1)),
        #     ('bn2_2', nn.BatchNorm2d(128)),
        #     ('relu2_2', nn.ReLU(inplace=True)),
        #     ('pool2', nn.MaxPool2d(2)),
        #     ('drop2', nn.Dropout(0.25)),

        #     # ('conv3_1', nn.Conv2d(128, 256, 3, padding=1)),
        #     # ('bn3_1', nn.BatchNorm2d(256)),
        #     # ('relu3_1', nn.ReLU(inplace=True)),
        #     # ('conv3_2', nn.Conv2d(256, 256, 3, padding=1)),
        #     # ('bn3_2', nn.BatchNorm2d(256)),
        #     # ('relu3_2', nn.ReLU(inplace=True)),
        #     # ('pool3', nn.MaxPool2d(2)),
        #     # ('drop3', nn.Dropout(0.25)),

        #     # ('conv4_1', nn.Conv2d(256, 512, 3, padding=1)),
        #     # ('bn4_1', nn.BatchNorm2d(512)),
        #     # ('relu4_1', nn.ReLU(inplace=True)),
        #     # ('conv4_2', nn.Conv2d(512, 512, 3, padding=1)),
        #     # ('bn4_2', nn.BatchNorm2d(512)),
        #     # ('relu4_2', nn.ReLU(inplace=True)),
        #     # ('pool4', nn.MaxPool2d(2)),
        #     # ('drop4', nn.Dropout(0.25)),
        # ]))

        # self.classifier = nn.Sequential(OrderedDict([
        #     ('fc1', nn.Linear(128 * 4 * 4, 1024)),
        #     ('relu1', nn.ReLU(inplace=True)),
        #     ('drop1', nn.Dropout(0.7)),
        #     ('fc2', nn.Linear(1024, 512)),
        #     ('relu2', nn.ReLU(inplace=True)),
        #     ('drop2', nn.Dropout(0.5)),
        #     ('fc3', nn.Linear(512, 100))
        # ]))

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

# optimizer = torch.optim.AdamW(
#     mlp.parameters(),
#     lr=1e-3,  # This will be the max_lr
#     weight_decay=5e-3
# )

# new_max_lr = 1e-3 * (1024 / 128)**0.25

# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=new_max_lr,  # Set a reasonable max learning rate
#     epochs=num_epochs,
#     steps_per_epoch=len(train_loader),
#     pct_start=0.1, # Use a smaller warmup phase
#     anneal_strategy='cos'
# )

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
        # utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
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

def prune_and_finetune_model(model, amount=0.3, finetune_epochs=5):
    """
    Applies structured pruning to the model and finetunes it to recover lost progress
    """

    print(f"Pruning {amount*100}% of channels from all Conv2d layers...")

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)

    optimizer.param_groups[0]['lr'] = 1e-5

    for epoch in range(finetune_epochs):
        model.train()
        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = loss_function(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        print(f"Fine-tuning epoch {epoch+1}/{finetune_epochs} complete.")

    print("\nMaking pruning permanent...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, 'weight')

    return model

def fuse_model(model):
    """
    Fuses Conv-BN-ReLU layers in a model that uses nn.Sequential with OrderedDict.
    """
    modules_to_fuse = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            # Convert children to a list to allow indexing
            layer_list = list(module.children())
            # Get the string names of the layers
            layer_names = [n for n, _ in module.named_children()]

            for i in range(len(layer_list) - 2):
                if (isinstance(layer_list[i], nn.Conv2d) and
                    isinstance(layer_list[i + 1], nn.BatchNorm2d) and
                    isinstance(layer_list[i + 2], nn.ReLU)):

                    # Construct the full string names for fuse_modules
                    modules_to_fuse.append([
                        f'{name}.{layer_names[i]}',
                        f'{name}.{layer_names[i+1]}',
                        f'{name}.{layer_names[i+2]}'
                    ])

    if modules_to_fuse:
        print(f"Fusing {len(modules_to_fuse)} layers...")
        # Fusion must be done in eval mode.
        model.eval()
        torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
    return model

# mlp = prune_and_finetune_model(mlp, amount=0.4, finetune_epochs=5)

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
    # Load the saved model
    loaded_model_state = torch.load('trained_model_gpu.pth')

    # Recreate the model architecture
    loaded_mlp = MLP()

    # Load the state dictionary
    loaded_mlp.load_state_dict(loaded_model_state['model_state_dict'])

    # Move the loaded model to the appropriate device
    loaded_mlp.to(device)
    loaded_mlp.eval()

    fused_mlp = fuse_model(loaded_mlp)
    print("Model after fusion:\n", fused_mlp)

    test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    test_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    test_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
    test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
        record_shapes=True,
        with_stack=True,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2) # Add a schedule
    ) as profiler:
        with torch.no_grad():
            for i, test_data in enumerate(test_loader):
                test_inputs, test_targets = test_data
                test_inputs = test_inputs.to(device)
                test_targets = test_targets.to(device)

                with torch.profiler.record_function("model_inference"): # FIX: Use torch.profiler.record_function
                    test_outputs = loaded_mlp(test_inputs)

                test_accuracy.update(test_outputs, test_targets)
                test_precision.update(test_outputs, test_targets)
                test_recall.update(test_outputs, test_targets)
                test_f1.update(test_outputs, test_targets)

                profiler.step()

    test_acc = test_accuracy.compute()
    test_prec = test_precision.compute()
    test_rec = test_recall.compute()
    test_f1_score = test_f1.compute()

    print("\n=== Final Test Results ===")
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test Precision: {test_prec:.4f}')
    print(f'Test Recall: {test_rec:.4f}')
    print(f'Test F1-Score: {test_f1_score:.4f}')
    print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))

evaluate_test_set()