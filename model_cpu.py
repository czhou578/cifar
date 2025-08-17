import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import nn
# REMOVE: from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from collections import OrderedDict
from torch.utils.data import Subset

torch.backends.cudnn.benchmark = False  # We're on CPU
torch.set_float32_matmul_precision('medium')  # Allow some precision loss for speed

torch.set_num_threads(4)  # Adjust based on your CPU cores
torch.set_num_interop_threads(2)

'''
THIS IS THE CPU VERSION OF THE MODEL
'''

# Add simple accuracy calculation functions
def calculate_accuracy(outputs, targets):
    """Calculate accuracy manually"""
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return correct / total

def reset_metrics():
    """Reset metric tracking"""
    return {'total_correct': 0, 'total_samples': 0}

def update_metrics(metrics, outputs, targets):
    """Update metric tracking"""
    _, predicted = torch.max(outputs.data, 1)
    metrics['total_correct'] += (predicted == targets).sum().item()
    metrics['total_samples'] += targets.size(0)

def compute_accuracy(metrics):
    """Compute final accuracy"""
    if metrics['total_samples'] == 0:
        return 0.0
    return metrics['total_correct'] / metrics['total_samples']

# Simplified augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

test_transform = transforms.Compose([
    transforms.ToTensor(), # Moved ToTensor before Normalize (good practice)
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

# Download CIFAR-100 WITHOUT transform first
cifar_train_raw = datasets.CIFAR100(
    root="./data",
    train=True,
    download=True,
    transform=None  # No transform initially
)

# Split the raw dataset
total_size = len(cifar_train_raw)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size) 
test_size = total_size - train_size - val_size

train_indices = list(range(0, train_size))
val_indices = list(range(train_size, train_size + val_size))
test_indices = list(range(train_size + val_size, total_size))

# Apply transforms to each split
cifar_train = datasets.CIFAR100(root="./data", train=True, transform=transform)
cifar_val = datasets.CIFAR100(root="./data", train=True, transform=test_transform)
cifar_test = datasets.CIFAR100(root="./data", train=True, transform=test_transform)

# Create subsets with indices
cifar_train = Subset(cifar_train, train_indices)
cifar_val = Subset(cifar_val, val_indices)
cifar_test = Subset(cifar_test, test_indices)

class CIFAR100Dataset(Dataset):
    def __init__(self, cifar_dataset):
        self.cifar_dataset = cifar_dataset

    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        return image, label

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

def evaluate_model(model, data_loader, device):
    """Evaluates a given model."""
    model.eval()
    metrics = reset_metrics()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            update_metrics(metrics, outputs, targets)
    
    final_acc = compute_accuracy(metrics)
    print(f"Final Test Accuracy: {final_acc:.4f}")


train_dataset = CIFAR100Dataset(cifar_train)
val_dataset = CIFAR100Dataset(cifar_val)
test_dataset = CIFAR100Dataset(cifar_test)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # OPTIMIZATION: Smaller, faster model for CPU
        self.layers = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(3, 32, 3, padding=1)),  # Reduced from 64 to 32
            ('bn1_1', nn.BatchNorm2d(32)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2)),
            ('drop1', nn.Dropout(0.1)),

            ('conv2_1', nn.Conv2d(32, 64, 3, padding=1)),  # Reduced from 128 to 64
            ('bn2_1', nn.BatchNorm2d(64)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2)),
            ('drop2', nn.Dropout(0.1)),

            ('conv3_1', nn.Conv2d(64, 128, 3, padding=1)),  # Reduced from 256 to 128
            ('bn3_1', nn.BatchNorm2d(128)),
            ('relu3_1', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(2)),
            ('drop3', nn.Dropout(0.1)),
        ]))

        # OPTIMIZATION: Smaller classifier
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(128 * 4 * 4, 256)),  # Much smaller
            ('relu1', nn.ReLU(inplace=True)),
            ('drop1', nn.Dropout(0.1)),
            ('fc2', nn.Linear(256, 100))  # Direct to output
        ]))

    def forward(self, x):
        x = self.quant(x)
        x = self.layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

train_loader = DataLoader(
    cifar_train,
    batch_size=128,  # OPTIMIZATION: Smaller batch size for CPU
    shuffle=True,
    num_workers=2,   # OPTIMIZATION: Reduce workers for CPU
    pin_memory=False,
    persistent_workers=True,  # OPTIMIZATION: Keep workers alive
    prefetch_factor=4
)

val_loader = DataLoader(
    cifar_val,
    batch_size=128,  # OPTIMIZATION: Smaller batch size
    shuffle=False,
    num_workers=2,   # OPTIMIZATION: Reduce workers
    pin_memory=False,
    persistent_workers=True,
    prefetch_factor=4
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1024,
    shuffle=False,
    num_workers=4,
    pin_memory=False
)

# --- WRAP ALL EXECUTION CODE IN THIS BLOCK ---
if __name__ == '__main__':
    device = 'cpu'
    print(f"Using device: {device}")

    num_classes = 100
    # CHANGE: Replace torchmetrics with simple tracking
    train_metrics = reset_metrics()
    val_metrics = reset_metrics()

    mlp = MLP().to(device)
    mlp = mlp.to(memory_format=torch.channels_last)

    # OPTIMIZATION: Enable optimized attention (if available)
    try:
        torch.backends.cpu.enable_onednn_fusion(True)
    except:
        pass

    if hasattr(torch, 'compile'):
        print("Compiling model for optimization...")
        mlp = torch.compile(mlp, mode="reduce-overhead")  # Use faster compile mode

    num_epochs = 20
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-2)

    new_max_lr = 1e-3 * (256 / 128)**0.5
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=new_max_lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )

    accumulation_steps = 4 # simulate batch size of 128*4 = 512

    for epoch in range(num_epochs):
        print(f'Starting Epoch {epoch+1}')
        mlp.train()

        current_loss = 0.0
        num_batches = 0
        train_metrics = reset_metrics()  # Reset each epoch

        for i, data in enumerate(train_loader):
            inputs, targets = data
            inputs, targets = inputs.to(device, memory_format=torch.channels_last), targets.to(device)

            outputs = mlp(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            current_loss += loss.item() * accumulation_steps
            num_batches += 1
            update_metrics(train_metrics, outputs.detach(), targets)
            
            # Add progress monitoring
            if i % 50 == 0:
                print(f'Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}')

        avg_train_loss = current_loss / num_batches
        train_acc = compute_accuracy(train_metrics)

        print(f'Epoch {epoch+1} finished')
        print(f'Training - Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}')

        if epoch % 2 == 0 or epoch == num_epochs - 1:
        # Validation loop
            mlp.eval()
            val_loss = 0.0
            val_batches = 0
            val_metrics = reset_metrics()

            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_targets = val_data
                    val_inputs = val_inputs.to(device, memory_format=torch.channels_last)
                    val_targets = val_targets.to(device)

                    val_outputs = mlp(val_inputs)
                    val_batch_loss = loss_function(val_outputs, val_targets)

                    val_loss += val_batch_loss.item()
                    val_batches += 1
                    update_metrics(val_metrics, val_outputs, val_targets)

            avg_val_loss = val_loss / val_batches
            val_acc = compute_accuracy(val_metrics)

            print(f'Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}')

    print("Training has completed")

    print("\n--- Preparing Model for QAT ---")

    fused_mlp = fuse_model(mlp)

    fused_mlp.train()
    fused_mlp.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(fused_mlp, inplace=True)

    print("\n--- Starting QAT Fine-tuning ---")
    qat_epochs = 3
    optimizer = torch.optim.AdamW(fused_mlp.parameters(), lr=1e-5) # Use a very low LR
    fused_mlp.train()

    for epoch in range(qat_epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device, memory_format=torch.channels_last)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = fused_mlp(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"QAT fine-tuning epoch {epoch+1}/{qat_epochs} complete.")

    print("\n--- Converting to Final INT8 Model ---")
    fused_mlp.eval()
    quantized_model = torch.quantization.convert(fused_mlp, inplace=False)

    print("\n--- Evaluating Final INT8 Model ---")
    evaluate_model(quantized_model, test_loader, device)


# def prune_and_finetune_model(model, amount=0.3, finetune_epochs=5):
#     """
#     Applies structured pruning to the model and finetunes it to recover lost progress
#     """

#     print(f"Pruning {amount*100}% of channels from all Conv2d layers...")

#     for name, module in model.named_modules():
#         if isinstance(module, nn.Conv2d):
#             prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
    
#     optimizer.param_groups[0]['lr'] = 1e-5

#     for epoch in range(finetune_epochs):
#         model.train()
#         for data in train_loader:
#             inputs, targets = data
#             inputs, targets = inputs.to(device), targets.to(device)

#             with autocast(device_type='cuda'):
#                 outputs = model(inputs)
#                 loss = loss_function(outputs, targets)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad(set_to_none=True)    

#         print(f"Fine-tuning epoch {epoch+1}/{finetune_epochs} complete.")

#     print("\nMaking pruning permanent...")
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Conv2d):
#             prune.remove(module, 'weight')

#     return model        

# mlp = prune_and_finetune_model(mlp, amount=0.4, finetune_epochs=5)