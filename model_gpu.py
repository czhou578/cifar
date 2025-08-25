import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import nn
from collections import OrderedDict
from torch.utils.data import Subset

# GPU-specific optimizations for Google Colab
torch.backends.cudnn.benchmark = True  # Enable for GPU optimization
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
torch.set_float32_matmul_precision('high')  # Use high precision for GPU

'''
THIS IS THE GPU VERSION OF THE MODEL FOR GOOGLE COLAB
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

# GPU-optimized data augmentation (same transforms, optimized order)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))  # Reduced probability for better training
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
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
train_size = int(0.9 * total_size)
val_size = total_size - train_size 

train_indices = list(range(0, train_size))
val_indices = list(range(train_size, total_size))

# Apply transforms to each split
cifar_train = datasets.CIFAR100(root="./data", train=True, transform=transform)
cifar_val = datasets.CIFAR100(root="./data", train=True, transform=test_transform)
cifar_test = datasets.CIFAR100(root="./data", train=False, transform=test_transform)  # Use actual test set

# Create subsets with indices
cifar_train = Subset(cifar_train, train_indices)
cifar_val = Subset(cifar_val, val_indices)

class CIFAR100Dataset(Dataset):
    def __init__(self, cifar_dataset):
        self.cifar_dataset = cifar_dataset

    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        return image, label

class CachedCIFAR100Dataset(Dataset):
    def __init__(self, cifar_dataset, cache_in_memory=True):
        self.cifar_dataset = cifar_dataset
        self.cache = {}
        
        # For GPU, we don't cache in CPU memory to avoid GPU<->CPU transfers
        if cache_in_memory and torch.cuda.is_available():
            print("GPU detected: Disabling CPU caching for optimal GPU performance")
            cache_in_memory = False
        
        if cache_in_memory:
            print("Caching dataset in memory...")
            for i in range(len(cifar_dataset)):
                self.cache[i] = cifar_dataset[i]
            print("Dataset cached successfully")

    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        return self.cifar_dataset[idx]

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
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            update_metrics(metrics, outputs, targets)
    
    final_acc = compute_accuracy(metrics)
    print(f"Final Test Accuracy: {final_acc:.4f}")

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename):
    """Save training checkpoint"""
    # Move model to CPU before saving for cross-device compatibility
    model.cpu()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'model_architecture': 'MLP'
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")
    
    # Move model back to GPU
    device = next(model.parameters()).device
    if device.type == 'cpu' and torch.cuda.is_available():
        model.cuda()

# GPU-optimized datasets (no CPU caching)
train_dataset = CachedCIFAR100Dataset(cifar_train, cache_in_memory=False)
val_dataset = CachedCIFAR100Dataset(cifar_val, cache_in_memory=False)
test_dataset = CachedCIFAR100Dataset(cifar_test, cache_in_memory=False)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # Keep same architecture for cross-device compatibility
        self.layers = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(3, 32, 3, padding=1)),
            ('bn1_1', nn.BatchNorm2d(32)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2)),
            ('drop1', nn.Dropout(0.1)),

            ('conv2_1', nn.Conv2d(32, 64, 3, padding=1)),
            ('bn2_1', nn.BatchNorm2d(64)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2)),
            ('drop2', nn.Dropout(0.1)),

            ('conv3_1', nn.Conv2d(64, 128, 3, padding=1)),
            ('bn3_1', nn.BatchNorm2d(128)),
            ('relu3_1', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(2)),
            ('drop3', nn.Dropout(0.1)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(128 * 4 * 4, 256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('drop1', nn.Dropout(0.1)),
            ('fc2', nn.Linear(256, 100))
        ]))

    def forward(self, x):
        x = self.quant(x)
        x = self.layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

# GPU-optimized data loaders
train_loader = DataLoader(
    cifar_train,
    batch_size=256,  # Much larger batch size for GPU
    shuffle=True,
    num_workers=4,   # More workers for GPU
    pin_memory=True,  # Essential for GPU performance
    persistent_workers=True,
    prefetch_factor=2,  # Reduced for GPU memory efficiency
    drop_last=True  # For consistent batch sizes on GPU
)

val_loader = DataLoader(
    cifar_val,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)

if __name__ == '__main__':
    # GPU device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Clear GPU cache
        torch.cuda.empty_cache()

    num_classes = 100
    train_metrics = reset_metrics()
    val_metrics = reset_metrics()

    mlp = MLP().to(device)
    
    # GPU-specific optimizations
    if device == 'cuda':
        # Use mixed precision for faster training
        scaler = torch.cuda.amp.GradScaler()
        # Compile model for faster execution (PyTorch 2.0+)
        try:
            mlp = torch.compile(mlp)
            print("Model compiled for faster execution")
        except:
            print("Model compilation not available, using standard mode")
    else:
        # CPU optimizations (fallback)
        mlp = mlp.to(memory_format=torch.channels_last)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)

    # Longer training for better accuracy
    num_epochs = 50
    loss_function = nn.CrossEntropyLoss()
    
    # GPU-optimized learning rate and weight decay
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=5e-4)

    # Scheduler for longer training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the restart period each time
        eta_min=1e-6
    )

    # Reduced accumulation steps for larger GPU batch sizes
    accumulation_steps = 2  # Effective batch size: 256 * 2 = 512

    # Early stopping
    best_val_acc = 0.0
    patience = 15  # Increased patience for longer training
    patience_counter = 0
    best_model_state = None

    print(f"Starting training for {num_epochs} epochs...")
    print(f"Effective batch size: {train_loader.batch_size * accumulation_steps}")

    for epoch in range(num_epochs):
        print(f'Starting Epoch {epoch+1}/{num_epochs}')
        mlp.train()

        current_loss = 0.0
        num_batches = 0
        train_metrics = reset_metrics()

        for i, data in enumerate(train_loader):
            inputs, targets = data
            
            if device == 'cuda':
                # GPU-optimized data transfer
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = mlp(inputs)
                    loss = loss_function(outputs, targets) / accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step(epoch + i / len(train_loader))
            else:
                # CPU fallback
                inputs = inputs.to(device, memory_format=torch.channels_last)
                targets = targets.to(device)
                
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
            
            # Progress monitoring
            if i % 50 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f'Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {lr:.6f}')

        avg_train_loss = current_loss / num_batches
        train_acc = compute_accuracy(train_metrics)

        print(f'Epoch {epoch+1} finished')
        print(f'Training - Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}')

        # Validation every 2 epochs or at the end
        if epoch % 2 == 0 or epoch == num_epochs - 1:
            mlp.eval()
            val_loss = 0.0
            val_batches = 0
            val_metrics = reset_metrics()

            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_targets = val_data
                    
                    if device == 'cuda':
                        val_inputs = val_inputs.to(device, non_blocking=True)
                        val_targets = val_targets.to(device, non_blocking=True)
                        
                        with torch.cuda.amp.autocast():
                            val_outputs = mlp(val_inputs)
                            val_batch_loss = loss_function(val_outputs, val_targets)
                    else:
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
            
            # Early stopping logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = mlp.state_dict().copy()
                print(f'New best validation accuracy: {best_val_acc:.4f}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(mlp, optimizer, scheduler, epoch + 1, avg_train_loss, 
                          f'checkpoint_epoch_{epoch+1}.pth')

    print("Training has completed")
    
    # Load best model
    if best_model_state is not None:
        mlp.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")

    print("\n--- Saving Trained Model ---")
    
    # Move to CPU before saving for cross-device compatibility
    mlp.cpu()
    
    torch.save({
        'model_state_dict': mlp.state_dict(),
        'model_architecture': 'MLP',
        'num_classes': 100,
        'input_size': (3, 32, 32),
        'epoch': num_epochs,
        'best_val_accuracy': best_val_acc,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, 'trained_model_gpu.pth')
    
    print("GPU-trained model saved as 'trained_model_gpu.pth'")
    
    # Move back to device for QAT
    mlp.to(device)

    print("\n--- Preparing Model for QAT ---")

    fused_mlp = fuse_model(mlp)

    fused_mlp.train()
    fused_mlp.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(fused_mlp, inplace=True)

    print("\n--- Starting QAT Fine-tuning ---")
    qat_epochs = 3
    optimizer = torch.optim.AdamW(fused_mlp.parameters(), lr=1e-5)
    fused_mlp.train()

    for epoch in range(qat_epochs):
        for inputs, targets in train_loader:
            if device == 'cuda':
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            else:
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
    fused_mlp.cpu()  # Move to CPU for quantization
    quantized_model = torch.quantization.convert(fused_mlp, inplace=False)

    print("\n--- Saving Quantized Model ---")
    torch.jit.save(torch.jit.script(quantized_model), 'quantized_model_gpu.pth')
    print("Quantized model saved as 'quantized_model_gpu.pth'")

    print("\n--- Evaluating Final INT8 Model ---")
    # Move test loader data to CPU for quantized model evaluation
    test_loader_cpu = DataLoader(
        test_dataset,
        batch_size=64,  # Smaller batch for CPU evaluation
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    evaluate_model(quantized_model, test_loader_cpu, 'cpu')

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Models saved:")
    print(f"  - trained_model_gpu.pth (for inference)")
    print(f"  - quantized_model_gpu.pth (optimized for CPU deployment)")
