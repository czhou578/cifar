import torch
import torch.nn.utils as utils
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import random_split
import torchmetrics
import torch.profiler
from collections import OrderedDict

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()

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
from torch.utils.data import Subset
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

train_dataset = CIFAR100Dataset(cifar_train)
val_dataset = CIFAR100Dataset(cifar_val)
test_dataset = CIFAR100Dataset(cifar_test)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(3, 64, 3, padding=1)),
            ('bn1_1', nn.BatchNorm2d(64)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(64, 64, 3, padding=1)),
            ('bn1_2', nn.BatchNorm2d(64)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2)),
            ('drop1', nn.Dropout(0.1)),

            ('conv2_1', nn.Conv2d(64, 128, 3, padding=1)),
            ('bn2_1', nn.BatchNorm2d(128)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv2d(128, 128, 3, padding=1)),
            ('bn2_2', nn.BatchNorm2d(128)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2)),
            ('drop2', nn.Dropout(0.1)),

            ('conv3_1', nn.Conv2d(128, 256, 3, padding=1)),
            ('bn3_1', nn.BatchNorm2d(256)),
            ('relu3_1', nn.ReLU(inplace=True)),
            ('conv3_2', nn.Conv2d(256, 256, 3, padding=1)),
            ('bn3_2', nn.BatchNorm2d(256)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(2)),
            ('drop3', nn.Dropout(0.1)),

            ('conv4_1', nn.Conv2d(256, 512, 3, padding=1)),
            ('bn4_1', nn.BatchNorm2d(512)),
            ('relu4_1', nn.ReLU(inplace=True)),
            ('conv4_2', nn.Conv2d(512, 512, 3, padding=1)),
            ('bn4_2', nn.BatchNorm2d(512)),
            ('relu4_2', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(2)),
            ('drop4', nn.Dropout(0.1)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512 * 2 * 2, 1024)),
            ('relu1', nn.ReLU(inplace=True)),
            ('drop1', nn.Dropout(0.1)),
            ('fc2', nn.Linear(1024, 512)),
            ('relu2', nn.ReLU(inplace=True)),
            ('drop2', nn.Dropout(0.1)),
            ('fc3', nn.Linear(512, 100))
        ]))

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1) # flatten [batch_size, 8*32*32]
        x = self.classifier(x)
        return x

train_loader = DataLoader(
    cifar_train,  # Use directly
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    cifar_val,  # Use directly
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1024,
    shuffle=False,
    num_workers=2,
    pin_memory=True
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

loss_function = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(
#     mlp.parameters(),
#     lr=1e-4,  # Reduced from 1e-3
#     weight_decay=1e-4,
#     eps=1e-8
# )

# optimizer = torch.optim.SGD(params=mlp.parameters(), momentum=0.9, weight_decay=1e-5, lr=1e-1)

optimizer = torch.optim.AdamW(
    mlp.parameters(),
    lr=1e-3,  # This will be the max_lr
    weight_decay=1e-2
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,  # Set a reasonable max learning rate
    total_steps=50 * len(train_loader),
    pct_start=0.1, # Use a smaller warmup phase
    anneal_strategy='cos'
)
# optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-3)
scaler = GradScaler()

for epoch in range(25):
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


print("Training has completed")

def fuse_model(model):
    modules_to_fuse = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            for i in range(len(module) - 2):
                if (isinstance(module[i], nn.Conv2d) and isinstance(module[i + 1], nn.BatchNorm2d) and isinstance(module[i + 2], nn.ReLU)):
                    modules_to_fuse.append([f'{name}.{i}', f'{name}.{i+1}', f'{name}.{i+2}'])
    if modules_to_fuse:
        print(f"Fusing {len(modules_to_fuse)} layers...")
        torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
    return model

def evaluate_test_set():
    mlp.eval()

    fused_mlp = fuse_model(mlp)
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
                    test_outputs = mlp(test_inputs)

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