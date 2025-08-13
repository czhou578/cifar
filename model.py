import torch
import torch.nn.utils as utils
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import random_split
import torchmetrics

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()

# More aggressive augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),  # NEW
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),  # Enhanced
    transforms.RandomErasing(p=0.1),  # NEW - Cutout-like augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

# Download CIFAR-100
cifar_train = datasets.CIFAR100(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

total_size = len(cifar_train)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

# Get classes before splitting
cifar_classes = cifar_train.classes

cifar_train, cifar_val, cifar_test = random_split(cifar_train, [train_size, val_size, test_size])


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
        self.layers = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 3: 8x8 -> 4x4 (NEW!)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1) # flatten [batch_size, 8*32*32]
        x = self.classifier(x)
        return x

wrapped = CIFAR100Dataset(cifar_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1024,
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


mlp = MLP().to(device).half()
if hasattr(torch, 'compile'):
    mlp = torch.compile(mlp)

loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(
    mlp.parameters(), 
    lr=1e-3, 
    weight_decay=5e-4,
    eps=1e-4  # Higher epsilon for FP16 stability
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=3e-3,  # Higher peak learning rate
    total_steps=50 * len(train_loader),  # More epochs
    pct_start=0.1,
    anneal_strategy='cos'
)
# optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-3)
# scaler = GradScaler()

for epoch in range(50):
    print(f'Starting Epoch {epoch+1}')
    mlp.train()  # Add this line

    current_loss = 0.0
    num_batches = 0
    train_accuracy.reset()

    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs = inputs.to(device).half()  # Convert inputs to FP16
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = mlp(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()

        utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
        optimizer.step()

        current_loss += loss.item()
        num_batches += 1

        train_accuracy.update(outputs, targets)

    scheduler.step()
    avg_train_loss = current_loss / num_batches
    train_acc = train_accuracy.compute()

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
            val_inputs = val_inputs.to(device).half()  # Convert inputs to FP16
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


def evaluate_test_set():
    mlp.eval()

    test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    test_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    test_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
    test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)

    with torch.no_grad():
        for test_data in test_loader:
            test_inputs, test_targets = test_data
            test_inputs = test_inputs.to(device).half()  # Convert inputs to FP16
            test_targets = test_targets.to(device)

            test_outputs = mlp(test_inputs)

            test_accuracy.update(test_outputs, test_targets)
            test_precision.update(test_outputs, test_targets)
            test_recall.update(test_outputs, test_targets)
            test_f1.update(test_outputs, test_targets)

    test_acc = test_accuracy.compute()
    test_prec = test_precision.compute()
    test_rec = test_recall.compute()
    test_f1_score = test_f1.compute()

    print("\n=== Final Test Results ===")
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test Precision: {test_prec:.4f}')
    print(f'Test Recall: {test_rec:.4f}')
    print(f'Test F1-Score: {test_f1_score:.4f}')

evaluate_test_set()