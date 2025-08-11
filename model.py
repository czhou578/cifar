import torch
import torch.nn.utils as utils
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import nn
from torch.amp import GradScaler, autocast


# Basic preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

cifar_classes = cifar_train.classes

class CIFAR100Dataset(Dataset):
    def __init__(self, cifar_dataset):
        self.cifar_dataset = cifar_dataset
    
    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        return image, label

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # nn.Linear(3072, 64),
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1) # flatten [batch_size, 8*32*32]
        x = self.classifier(x)
        return x

wrapped = CIFAR100Dataset(cifar_train)
train_loader = DataLoader(
    wrapped, 
    batch_size=512,           # Larger batch for GPU
    shuffle=True, 
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

mlp = MLP().to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.001)
# optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-3)
# optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3)
scaler = GradScaler()

for epoch in range(30):
    print(f'Starting Epoch {epoch+1}')

    current_loss = 0.0
    num_batches = 0

    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = mlp(inputs)
            loss = loss_function(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        current_loss += loss.item()
        num_batches += 1

    scheduler.step()
    avg_loss = current_loss / num_batches

    print(f'Epoch {epoch+1} finished')
    print(f'average loss is {avg_loss:.4f}')


print("Training has completed")