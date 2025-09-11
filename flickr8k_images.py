from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import nn
from collections import OrderedDict
import traceback
import torch.nn.functional as F
import glob
import os
from torch.utils.data import Subset, random_split


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()

batch_size = 32
num_workers = 2
train_split = 0.8
val_split = 0.1
data_dir="./data/flickr8k" 

class Flickr8kImageDataset(Dataset):
    """PyTorch Dataset for Flickr8K images only (no captions)"""
    
    def __init__(self, data_dir="./data/flickr8k", transform=None):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "Images")
        self.transform = transform
        
        # Find all image files
        self.image_paths = self._find_image_files()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.images_dir}. Please download the dataset first.")
        
        print(f"Found {len(self.image_paths)} images")
    
    def _find_image_files(self):
        """Find all image files in the directory"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
        
        return sorted(image_files)  # Sort for consistent ordering
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            # print("the image is, ", image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Return image and filename (for identification)
        filename = os.path.basename(image_path)
        return image, filename
    
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset with training transforms first
full_dataset = Flickr8kImageDataset(data_dir, train_transform)

# Split dataset
total_size = len(full_dataset)

train_size = int(train_split * total_size)
val_size = int(val_split * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # For reproducible splits
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True if torch.cuda.is_available() else False
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True if torch.cuda.is_available() else False
)

print(f"Dataset splits:")
print(f"  Train: {len(train_dataset)} images")
print(f"  Val:   {len(val_dataset)} images")
print(f"  Test:  {len(test_dataset)} images")
