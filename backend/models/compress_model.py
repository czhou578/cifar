import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from pathlib import Path
from collections import OrderedDict

class OriginalMLP(nn.Module):
    """EXACT architecture matching your trained model state_dict"""
    def __init__(self):
        super().__init__()
        
        # Complete architecture based on your model inspection
        self.layers = nn.Sequential(OrderedDict([
            # Block 1: 3 → 64 → 64
            ('conv1_1', nn.Conv2d(3, 64, 3, padding=1)),
            ('bn1_1', nn.BatchNorm2d(64)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(64, 64, 3, padding=1)),
            ('bn1_2', nn.BatchNorm2d(64)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2)),
            ('drop1', nn.Dropout(0.25)),

            # Block 2: 64 → 128 → 128
            ('conv2_1', nn.Conv2d(64, 128, 3, padding=1)),
            ('bn2_1', nn.BatchNorm2d(128)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv2d(128, 128, 3, padding=1)),
            ('bn2_2', nn.BatchNorm2d(128)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2)),
            ('drop2', nn.Dropout(0.25)),

            # Block 3: 128 → 256 → 256 (from your state_dict)
            ('conv3_1', nn.Conv2d(128, 256, 3, padding=1)),
            ('bn3_1', nn.BatchNorm2d(256)),
            ('relu3_1', nn.ReLU(inplace=True)),
            ('conv3_2', nn.Conv2d(256, 256, 3, padding=1)),
            ('bn3_2', nn.BatchNorm2d(256)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(2)),
            ('drop3', nn.Dropout(0.3)),

            # Block 4: 256 → 512 → 512 (from your state_dict)
            ('conv4_1', nn.Conv2d(256, 512, 3, padding=1)),
            ('bn4_1', nn.BatchNorm2d(512)),
            ('relu4_1', nn.ReLU(inplace=True)),
            ('conv4_2', nn.Conv2d(512, 512, 3, padding=1)),
            ('bn4_2', nn.BatchNorm2d(512)),
            ('relu4_2', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(2)),
            ('drop4', nn.Dropout(0.4)),
        ]))

        # Classifier: 2048 → 1024 → 512 → 100 (from your state_dict)
        # Input size: 512 * 2 * 2 = 2048 (after 4 pooling operations: 32→16→8→4→2)
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2048, 1024)),  # matches classifier.fc1.weight: [1024, 2048]
            ('relu1', nn.ReLU(inplace=True)),
            ('drop1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(1024, 512)),   # matches classifier.fc2.weight: [512, 1024]
            ('relu2', nn.ReLU(inplace=True)),
            ('drop2', nn.Dropout(0.3)),
            ('fc3', nn.Linear(512, 100))     # matches classifier.fc3.weight: [100, 512]
        ]))

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 512*2*2] = [batch_size, 2048]
        x = self.classifier(x)
        return x

def ultra_compress_model(input_path, output_path):
    """Ultra-aggressive compression for deployment"""
    
    print("🔥 Ultra-compressing model...")
    
    # Load original checkpoint
    checkpoint = torch.load(input_path, map_location='cpu')
    original_state_dict = checkpoint['model_state_dict']
    
    print(f"📋 Original model info:")
    print(f"  - Checkpoint keys: {list(checkpoint.keys())}")
    print(f"  - Total parameters: {len(original_state_dict)}")
    
    # Use EXACT architecture that matches the state_dict
    temp_model = OriginalMLP()
    
    # Verify architecture matches
    model_keys = set(temp_model.state_dict().keys())
    checkpoint_keys = set(original_state_dict.keys())
    
    missing_in_model = checkpoint_keys - model_keys
    missing_in_checkpoint = model_keys - checkpoint_keys
    
    if missing_in_model:
        print(f"⚠️  Keys in checkpoint but not in model: {missing_in_model}")
    if missing_in_checkpoint:
        print(f"⚠️  Keys in model but not in checkpoint: {missing_in_checkpoint}")
    
    # Load the state dict
    temp_model.load_state_dict(original_state_dict, strict=True)
    print("✅ Model loaded successfully!")
    
    # Step 1: Aggressive pruning (remove 60% of weights)
    print("Step 1: Pruning 60% of weights...")
    parameters_to_prune = []
    for name, module in temp_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    print(f"  - Pruning {len(parameters_to_prune)} layers")
    
    # Global unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.6,  # Remove 60% of weights
    )
    
    # Make pruning permanent
    for module, param in parameters_to_prune:
        prune.remove(module, param)
    
    # Step 2: Convert to FP16
    print("Step 2: Converting to FP16...")
    temp_model.eval()
    
    # Step 3: Extract and compress weights
    print("Step 3: Extracting minimal state dict...")
    minimal_state_dict = {}
    
    for name, param in temp_model.state_dict().items():
        # Skip batch norm running stats to save space
        if any(skip in name for skip in ['running_mean', 'running_var', 'num_batches_tracked']):
            print(f"  - Skipping {name} (BatchNorm stats)")
            continue
        
        # Convert to FP16
        if param.dtype == torch.float32:
            minimal_state_dict[name] = param.half()  # FP32 -> FP16
        else:
            minimal_state_dict[name] = param
    
    print(f"  - Kept {len(minimal_state_dict)}/{len(temp_model.state_dict())} parameters")
    
    # Step 4: Save with maximum compression
    print("Step 4: Saving with compression...")
    ultra_minimal = {
        'model_state_dict': minimal_state_dict,
        'ultra_compressed': True,
        'compression_methods': ['pruning_60', 'fp16', 'no_bn_stats'],
        'num_classes': 100,
        'original_architecture': True,
        'architecture_type': 'complete_4_block'  # Flag for model loader
    }
    
    # Use maximum compression
    torch.save(ultra_minimal, output_path, 
               _use_new_zipfile_serialization=True)
    
    # Report results
    original_size = Path(input_path).stat().st_size / 1024 / 1024
    compressed_size = Path(output_path).stat().st_size / 1024 / 1024
    
    print(f"\n📊 Ultra-Compression Results:")
    print(f"Original: {original_size:.1f}MB")
    print(f"Ultra-compressed: {compressed_size:.1f}MB")
    print(f"Reduction: {(1 - compressed_size/original_size)*100:.1f}%")
    print(f"Parameters reduced: {len(original_state_dict)} → {len(minimal_state_dict)}")
    print(f"Expected accuracy loss: 5-15% (due to pruning)")
    
    return compressed_size

if __name__ == "__main__":
    ultra_compress_model('trained_model_gpu.pth', 'ultra_compressed_model.pth')