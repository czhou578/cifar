import torch
import torch.nn as nn
from pathlib import Path

def compress_existing_model(input_path, output_path):
    """Compress without changing architecture"""
    
    # Load your trained model
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Convert to half precision (FP16)
    state_dict = checkpoint['model_state_dict']
    compressed_state_dict = {}
    
    for key, tensor in state_dict.items():
        # Convert weights to half precision
        if tensor.dtype == torch.float32:
            compressed_state_dict[key] = tensor.half()
        else:
            compressed_state_dict[key] = tensor
    
    # Save compressed model
    torch.save({
        'model_state_dict': compressed_state_dict,
        'compressed': True
    }, output_path)
    
    original_size = Path(input_path).stat().st_size / 1024 / 1024
    compressed_size = Path(output_path).stat().st_size / 1024 / 1024
    
    print(f"Original: {original_size:.1f}MB")
    print(f"Compressed: {compressed_size:.1f}MB")
    print(f"Reduction: {(1 - compressed_size/original_size)*100:.1f}%")

# Run this locally
compress_existing_model('trained_model_gpu.pth', 'compressed_model.pth')