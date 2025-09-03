# Run this script to see the exact architecture:
import torch

# Load and inspect the saved model
checkpoint = torch.load('trained_model_gpu.pth', map_location='cpu')
state_dict = checkpoint['model_state_dict']

# Print all layer names and shapes
for key, tensor in state_dict.items():
    print(f"{key}: {tensor.shape}")