# Quick inspection in Python REPL or notebook
import torch

# Load and inspect
checkpoint = torch.load('../backend/models/trained_model_gpu.pth', map_location='cpu')

# Print all top-level keys
print("Checkpoint keys:", list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Direct state_dict")

# Print all layer names and shapes
state_dict = checkpoint.get('model_state_dict', checkpoint)
for name, tensor in state_dict.items():
    print(f"{name}: {tensor.shape}")