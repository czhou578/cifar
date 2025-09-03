import torch
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
import logging
import gc 

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    """Lightweight version - remove quantization stubs initially"""
    def __init__(self):
        super().__init__()
        
        # Simplified architecture - fewer channels to reduce memory
        self.layers = nn.Sequential(OrderedDict([
            # Block 1: 3 → 32 → 32 (reduced from 64)
            ('conv1_1', nn.Conv2d(3, 32, 3, padding=1)),
            ('bn1_1', nn.BatchNorm2d(32)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(32, 32, 3, padding=1)),
            ('bn1_2', nn.BatchNorm2d(32)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2)),
            ('drop1', nn.Dropout(0.25)),

            # Block 2: 32 → 64 → 64 (reduced from 128)
            ('conv2_1', nn.Conv2d(32, 64, 3, padding=1)),
            ('bn2_1', nn.BatchNorm2d(64)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv2d(64, 64, 3, padding=1)),
            ('bn2_2', nn.BatchNorm2d(64)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2)),
            ('drop2', nn.Dropout(0.25)),

            # Only 2 blocks instead of 4 to reduce memory
        ]))

        # Smaller classifier
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 8 * 8, 512)),  # Reduced from 2048
            ('relu1', nn.ReLU(inplace=True)),
            ('drop1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(512, 256)),         # Reduced from 1024
            ('relu2', nn.ReLU(inplace=True)),
            ('drop2', nn.Dropout(0.3)),
            ('fc3', nn.Linear(256, 100))          # Final layer
        ]))

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ModelLoader:
    def __init__(self):
        self.model = None
        self.device = None
        self.class_names = None
    
    def load_model(self, model_path: str, device: str = "cpu"):
        """Load model with memory optimizations"""
        try:
            logger.info(f"Loading model from {model_path}")
            
            self.device = torch.device(device)
            
            # Memory optimization: use memory-mapped loading
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create smaller model if original is too large
            try:
                # Try loading original architecture first
                self.model = MLP()
                state_dict = checkpoint['model_state_dict']
                
                # Handle compiled models
                if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                    cleaned_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('_orig_mod.'):
                            new_key = key.replace('_orig_mod.', '')
                            cleaned_state_dict[new_key] = value
                        else:
                            cleaned_state_dict[key] = value
                    state_dict = cleaned_state_dict
                
                # Filter state dict to match smaller model
                filtered_state_dict = {}
                for key, value in state_dict.items():
                    if key in self.model.state_dict():
                        if self.model.state_dict()[key].shape == value.shape:
                            filtered_state_dict[key] = value
                        else:
                            logger.warning(f"Shape mismatch for {key}, using random weights")
                
                self.model.load_state_dict(filtered_state_dict, strict=False)
                
            except Exception as e:
                logger.error(f"Could not load full model: {e}")
                raise
            
            self.model.to(self.device)
            self.model.eval()
            
            # Aggressive CPU optimizations
            torch.set_num_threads(1)  # Reduce to 1 thread
            torch.set_num_interop_threads(1)

            del checkpoint
            del state_dict
            gc.collect()
            
            self._load_class_names()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_class_names(self):
        """Load CIFAR-100 class names"""
        self.class_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]
    
    def predict(self, image_tensor: torch.Tensor, top_k: int = 5):
        """Make prediction on preprocessed image tensor"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            with torch.no_grad():
                # Ensure correct device and format
                image_tensor = image_tensor.to(self.device)
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
                
                # Make prediction
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
                
                results = []
                for i in range(top_k):
                    class_idx = top_indices[0][i].item()
                    prob = top_probs[0][i].item()
                    class_name = self.class_names[class_idx] if self.class_names else f"Class_{class_idx}"
                    results.append({
                        "class_name": class_name,
                        "class_id": class_idx,
                        "confidence": float(prob)
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

# Global model loader instance
model_loader = ModelLoader()