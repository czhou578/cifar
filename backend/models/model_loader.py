import torch
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    """Same model architecture as your training script"""
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
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

class ModelLoader:
    def __init__(self):
        self.model = None
        self.device = None
        self.class_names = None
    
    def load_model(self, model_path: str, device: str = "cpu"):
        """Load the trained model from checkpoint"""
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Set device
            self.device = torch.device(device)
            
            # Load model
            if Path(model_path).suffix == '.pth':
                # Load regular PyTorch model
                self.model = MLP()
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                # Load quantized model
                self.model = torch.jit.load(model_path, map_location=self.device)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Set CPU optimizations
            if device == "cpu":
                torch.set_num_threads(4)
                torch.set_num_interop_threads(2)
                try:
                    torch.backends.cpu.enable_onednn_fusion(True)
                except:
                    pass
            
            # Load CIFAR-100 class names
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