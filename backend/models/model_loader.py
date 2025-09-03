import torch
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    """Match the EXACT architecture from the saved model"""
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # Architecture based on actual saved model
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

            # Block 3: 128 → 256 → 256
            ('conv3_1', nn.Conv2d(128, 256, 3, padding=1)),
            ('bn3_1', nn.BatchNorm2d(256)),
            ('relu3_1', nn.ReLU(inplace=True)),
            ('conv3_2', nn.Conv2d(256, 256, 3, padding=1)),
            ('bn3_2', nn.BatchNorm2d(256)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(2)),
            ('drop3', nn.Dropout(0.3)),

            # Block 4: 256 → 512 → 512
            ('conv4_1', nn.Conv2d(256, 512, 3, padding=1)),
            ('bn4_1', nn.BatchNorm2d(512)),
            ('relu4_1', nn.ReLU(inplace=True)),
            ('conv4_2', nn.Conv2d(512, 512, 3, padding=1)),
            ('bn4_2', nn.BatchNorm2d(512)),
            ('relu4_2', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(2)),
            ('drop4', nn.Dropout(0.4)),
        ]))

        # Classifier based on actual saved weights
        # After 4 pooling operations: 32→16→8→4→2, so 512*2*2=2048 input features
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2048, 1024)),  # 512*2*2 → 1024
            ('relu1', nn.ReLU(inplace=True)),
            ('drop1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(1024, 512)),   # 1024 → 512
            ('relu2', nn.ReLU(inplace=True)),
            ('drop2', nn.Dropout(0.3)),
            ('fc3', nn.Linear(512, 100))     # 512 → 100 (CIFAR-100 classes)
        ]))

    def forward(self, x):
        x = self.quant(x)
        x = self.layers(x)
        x = x.view(x.size(0), -1)  # Flatten: [batch_size, 512*2*2]
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
                
                state_dict = checkpoint['model_state_dict']

                if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                    print("Detected compiled model state dict - extracting original weights...")
                    # Remove '_orig_mod.' prefix from all keys
                    cleaned_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('_orig_mod.'):
                            new_key = key.replace('_orig_mod.', '')
                            cleaned_state_dict[new_key] = value
                        else:
                            cleaned_state_dict[key] = value
                    state_dict = cleaned_state_dict
                
                self.model.load_state_dict(state_dict)
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