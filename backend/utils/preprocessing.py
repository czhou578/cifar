import torch
from torchvision import transforms
from PIL import Image
import io 
import logging

logger = logging.getLogger(__name__)

inference_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Ensure correct size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocess uploaded image bytes into tensor ready for inference
    """

    try:
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != "RGB":
            image = image.convert('RGB')
        
        tensor = inference_transform(image)

        logger.info(f"Preprocessed image shape: {tensor.shape}")

        return tensor
    
    except Exception as e:
        logger.error(f'Error preprocessing imageL {e}')
        raise ValueError(f"Invalid image format: {e}")
