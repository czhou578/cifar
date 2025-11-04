from PIL import Image
import io
import logging
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Generator, Callable
import torch

# Initialize the local pipeline

logger = logging.getLogger(__name__)

class StreamingCaptionGenerator:
    def __init__(self, preload=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self._load_error = None
        self.processor = None
        
        # Preload model if requested
        if preload:
            self._ensure_loaded()
    
    def _ensure_loaded(self):
        if self.model is not None: 
            return True

        if self._load_error is not None:
            return False

        try:
            logger.info("Loading the BLIP model for streaming captions...")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                torch_dtype=torch.float32  # Use float32 for CPU
            ).to(self.device)
            
            # Optimize model
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"✅ Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            self._load_error = e
            logger.error(f"❌ Error loading model: {e}")
            return False

    def generate_caption_stream(self, image_input, callback: Callable[[str], None] = None) -> Generator[str, None, str]:
        """
        Generate caption for an image in a streaming fashion
        
        Args:
            image_input: Can be:
                - str: Path to image file
                - PIL.Image.Image: PIL Image object
                - bytes: Raw image bytes
            callback: Optional function to call with each new token
        
        Yields:
            Generated caption tokens one by one
        """
        logger.info("Generating caption with streaming")

        try:
            # Convert input to PIL Image
            if isinstance(image_input, str):
                image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, Image.Image):
                image = image_input.convert('RGB')
            elif isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input)).convert('RGB')
            else:
                raise ValueError(f"image_input must be a file path, PIL Image, or bytes. Got {type(image_input)}")

            # Prepare inputs
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            logger.info("Starting caption generation...")

            # Generate caption with optimized settings
            with torch.no_grad():  # Disable gradient computation for faster inference
                output_ids = self.model.generate(
                    **inputs, 
                    max_length=30,  # Shorter for faster generation
                    num_beams=3,    # Fewer beams for speed (was 5)
                    early_stopping=True,
                    do_sample=False  # Greedy decoding for speed
                )
            
            caption = self.processor.decode(output_ids[0], skip_special_tokens=True)

            logger.info(f"✅ Generated caption: {caption}")

            # Stream tokens word by word
            words = caption.split()
            for i, word in enumerate(words):
                token = word if i == 0 else f" {word}"
                if callback:
                    callback(token)
                yield token
            
            logger.info("Caption generation completed")
            return caption

        except Exception as e:
            logger.error(f"Caption generation error: {e}")
            # fallback = get_fallback_caption_with_class()

    def unload_model(self):
        """Unload the model to free up resources"""
        if self.model is not None:
            logger.info("Unloading the caption model from memory")
            del self.model
            self.model = None
            self.processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded successfully")

# streaming_generator = StreamingCaptionGenerator(preload=True)  # Preload model at startup


# Convenience function
# def generate_caption_streaming(image_input, callback: Callable[[str], None] = None):
#     """Stream caption generation - yields tokens as they're generated"""
#     return streaming_generator.generate_caption_stream(image_input, callback)


# def generate_caption(image_input):
#     """
#     Generate a caption for an image (streaming version)
    
#     Args:
#         image_input: Can be:
#             - str: Path to image file
#             - PIL.Image.Image: PIL Image object
#             - bytes: Raw image bytes
    
#     Returns:
#         Generated caption string or None
#     """
#     try:
#         # Use streaming generator but collect all tokens
#         caption = ""
#         for token in generate_caption_streaming(image_input):
#             caption += token
        
#         if caption:
#             logger.info(f"✅ Generated caption: {caption}")
#             return caption
        
#         logger.warning("Caption generation returned empty result")
#         return None
            
#     except Exception as e:
#         logger.error(f"Caption generation error: {e}")
#         return None