from PIL import Image
import io
import random
import logging
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Generator, Callable

# Initialize the local pipeline
# pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

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
        if not self._ensure_loaded():
            error_msg = f"Model not loaded: {self._load_error}"
            logger.error(error_msg)
            fallback = get_fallback_caption_with_class()
            yield fallback
            return fallback
        
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
            fallback = get_fallback_caption_with_class()
            yield fallback
            return fallback

streaming_generator = StreamingCaptionGenerator(preload=True)  # Preload model at startup


# Convenience function
def generate_caption_streaming(image_input, callback: Callable[[str], None] = None):
    """Stream caption generation - yields tokens as they're generated"""
    return streaming_generator.generate_caption_stream(image_input, callback)


def generate_caption(image_input):
    """
    Generate a caption for an image (non-streaming version)
    
    Args:
        image_input: Can be:
            - str: Path to image file
            - PIL.Image.Image: PIL Image object
            - bytes: Raw image bytes
    
    Returns:
        Generated caption string or None
    """
    try:
        # Use streaming generator but collect all tokens
        caption = ""
        for token in generate_caption_streaming(image_input):
            caption += token
        
        if caption:
            logger.info(f"✅ Generated caption: {caption}")
            return caption
        
        logger.warning("Caption generation returned empty result")
        return None
            
    except Exception as e:
        logger.error(f"Caption generation error: {e}")
        return None


def make_creative_caption(basic_caption):
    """
    Transform a basic caption into a creative one
    
    Args:
        basic_caption: Plain caption from the model (can be None)
    
    Returns:
        Creative caption string
    """
    # Handle None or empty captions
    if not basic_caption or basic_caption.strip() == "":
        # Return fun fallback captions
        fallback_captions = [
            "✨ A pixel-perfect moment captured in time",
            "🎨 An artistic composition of visual elements",
            "🌟 A scene that tells a thousand stories",
            "📸 A snapshot of digital beauty",
            "🎭 Visual poetry in its finest form",
            "🖼️ A masterpiece waiting to be discovered",
            "🌈 Colors and shapes in perfect harmony",
            "⚡ Energy captured in a single frame",
            "💫 A moment frozen in digital amber",
            "🎪 A spectacular display of pixels and imagination"
        ]
        return random.choice(fallback_captions)
    
    # Clean up the caption
    basic_caption = basic_caption.strip()
    
    creative_templates = [
        f"🎨 A whimsical scene featuring {basic_caption}",
        f"✨ Behold: {basic_caption}, captured in pixels",
        f"🌟 Picture this: {basic_caption}",
        f"📸 The lens reveals {basic_caption}",
        f"🎭 An artistic glimpse of {basic_caption}",
        f"🖼️ Framed within this image: {basic_caption}",
        f"🌈 A colorful vision of {basic_caption}",
        f"⚡ Electrifying imagery showing {basic_caption}",
        f"💫 Marvel at {basic_caption} in all its glory",
        f"🎪 Step right up and witness {basic_caption}",
        f"🌠 A celestial view of {basic_caption}",
        f"🎬 Action! Starring {basic_caption}",
    ]
    
    return random.choice(creative_templates)


def get_fallback_caption_with_class(class_name=None):
    """
    Get a fun caption based on the predicted class
    
    Args:
        class_name: CIFAR-100 class name
    
    Returns:
        Fun caption string
    """
    if class_name:
        class_based = [
            f"🎯 Looks like a {class_name} living its best life!",
            f"✨ Just a {class_name} doing {class_name} things",
            f"🌟 When you're a {class_name} but make it ✨aesthetic✨",
            f"📸 POV: You're a {class_name} being fabulous",
            f"💫 This {class_name} has main character energy",
            f"🎨 A {class_name} in all its glory",
            f"⚡ {class_name.capitalize()} vibes only",
            f"🎭 The {class_name} everyone's been talking about",
        ]
        return random.choice(class_based)
    
    generic = [
        "✨ A picture worth a thousand tokens",
        "🎨 Art in its purest digital form",
        "🌟 Captured: one perfect pixel moment",
        "📸 This image speaks louder than words",
        "💫 Visual poetry in motion",
        "🎭 A scene that defies description",
        "🌈 Colors dancing in digital harmony",
        "⚡ Energy captured in a single frame"
    ]
    return random.choice(generic)