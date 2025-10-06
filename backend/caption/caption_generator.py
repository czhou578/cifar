import requests
from PIL import Image
import io
import random
import logging
from transformers import pipeline

# Initialize the local pipeline
pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

logger = logging.getLogger(__name__)


def generate_caption(image_input):
    """
    Generate a caption for an image using local transformers pipeline
    
    Args:
        image_input: Can be:
            - str: Path to image file
            - PIL.Image.Image: PIL Image object
            - bytes: Raw image bytes
    
    Returns:
        Generated caption string or None
    """
    try:
        # Convert input to PIL Image for the pipeline
        if isinstance(image_input, str):
            # File path
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            # Already a PIL Image
            image = image_input.convert('RGB')
        elif isinstance(image_input, bytes):
            # Raw bytes from file upload
            image = Image.open(io.BytesIO(image_input)).convert('RGB')
        else:
            raise ValueError(f"image_input must be a file path, PIL Image, or bytes. Got {type(image_input)}")
        
        # Use local pipeline for caption generation
        logger.info("Generating caption with local pipeline")
        result = pipe(image)
        
        # Pipeline returns a list like: [{'generated_text': 'a cat sitting on a couch'}]
        if isinstance(result, list) and len(result) > 0:
            caption = result[0].get('generated_text', '')
            if caption:
                logger.info(f"✅ Generated caption: {caption}")
                return caption
        
        logger.warning(f"Unexpected pipeline result format: {result}")
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