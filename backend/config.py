import os
from pathlib import Path

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "C:\\Users\\mycol\\WebProjects\\cifar\\backend\\models\\trained_model_gpu.pth")
DEVICE = os.getenv("DEVICE", "cpu")

# API configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# File upload limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}