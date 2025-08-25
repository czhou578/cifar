from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path

from models.model_loader import model_loader
from routes.inference import router as inference_router
from config import MODEL_PATH, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

origins = [
    "http://localhost",
    "http://localhost:3000"
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler to load model at startup and cleanup
    """

    logger.info("starting up app")

    try:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"File not found: {MODEL_PATH}")
        
        model_loader.load_model(MODEL_PATH, device="cpu")
        logger.info("model load success")

        yield
    
    except Exception as e:
        logger.error(e)
        raise
    finally:
        logger.info('shutting down')

app = FastAPI(
    title="CIFAR-100 Inference API",
    description="backend for images",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference_router, prefix="/api/v1", tags=["inference"])

@app.get("/")
async def root():
    return {"message": "CIFAR-100 Inference API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model_loader.model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "device": str(model_loader.device) if model_loader.device else "unknown"
    }