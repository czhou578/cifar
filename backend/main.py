from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
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

async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode-block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; "
        "frame-ancestors 'none'; "
        "base-uri 'none'"
    )
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = (
        "geolocation=(), microphone=(), camera=(), "
        "payment=(), usb=(), magnetometer=(), accelerometer=(), gyroscope=()"
    )
    response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
    
    return response

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

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])
app.middleware("http")(security_headers)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)