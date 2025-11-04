from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import RedirectResponse
from pathlib import Path
from models.model_loader import model_loader
from routes.inference import router as inference_router
from job_queue.queue import get_job_queue
from routes import websocket_inference
from config import MODEL_PATH, LOG_LEVEL
from dotenv import load_dotenv
from workers.process_manager import get_worker_manager
import logging
import httpx
import os

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

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

        queue = get_job_queue()
        await queue.start()
        logger.info("job queue started")

        worker_manager = get_worker_manager()
        worker_manager.start()
        logger.info("worker manager started")

        yield

        logger.info("Shutting down app")
        await queue.stop()
        logger.info("job queue stopped")
    
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
app.include_router(websocket_inference.router, prefix="/api/v1", tags=["websocket"])


@app.get("/")
async def root():
    return {"message": "CIFAR-100 Inference API is running"}

@app.get("/login/github")
def login_github():
    redirect_url = (
        "https://github.com/login/oauth/authorize"
        f"?client_id={CLIENT_ID}&scope=read:user user:email"
    )

    return RedirectResponse(redirect_url)

@app.get('/auth/callback')
async def auth_callback(code: str):
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
    token_data = token_resp.json()
    access_token = token_data.get("access_token")

    return RedirectResponse(f"http://localhost:3000/auth/success?token={access_token}")    

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model_loader.model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "device": str(model_loader.device) if model_loader.device else "unknown"
    }