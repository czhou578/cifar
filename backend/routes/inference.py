from fastapi import APIRouter, File, UploadFile, HTTPException, Query
import logging
from typing import Dict, Any
import asyncio
from models.model_loader import model_loader
from utils.preprocessing import preprocess_image
import time
from concurrent.futures import ThreadPoolExecutor
from cache import prediction_cache


logger = logging.getLogger(__name__)

router = APIRouter()

executor =  ThreadPoolExecutor(max_workers=4)

async def run_inference(image_tensor):
    """Run Inference in thread_pool"""

    loop = asyncio.get_event_loop()

    return await loop.run_in_executor(executor, model_loader.predict, image_tensor, 5)

@router.post("/predict")
async def predict_image(file: UploadFile = File(...), top_k: int = Query(default=5, ge=1, le=20, description="Number of top predictions to return")) -> Dict[str, Any]:
    """
    Predict CIFAR-100 class for uploaded image
    """

    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        file.file.read(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large!")

        image_bytes = await file.read()
        image_tensor = await asyncio.get_event_loop().run_in_executor(executor, preprocess_image, image_bytes)

        cached_predictions = prediction_cache.get(image_tensor)

        if cached_predictions:
            logger.info(f"Cache hit for predictions. Fetching from cache")

            return {
                "status": "success",
                "filename": file.filename,
                "predictions": predictions,
                "top_prediction": predictions[0] if predictions else None,      
                "cached": True
            }

        predictions = await run_inference(image_tensor)

        prediction_cache.set(image_tensor, predictions)

        return {
            "status": "success",
            "filename": file.filename,
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None            
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction {e}")
        raise HTTPException(status_code=500, detail=f"Failure {e}")
