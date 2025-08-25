from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/predict")
async def predict_image(file: UploadFile = File(...), top_k: int = Query(default=5, ge=1, le=20, description="Number of top predictions to return")) -> Dict[str, Any]:
    """
    Predict CIFAR-100 class for uploaded image
    """

    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_bytes = await file.read()
        image_tensor = preprocess(image_bytes)

        predictions = model_loader.predict(image_tensor, top_k=top_k)

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
