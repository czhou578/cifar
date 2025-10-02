from fastapi import APIRouter, File, UploadFile, HTTPException, Query
import logging
from typing import Dict, Any, List
import asyncio
from models.model_loader import model_loader
from utils.preprocessing import preprocess_image
from concurrent.futures import ThreadPoolExecutor
from cache import prediction_cache
import time


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

        file.file.seek(0, 2)  # Seek to end to get file size
        file_size = file.file.tell()
        file.file.seek(0)     # Seek back to beginning

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
                "predictions": cached_predictions,  # Fix: Use cached_predictions instead of predictions
                "top_prediction": cached_predictions[0] if cached_predictions else None,      
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


@router.post("/predict-batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of top predictions to return")
) -> Dict[str, Any]:
    """
    Predict CIFAR-100 classes for multiple uploaded images in a single request
    """
    start_time = time.time()
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        # Validate and read all files
        valid_files = []
        file_names = []
        
        for file in files:
            # Validate content type
            if not file.content_type or not file.content_type.startswith("image/"):
                logger.warning(f"Skipping non-image file: {file.filename}")
                continue
            
            # Validate file size
            file.file.seek(0, 2)
            file_size = file.file.tell()
            file.file.seek(0)
            
            if file_size > 10 * 1024 * 1024:
                logger.warning(f"Skipping large file: {file.filename}")
                continue
            
            valid_files.append(file)
            file_names.append(file.filename)
        
        if not valid_files:
            raise HTTPException(status_code=400, detail="No valid image files provided")
        
        # Process all images in parallel
        image_tasks = []
        for file in valid_files:
            image_bytes = await file.read()
            task = asyncio.get_event_loop().run_in_executor(
                executor, preprocess_image, image_bytes
            )
            image_tasks.append(task)
        
        # Wait for all preprocessing to complete
        image_tensors = await asyncio.gather(*image_tasks)
        
        # Check cache and separate into cached and uncached
        cached_results = []
        uncached_indices = []
        uncached_tensors = []
        
        for i, tensor in enumerate(image_tensors):
            cached_predictions = prediction_cache.get(tensor)
            if cached_predictions:
                cached_results.append({
                    "index": i,
                    "filename": file_names[i],
                    "predictions": cached_predictions,
                    "top_prediction": cached_predictions[0] if cached_predictions else None,
                    "cached": True
                })
            else:
                uncached_indices.append(i)
                uncached_tensors.append(tensor)
        
        # Run inference on uncached images
        uncached_results = []
        if uncached_tensors:
            inference_tasks = []
            for tensor in uncached_tensors:
                task = asyncio.get_event_loop().run_in_executor(
                    executor, model_loader.predict, tensor, top_k
                )
                inference_tasks.append(task)
            
            predictions_list = await asyncio.gather(*inference_tasks)
            
            # Cache results and format response
            for i, predictions in enumerate(predictions_list):
                original_index = uncached_indices[i]
                tensor = uncached_tensors[i]
                
                # Cache the predictions
                prediction_cache.set(tensor, predictions)
                
                uncached_results.append({
                    "index": original_index,
                    "filename": file_names[original_index],
                    "predictions": predictions,
                    "top_prediction": predictions[0] if predictions else None,
                    "cached": False
                })
        
        # Combine and sort results by original index
        all_results = cached_results + uncached_results
        all_results.sort(key=lambda x: x["index"])
        
        # Remove index from final response
        for result in all_results:
            del result["index"]
        
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "total_images": len(files),
            "processed_images": len(all_results),
            "cache_hits": len(cached_results),
            "cache_misses": len(uncached_results),
            "processing_time": round(processing_time, 3),
            "results": all_results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
