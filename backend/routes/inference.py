from fastapi import APIRouter, File, UploadFile, HTTPException, Query
import logging
from typing import Dict, Any, List
import asyncio
from models.model_loader import model_loader
from utils.preprocessing import preprocess_image
from concurrent.futures import ThreadPoolExecutor
from cache import prediction_cache
from caption.caption_generator import generate_caption
from PIL import Image
import io
from datetime import datetime
from redis_cache.job_manager import BatchJob, JobStatus, job_manager


logger = logging.getLogger(__name__)

router = APIRouter()

executor =  ThreadPoolExecutor(max_workers=4)

async def run_inference(image_tensor):
    """Run Inference in thread_pool"""

    loop = asyncio.get_event_loop()

    return await loop.run_in_executor(executor, model_loader.predict, image_tensor, 5)

async def generation_caption_async(image_bytes):
    """Generate caption in thread_pool"""
    
    try:
        loop = asyncio.get_event_loop()

        image = Image.open(io.BytesIO(image_bytes))

        basic_caption = await loop.run_in_executor(executor, generate_caption, image)

        logger.info(f"Generated caption: {basic_caption}")

        return basic_caption
    except Exception as e:
        logger.error(f"Caption generation error: {e}")
        return "Caption generation failed"
    

async def process_batch_job(job_id: str, image_bytes_list: List[bytes], file_names: List[str], generate_captions: bool = False):
    """
    Process batch job and accumulate results
    """

    logger.debug(f"Starting batch job {job_id} with {len(image_bytes_list)} images")
    try:
        # Mark as in progress
        job_manager.update_job_status(job_id, JobStatus.IN_PROGRESS)

        # Get the job object
        job = job_manager.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        # Preprocess all images concurrently
        image_tasks = []
        for image_bytes in image_bytes_list:
            task = asyncio.get_event_loop().run_in_executor(
                executor, preprocess_image, image_bytes
            )
            image_tasks.append(task)
        
        image_tensors = await asyncio.gather(*image_tasks)
        logger.debug(f"Preprocessed {len(image_tensors)} images for job {job_id}")

        # Process each image
        for i, tensor in enumerate(image_tensors):
            try:
                # Get predictions (check cache first)
                cached = prediction_cache.get(tensor)
                logger.debug(f"Cache lookup for {file_names[i]}: {'HIT' if cached else 'MISS'}")
                logger.debug(f"Cache content: {cached}")
                if cached:
                    predictions = cached
                else:
                    predictions = await run_inference(tensor)
                    logger.debug(f"Predictions for {file_names[i]}: {predictions}")
                    prediction_cache.set(tensor, predictions)

                # ✅ Build ONLY JSON-serializable result
                result = {
                    "filename": str(file_names[i]),  # Ensure string
                    "predictions": [
                        {
                            "class": str(pred.get("class_name", "")),
                            "confidence": float(pred.get("confidence", 0.0))
                        }
                        for pred in predictions
                    ],
                    "top_prediction": {
                        "class": str(predictions[0].get("class_name", "")) if predictions else None,
                        "confidence": float(predictions[0].get("confidence", 0.0)) if predictions else None
                    } if predictions else None,
                    "cached": bool(cached is not None)
                }

                # Generate caption if requested
                if generate_captions:
                    caption = await generation_caption_async(image_bytes_list[i])
                    result["caption"] = str(caption)  # Ensure string
                    job.captions.append(str(caption))

                # Append result to job
                job.results.append(result)
                job.processed_images = i + 1
                
                # Update job in Redis
                job_manager.update_job(job)
                
                logger.info(f"Job {job_id}: Processed {i+1}/{len(image_bytes_list)} - {file_names[i]}")

            except Exception as e:
                logger.error(f"Error processing image {file_names[i]}: {e}")
                # Add error result
                job.results.append({
                    "filename": str(file_names[i]),
                    "error": str(e)
                })
                job.processed_images = i + 1
                job_manager.update_job(job)

        # Mark as completed
        job_manager.update_job_status(job_id, JobStatus.COMPLETED)
        logger.info(f"Job {job_id} completed successfully with {len(job.results)} results")

    except Exception as e:
        logger.error(f"Error processing batch job {job_id}: {e}")
        job_manager.update_job_status(job_id, JobStatus.FAILED, error_message=str(e))

@router.post("/predict-batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of top predictions to return"),
) -> Dict[str, Any]:
    """
    Predict CIFAR-100 classes for multiple uploaded images in a single request
    """
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        # Validate and read all files
        valid_files = []
        file_names = []
        image_bytes_list = []
        
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
            
            # Read file once and store
            image_bytes = await file.read()
            image_bytes_list.append(image_bytes)
            valid_files.append(file)
            file_names.append(file.filename)
        
        if not valid_files:
            raise HTTPException(status_code=400, detail="No valid image files provided")
        
        job_id = job_manager.create_job(total_images=len(valid_files))

        asyncio.create_task(process_batch_job(job_id, image_bytes_list, file_names))

        return {
            "status": "accepted",
            "job_id": job_id,
            "total_images": len(valid_files),
            "message": "Batch processing started. Check job status for results.",
            "status_url": f"/api/v1/batch-status/{job_id}"
        }
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
    

@router.get("/batch-status/{job_id}")
async def get_batch_status(job_id: str) -> Dict[str, Any]:
    """
    Get batch job status and results
    
    Returns:
    - For IN_PROGRESS: progress percentage and partial results
    - For COMPLETED: all results
    - For FAILED: error message
    """
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Calculate progress
    progress = 0
    if job.total_images > 0:
        progress = round((job.processed_images / job.total_images) * 100, 2)
    
    response = {
        "job_id": job.job_id,
        "status": job.status.value,
        "total_images": job.total_images,
        "processed_images": job.processed_images,
        "progress": progress,
        "created_at": job.created_at,
        "updated_at": job.updated_at
    }
    
    # Include results based on status
    if job.status == JobStatus.COMPLETED:
        response["results"] = job.results  # All results
        response["captions"] = job.captions
        logger.info(f"Returning {len(job.results)} results for job {job_id}")
    elif job.status == JobStatus.IN_PROGRESS:
        # Optionally return partial results
        response["partial_results"] = job.results  # Results processed so far
    elif job.status == JobStatus.FAILED:
        response["error"] = job.error_message
    
    return response

@router.get("/batch-jobs/")
async def list_batch_jobs() -> Dict[str, Any]:
    """
    List all batch jobs.
    """
    try:
        jobs = job_manager.get_jobs()
        job_list = [job.to_dict() for job in jobs]

        return {
            "status": "success",
            "jobs": job_list
        }
    except Exception as e:
        logger.error(f"Error listing batch jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to list batch jobs")

