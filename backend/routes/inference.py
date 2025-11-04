from fastapi import APIRouter, File, UploadFile, HTTPException, Query
import logging
from typing import Dict, Any, List
import asyncio
from models.model_loader import model_loader
from utils.preprocessing import preprocess_image
from concurrent.futures import ThreadPoolExecutor
from cache import prediction_cache
# from caption.caption_generator import generate_caption
from PIL import Image
import io
from redis_cache.job_manager import JobStatus, job_manager
from job_queue.queue import get_job_queue
from workers.process_manager import get_worker_manager
from time import time


logger = logging.getLogger(__name__)

router = APIRouter()

executor =  ThreadPoolExecutor(max_workers=4)

async def run_inference(image_tensor):
    """Run Inference in thread_pool"""

    loop = asyncio.get_event_loop()

    return await loop.run_in_executor(executor, model_loader.predict, image_tensor, 5)

# async def generation_caption_async(image_bytes):
#     """Generate caption in thread_pool"""
    
#     try:
#         loop = asyncio.get_event_loop()

#         image = Image.open(io.BytesIO(image_bytes))

#         basic_caption = await loop.run_in_executor(executor, generate_caption, image)

#         logger.info(f"Generated caption: {basic_caption}")

#         return basic_caption
#     except Exception as e:
#         logger.error(f"Caption generation error: {e}")
#         return "Caption generation failed"
    

async def process_batch_job(
    job_id: str, 
    image_bytes_list: List[bytes], 
    file_names: List[str], 
    generate_captions: bool = False
):
    """Process batch job and update Redis"""
    
    try:
        job_manager.update_job_status(job_id, JobStatus.IN_PROGRESS)
        job = job_manager.get_job(job_id)
        
        # Preprocess images
        image_tasks = [
            asyncio.get_event_loop().run_in_executor(
                executor, preprocess_image, img_bytes
            )
            for img_bytes in image_bytes_list
        ]
        image_tensors = await asyncio.gather(*image_tasks)
        
        # Get worker manager
        worker_manager = get_worker_manager()
        
        # Process each image
        for i, tensor in enumerate(image_tensors):
            # Get predictions (classification)
            cached = prediction_cache.get(tensor)
            if cached:
                predictions = cached
            else:
                predictions = await run_inference(tensor)
                prediction_cache.set(tensor, predictions)
            
            result = {
                "filename": str(file_names[i]),
                "predictions": [
                    {
                        "class": str(pred.get("class_name", "")),
                        "confidence": float(pred.get("confidence", 0.0))
                    }
                    for pred in predictions
                ],
                "cached": bool(cached is not None)
            }
            
            # ✅ Generate caption via WORKER PROCESS
            if generate_captions:
                # Submit to worker process (non-blocking)
                worker_manager.submit_caption_request(
                    job_id=f"{job_id}_{i}",
                    image_bytes=image_bytes_list[i],
                    streaming=False  # Or True if you want streaming
                )
                
                # Wait for response from worker
                timeout = 30  # 30 seconds
                start_time = time.time()
                caption_response = None
                
                while time.time() - start_time < timeout:
                    response = worker_manager.get_response(timeout=1.0)
                    if response and response.get("job_id") == f"{job_id}_{i}":
                        caption_response = response
                        break
                    await asyncio.sleep(0.1)
                
                if caption_response and caption_response.get("type") == "caption_complete":
                    caption = caption_response.get("caption", "Caption generation failed")
                    result["caption"] = caption
                    job.captions.append(caption)
                else:
                    result["caption"] = "Caption generation timed out"
            
            # Append result and update Redis
            job.results.append(result)
            job.processed_images = i + 1
            job_manager.update_job(job)
            
            logger.info(f"Job {job_id}: Processed {i+1}/{len(image_bytes_list)}")
        
        # Mark completed
        job_manager.update_job_status(job_id, JobStatus.COMPLETED)
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
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

        worker_manager = get_worker_manager()
        job_id = job_manager.create_job(total_images=len(valid_files))

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

            worker_manager.submit_caption_request(job_id, image_bytes, True)
        
        if not valid_files:
            raise HTTPException(status_code=400, detail="No valid image files provided")
        
        # submit the job to the queue
        queue = get_job_queue()
        await queue.submit(
            process_batch_job,
            job_id,
            image_bytes_list,
            file_names,
            generate_captions=True,
            priority=0
        )

        logger.info(f"Submitted batch job {job_id} with {len(valid_files)} images to the queue")

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

