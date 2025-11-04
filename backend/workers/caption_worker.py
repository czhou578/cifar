import multiprocessing as mp
import logging
import time
import traceback

# ✅ Import the class, not the convenience functions
from caption.caption_generator import StreamingCaptionGenerator

logger = logging.getLogger(__name__)


def caption_worker_process(
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    device: str = "cpu"
):
    """
    Worker process that runs caption model
    
    This runs in a completely separate process from FastAPI
    """
    
    mp.current_process().name = "CaptionWorker"
    logger.info(f"Caption worker starting (PID: {mp.current_process().pid})")
    
    # ✅ Create and load model in worker process
    try:
        generator = StreamingCaptionGenerator(preload=False)
        generator._ensure_loaded()  # Load model
        logger.info("✅ Caption worker ready")
        
    except Exception as e:
        logger.error(f"Failed to initialize worker: {e}")
        response_queue.put({
            "type": "error",
            "error": f"Worker initialization failed: {str(e)}"
        })
        return
    
    # Send ready signal
    response_queue.put({
        "type": "ready",
        "pid": mp.current_process().pid
    })
    
    # Main worker loop
    while True:
        try:
            # Wait for job
            try:
                message = request_queue.get(timeout=1.0)
            except:
                continue
            
            # Handle shutdown
            if message.get("type") == "shutdown":
                logger.info("Received shutdown signal")
                break
            
            # Handle heartbeat
            if message.get("type") == "heartbeat":
                response_queue.put({
                    "type": "heartbeat_ack",
                    "timestamp": time.time()
                })
                continue
            
            # Process caption request
            if message.get("type") == "caption_request":
                job_id = message.get("job_id")
                image_bytes = message.get("image_bytes")
                streaming = message.get("streaming", False)
                
                logger.info(f"Processing caption for job {job_id}")
                
                try:
                    if streaming:
                        # Stream tokens
                        response_queue.put({
                            "type": "caption_start",
                            "job_id": job_id
                        })
                        
                        full_caption = ""
                        
                        # ✅ Use the generator directly
                        for token in generator.generate_caption_stream(image_bytes):
                            full_caption += token
                            response_queue.put({
                                "type": "caption_token",
                                "job_id": job_id,
                                "token": token,
                                "partial": full_caption
                            })
                        
                        response_queue.put({
                            "type": "caption_complete",
                            "job_id": job_id,
                            "caption": full_caption
                        })
                    else:
                        # Generate full caption
                        caption = ""
                        for token in generator.generate_caption_stream(image_bytes):
                            caption += token
                        
                        response_queue.put({
                            "type": "caption_complete",
                            "job_id": job_id,
                            "caption": caption
                        })
                    
                    logger.info(f"Caption completed for job {job_id}")
                    
                except Exception as e:
                    logger.error(f"Caption generation error: {e}")
                    response_queue.put({
                        "type": "caption_error",
                        "job_id": job_id,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
            
        except KeyboardInterrupt:
            logger.info("Worker interrupted")
            break
            
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
    
    # Cleanup
    logger.info("Shutting down caption worker...")
    generator.unload_model()
    logger.info("Caption worker stopped")