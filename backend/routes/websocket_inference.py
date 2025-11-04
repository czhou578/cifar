from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import base64
import logging
import time
import uuid

# ✅ Import worker manager instead of caption generator
from workers.process_manager import get_worker_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/caption")
async def websocket_caption(websocket: WebSocket):
    """
    WebSocket endpoint for streaming caption generation
    
    Flow:
    1. Client sends image via WebSocket
    2. FastAPI submits to worker process
    3. Worker generates caption and sends tokens back
    4. FastAPI streams tokens to client via WebSocket
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    try:
        # Get worker manager
        worker_manager = get_worker_manager()
        
        while True:
            # Receive data from client
            data = await websocket.receive_json()

            if data.get("type") == "generate_caption":
                # Decode image
                image_bytes = base64.b64decode(data["image_base64"])
                filename = data.get("filename", "unknown")
                
                # Generate unique job ID for this caption request
                job_id = str(uuid.uuid4())
                
                logger.info(f"Caption request received: {filename} (job_id: {job_id})")
                
                # Send start notification to client
                await websocket.send_json({
                    "type": "caption_start",
                    "filename": filename,
                    "job_id": job_id
                })
                
                try:
                    # ✅ Submit to worker process (non-blocking)
                    worker_manager.submit_caption_request(
                        job_id=job_id,
                        image_bytes=image_bytes,
                        streaming=True  # Request streaming tokens
                    )
                    
                    logger.info(f"Caption job {job_id} submitted to worker")
                    
                    # ✅ Monitor response queue for this job's results
                    full_caption = ""
                    timeout = 30  # 30 seconds timeout
                    start_time = time.time()
                    caption_started = False
                    
                    while time.time() - start_time < timeout:
                        # Check for response from worker (non-blocking)
                        response = worker_manager.get_response(timeout=0.5)
                        
                        if response and response.get("job_id") == job_id:
                            response_type = response.get("type")
                            
                            if response_type == "caption_start":
                                caption_started = True
                                logger.info(f"Caption generation started for {job_id}")
                            
                            elif response_type == "caption_token":
                                # Stream token to client
                                token = response.get("token", "")
                                full_caption = response.get("partial", full_caption)
                                
                                await websocket.send_json({
                                    "type": "caption_token",
                                    "token": token,
                                    "partial": full_caption,
                                    "filename": filename,
                                    "job_id": job_id
                                })
                                
                                # Add slight delay for smoother streaming
                                await asyncio.sleep(0.05)
                            
                            elif response_type == "caption_complete":
                                # Final caption received
                                full_caption = response.get("caption", full_caption)
                                
                                logger.info(f"Caption complete for {job_id}: {full_caption}")
                                
                                await websocket.send_json({
                                    "type": "caption_complete",
                                    "caption": full_caption,
                                    "filename": filename,
                                    "job_id": job_id
                                })
                                
                                break  # Done with this caption
                            
                            elif response_type == "caption_error":
                                # Error occurred in worker
                                error = response.get("error", "Unknown error")
                                logger.error(f"Caption error for {job_id}: {error}")
                                
                                await websocket.send_json({
                                    "type": "caption_error",
                                    "error": error,
                                    "filename": filename,
                                    "job_id": job_id
                                })
                                
                                break
                        
                        # Small sleep to prevent busy-waiting
                        await asyncio.sleep(0.1)
                    
                    # Check if we timed out
                    if time.time() - start_time >= timeout:
                        logger.error(f"Caption generation timed out for {job_id}")
                        await websocket.send_json({
                            "type": "caption_error",
                            "error": "Caption generation timed out",
                            "filename": filename,
                            "job_id": job_id
                        })
                
                except Exception as e:
                    logger.error(f"Error processing caption request: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "caption_error",
                        "error": str(e),
                        "filename": filename
                    })
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.close()
        except:
            pass