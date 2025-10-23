from fastapi import APIRouter, WebSocket
from caption.caption_generator import generate_caption_streaming
import asyncio
import base64
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.websocket("/ws/caption")
async def websocket_caption(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "generate_caption":
                image_bytes = base64.b64decode(data["image_base64"])

                await websocket.send_json({"type": "caption_start"})

                loop = asyncio.get_event_loop()

                def stream_token():
                    tokens = []
                    for token in generate_caption_streaming(image_bytes):
                        tokens.append(token)
                    
                    return tokens
                
                tokens = await loop.run_in_executor(None, stream_token)

                full_caption = ""

                for token in tokens:
                    full_caption += token
                    await websocket.send_json({
                        "type": "caption_token",
                        "token": token,
                        "partial": full_caption,
                        "filename": data.get("filename", "")
                    })
                    await asyncio.sleep(0.1)
                
                logger.info(f"Completed caption: {full_caption}")
                
                await websocket.send_json({
                    "type": "caption_complete",
                    "caption": full_caption 
                })  # Simulate streaming delay
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()