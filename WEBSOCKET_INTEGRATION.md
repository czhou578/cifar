# WebSocket Streaming Caption Integration

## 📚 What are WebSockets?

### Overview

WebSockets are a communication protocol that provides **full-duplex** (two-way) communication channels over a single TCP connection. Unlike traditional HTTP requests, WebSockets maintain a persistent connection between client and server, enabling real-time, bidirectional data exchange.

### How WebSockets Work

1. **Handshake Phase (HTTP Upgrade)**

   ```
   Client: "Hey server, can we upgrade to WebSocket?"
   HTTP GET /ws/caption
   Connection: Upgrade
   Upgrade: websocket

   Server: "Sure! WebSocket connection established ✅"
   HTTP 101 Switching Protocols
   ```

2. **Persistent Connection**

   - Connection stays open (unlike HTTP which closes after each request/response)
   - Both client and server can send messages anytime
   - No need to repeatedly establish connections

3. **Message Exchange**
   ```
   Client → Server: {"type": "generate_caption", "image": "..."}
   Server → Client: {"type": "caption_token", "token": "a"}
   Server → Client: {"type": "caption_token", "token": " photo"}
   Server → Client: {"type": "caption_token", "token": " of"}
   Server → Client: {"type": "caption_complete", "caption": "a photo of..."}
   ```

### HTTP vs WebSocket Comparison

| Feature        | HTTP                                             | WebSocket                     |
| -------------- | ------------------------------------------------ | ----------------------------- |
| **Connection** | Request → Response → Close                       | Persistent connection         |
| **Direction**  | Client initiates only                            | Both can initiate             |
| **Overhead**   | Headers sent with every request (~500-800 bytes) | Headers only on handshake     |
| **Latency**    | Higher (new connection each time)                | Lower (reuse connection)      |
| **Real-time**  | Polling required (inefficient)                   | True real-time push           |
| **Use Case**   | Traditional web pages, REST APIs                 | Chat, live updates, streaming |

### Why Use WebSockets?

✅ **Real-Time Applications**

- Live chat and messaging
- Multiplayer games
- Collaborative editing (Google Docs style)
- Stock tickers and live sports scores

✅ **Streaming Data**

- Progressive content loading
- Server-sent events (our caption streaming!)
- Live video/audio feeds
- IoT sensor data

✅ **Efficiency**

- Reduces network overhead (no repeated HTTP headers)
- Lower latency (no connection setup/teardown)
- Less server load (fewer connections)

✅ **Bidirectional Communication**

- Server can push updates without client asking
- Client can send updates anytime
- Perfect for interactive applications

### When to Use WebSockets

**✅ Use WebSockets when:**

- You need real-time, bidirectional communication
- Server needs to push updates to client frequently
- Low latency is critical
- You're streaming data progressively
- Building chat, notifications, or live feeds
- Client and server have ongoing conversation

**❌ Don't use WebSockets when:**

- Simple request/response is sufficient (use HTTP)
- Data updates infrequently (use polling or SSE)
- You need HTTP caching, proxies, or load balancing
- Dealing with sensitive operations (HTTP has better security tooling)
- Building a standard REST API

### Our Use Case: Streaming Captions

**Why WebSockets for caption generation?**

1. **Progressive Rendering**: Users see caption appear word-by-word (typewriter effect)

   - Better UX than waiting for complete caption
   - Feels more interactive and engaging

2. **Real-Time Feedback**: Server pushes tokens as they're generated

   - No polling needed
   - Instant updates

3. **Efficient**: Reuse connection for multiple images

   - Less overhead than multiple HTTP requests
   - Faster overall processing

4. **Bidirectional**:
   - Client sends images
   - Server streams back caption tokens
   - Natural fit for our workflow

**Alternative: Server-Sent Events (SSE)**

- One-way: Server → Client only
- Simpler than WebSocket
- But we also need Client → Server for images
- WebSocket is better fit

## ✅ What's Implemented

### Frontend Changes:

1. **Custom Hook**: `src/hooks/useWebSocket.ts`

   - Manages WebSocket connection lifecycle
   - Auto-reconnect on disconnect
   - Message handling and state management

2. **ImageClassifier Component Updates**:

   - WebSocket connection for streaming captions
   - Real-time token streaming display
   - Blinking cursor during caption generation
   - Connection status indicator (🟢 connected / 🔴 disconnected)
   - Hybrid mode: HTTP for predictions + WebSocket for captions

3. **CSS Enhancements**:
   - WebSocket status indicator styling
   - Blinking cursor animation
   - Smooth caption streaming effects

### How It Works:

1. **When user clicks "Classify" with captions enabled:**

   - Frontend sends predictions via HTTP (fast)
   - Opens WebSocket connection for caption streaming
   - Sends base64-encoded image via WebSocket

2. **Backend streams caption tokens:**

   - Each token appears in real-time
   - Cursor blinks while generating
   - Cursor disappears when complete

3. **User sees:**
   ```
   🎨 A whimsical scene featuring... |  ← Cursor blinks
   🎨 A whimsical scene featuring apples on a white background  ← Complete
   ```

## 🔧 Backend Setup Required

You need to create the WebSocket endpoint:

### File: `backend/routes/websocket_inference.py`

```python
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
    logger.info("✅ WebSocket client connected")

    try:
        while True:
            # Receive image data
            data = await websocket.receive_json()

            if data.get("type") == "generate_caption":
                logger.info(f"Generating caption for {data.get('filename')}")

                # Decode base64 image
                image_bytes = base64.b64decode(data["image"])
                filename = data.get("filename", "unknown.jpg")

                # Send start message
                await websocket.send_json({
                    "type": "caption_start",
                    "filename": filename
                })

                # Stream tokens
                loop = asyncio.get_event_loop()

                def stream_tokens():
                    tokens = []
                    for token in generate_caption_streaming(image_bytes):
                        tokens.append(token)
                    return tokens

                # Run in executor to avoid blocking
                tokens = await loop.run_in_executor(None, stream_tokens)

                full_caption = ""
                for token in tokens:
                    full_caption += token
                    await websocket.send_json({
                        "type": "caption_token",
                        "token": token,
                        "partial": full_caption,
                        "filename": filename
                    })
                    await asyncio.sleep(0.05)  # Smooth streaming effect

                # Send complete message
                await websocket.send_json({
                    "type": "caption_complete",
                    "caption": full_caption,
                    "filename": filename
                })

                logger.info(f"✅ Caption complete: {full_caption}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()
```

### Register the WebSocket route in `backend/main.py`:

```python
from routes import websocket_inference

app.include_router(websocket_inference.router, prefix="/api/v1", tags=["websocket"])
```

## 🚀 Testing

1. **Start Backend:**

   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. **Start Frontend:**

   ```bash
   cd frontend
   npm start
   ```

3. **Test Streaming:**
   - Upload an image
   - Enable "Generate creative captions ✨"
   - Click "Classify"
   - Watch captions appear word-by-word with blinking cursor!

## 🎨 Features

- ✅ Real-time token streaming
- ✅ Blinking cursor during generation
- ✅ Connection status indicator
- ✅ Auto-reconnect on disconnect
- ✅ Hybrid HTTP + WebSocket approach
- ✅ Smooth typewriter effect
- ✅ Error handling and fallbacks

## 📝 Message Protocol

**Client → Server:**

```json
{
  "type": "generate_caption",
  "image": "base64_encoded_image",
  "filename": "image.jpg"
}
```

**Server → Client:**

```json
// Start
{"type": "caption_start", "filename": "image.jpg"}

// Token stream
{"type": "caption_token", "token": " apples", "partial": "a photo of apples", "filename": "image.jpg"}

// Complete
{"type": "caption_complete", "caption": "a photo of apples on a white background", "filename": "image.jpg"}
```

## 🔄 Fallback Mode

If WebSocket fails, the app automatically falls back to HTTP batch processing with non-streaming captions.

---

**Status**: ✅ Frontend complete, Backend endpoint needed
