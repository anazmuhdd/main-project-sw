import cv2
import json
import asyncio
import base64
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO
import uvicorn
import threading
import torch
import time

app = FastAPI()

# --- AI MODELS ---
print("--- [LAPTOP SERVER]: Initializing Models ---")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLO models
model_obj = YOLO("yolo26s.pt").to(device)    # Standard Objects
model_curr = YOLO("yolo26_seg.pt").to(device) 
current_mode = "OBJECT"

# LLM State
llm_busy = False
LLM_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:4b"

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[LAPTOP]: Radxa connected.")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[LAPTOP]: Radxa disconnected.")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global current_mode
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("MODE:"):
                current_mode = data.split(":")[1]
                print(f"[LAPTOP]: Mode switched to {current_mode}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def process_llm(labels, mode):
    """Calls Ollama LLM and streams words to Radxa."""
    global llm_busy
    if not labels:
        return
        
    llm_busy = True
    print(f"[LLM]: Processing labels: {labels}")
    
    if mode == "OBJECT":
        prompt = f"You are a helpful visual assistant for a blind person. Based on the camera, I see: {', '.join(labels)}. Briefly describe the scene in one natural, short sentence for the user. Be direct and helpful."
    elif mode == "CURRENCY":
        prompt = f"The user is holding currency notes: {', '.join(labels)}. Tell them exactly what they have in one short, clear sentence."
    else:
        llm_busy = False
        return

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", LLM_URL, 
                                   json={"model": MODEL_NAME, "prompt": prompt, "stream": True, "think": False}) as response:
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        word = chunk.get("response", "")
                        if word:
                            # Send word to Radxa immediately
                            await manager.broadcast(json.dumps({"type": "speech_word", "content": word}))
                            
                # Signal end of speech
                await manager.broadcast(json.dumps({"type": "speech_end"}))
    except Exception as e:
        print(f"[LLM ERROR]: {e}")
        await manager.broadcast(json.dumps({"type": "speech_word", "content": "Error communicating with intelligence model."}))
        await manager.broadcast(json.dumps({"type": "speech_end"}))
    finally:
        # Prevent immediate re-triggering
        await asyncio.sleep(3.0) 
        llm_busy = False
        print("[LLM]: Ready for next request.")

async def vision_loop():
    """Main loop for camera, YOLO, and UI."""
    global current_mode, llm_busy
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERR]: Could not open camera.")
        return

    print("[LAPTOP]: Vision loop started. Displaying monitor...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        labels = []
        # AI Processing
        if current_mode == "OBJECT":
            results = model_obj.predict(frame, conf=0.6, verbose=False)
            labels = [model_obj.names[int(b.cls[0])] for b in results[0].boxes]
            frame = results[0].plot() # Draw on frame
                
        elif current_mode == "CURRENCY":
            results = model_curr.predict(frame, conf=0.7, verbose=False)
            labels = [model_curr.names[int(b.cls[0])] for b in results[0].boxes]
            frame = results[0].plot()
                
        elif current_mode == "OCR":
            cv2.putText(frame, "OCR Mode: Coming Soon", (50, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            labels = []

        # Trigger LLM if we see something and aren't busy
        if labels and not llm_busy:
            unique_labels = list(set(labels))
            asyncio.create_task(process_llm(unique_labels, current_mode))

        # 2. Laptop-Side Visual Feedback
        cv2.putText(frame, f"Mode: {current_mode} | LLM Busy: {llm_busy}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Smart Glass - Laptop Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 3. Broadcast status/image to Radxa
        # We send frames at a lower frequency/resolution to Radxa to save bandwidth
        # since the user wants display on laptop primarily.
        resized = cv2.resize(frame, (320, 240))
        _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 30])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        data = {
            "type": "frame",
            "mode": current_mode,
            "image": jpg_as_text,
            "llm_busy": llm_busy
        }
        await manager.broadcast(json.dumps(data))
        
        # Small sleep to yield
        await asyncio.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

def start_async_logic():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(vision_loop())

if __name__ == "__main__":
    # Run vision loop in a thread
    threading.Thread(target=start_async_logic, daemon=True).start()
    
    print("[LAPTOP]: Starting FastAPI server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
