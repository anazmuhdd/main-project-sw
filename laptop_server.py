import cv2
import json
import asyncio
import base64
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO
import uvicorn
from contextlib import asynccontextmanager
import torch
import time

# --- LOGGING UTILITY ---
def log_stage(stage, message):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{stage.upper()}]: {message}")

# --- AI MODELS ---
log_stage("INIT", "Initializing Models...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_stage("INIT", f"Using device: {device}")

# Load YOLO models
model_obj = YOLO("yolo26s.pt").to(device)    # Standard Objects
model_curr = YOLO("yolo26_seg.pt").to(device) 
current_mode = "OBJECT"

# LLM State
llm_busy = False
LLM_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:4b"

# Global Event for Radxa Connection
radxa_connected_event = asyncio.Event()

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        log_stage("NETWORK", f"Radxa station connected. (Total: {len(self.active_connections)})")
        radxa_connected_event.set()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            log_stage("NETWORK", f"Radxa station disconnected. (Remaining: {len(self.active_connections)})")
            if not self.active_connections:
                radxa_connected_event.clear()

    async def broadcast(self, message: str):
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except Exception:
                    pass

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the vision task when the app starts
    vision_task = asyncio.create_task(vision_loop())
    yield
    # Cleanup
    vision_task.cancel()
    try:
        await vision_task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global current_mode
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("MODE:"):
                current_mode = data.split(":")[1]
                log_stage("CONTROL", f"Mode switched to {current_mode}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def process_llm(labels, mode):
    """Calls Ollama LLM and streams words to Radxa."""
    global llm_busy
    if not labels:
        return
        
    llm_busy = True
    log_stage("LLM-INPUT", f"Processing labels: {labels}")
    
    if mode == "OBJECT":
        prompt = f"You are a helpful visual assistant for a blind person. Based on the camera, I see: {', '.join(labels)}. Briefly describe the scene in one natural, short sentence for the user. Be direct and helpful."
    elif mode == "CURRENCY":
        prompt = f"The user is holding currency notes: {', '.join(labels)}. Tell them exactly what they have in one short, clear sentence."
    else:
        llm_busy = False
        return

    log_stage("LLM-SEND", f"Sending prompt to {MODEL_NAME}...")

    full_response = ""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", LLM_URL, 
                                   json={"model": MODEL_NAME, "prompt": prompt, "stream": True, "think": False}) as response:
                log_stage("LLM-STREAM", "Starting word stream to Radxa:")
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        word = chunk.get("response", "")
                        if word:
                            full_response += word
                            # Print word to laptop console (no newline for flow)
                            print(word, end="", flush=True)
                            # Send word to Radxa immediately
                            await manager.broadcast(json.dumps({"type": "speech_word", "content": word}))
                            
                print() # Newline after stream
                # Signal end of speech
                log_stage("LLM-DONE", f"Full Sentence: {full_response}")
                await manager.broadcast(json.dumps({"type": "speech_end"}))
    except Exception as e:
        log_stage("LLM-ERROR", f"Error during streaming: {e}")
        await manager.broadcast(json.dumps({"type": "speech_word", "content": "Error communicating with intelligence model."}))
        await manager.broadcast(json.dumps({"type": "speech_end"}))
    finally:
        # Prevent immediate re-triggering (Cooldown)
        await asyncio.sleep(4.0) 
        llm_busy = False
        log_stage("LLM-READY", "System ready for next detection.")

async def vision_loop():
    """Main loop for camera, YOLO, and UI."""
    global current_mode, llm_busy
    
    while True:
        log_stage("SYSTEM", "Waiting for Radxa connection to start camera...")
        await radxa_connected_event.wait()
        
        log_stage("CAMERA", "Radxa connected. Opening camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            log_stage("ERROR", "Could not open camera.")
            await asyncio.sleep(5)
            continue

        log_stage("VISION", "Processing started. Displaying monitor...")
        
        try:
            while radxa_connected_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    log_stage("ERROR", "Failed to grab frame.")
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
                    log_stage("YOLO", f"Detected: {unique_labels}")
                    asyncio.create_task(process_llm(unique_labels, current_mode))

                # 2. Laptop-Side Visual Feedback
                status_color = (0, 0, 255) if llm_busy else (0, 255, 0)
                cv2.putText(frame, f"Mode: {current_mode}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(frame, f"LLM: {'BUSY' if llm_busy else 'IDLE'}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                cv2.imshow("Smart Glass - Laptop Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    log_stage("SYSTEM", "Exit signal received.")
                    # In a real setup, we might not want to exit the whole server
                    break

                # 3. Broadcast status/image to Radxa
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
                
                # Yield to other async tasks
                await asyncio.sleep(0.05)
                
        finally:
            log_stage("CAMERA", "Closing camera and cleaning up...")
            cap.release()
            cv2.destroyAllWindows()
            if not radxa_connected_event.is_set():
                log_stage("SYSTEM", "Camera dormant until next connection.")

if __name__ == "__main__":
    log_stage("SYSTEM", "Starting FastAPI server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
