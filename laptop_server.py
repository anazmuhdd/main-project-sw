import cv2
import json
import asyncio
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO
import uvicorn
import threading

app = FastAPI()

# --- AI MODELS ---
print("--- [LAPTOP SERVER]: Initializing Models ---")
model_obj = YOLO("yolov8n.pt")    # Standard Objects
model_curr = YOLO("yolo26s.pt")  # Your Currency Model
current_mode = "OBJECT"

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[LAPTOP]: Radxa connected.")

    def disconnect(self, websocket: WebSocket):
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
            # Receive commands from Radxa
            data = await websocket.receive_text()
            if data.startswith("MODE:"):
                current_mode = data.split(":")[1]
                print(f"[LAPTOP]: Mode switched to {current_mode}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def get_frame():
    """Generator to capture frames and run AI."""
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        speech_text = None
        
        # 1. BRAIN: Core AI Processing
        if current_mode == "OBJECT":
            results = model_obj.predict(frame, conf=0.6, verbose=False)
            labels = [model_obj.names[int(b.cls[0])] for b in results[0].boxes]
            if labels:
                speech_text = f"Objects: {', '.join(set(labels))}"
                
        elif current_mode == "CURRENCY":
            results = model_curr.predict(frame, conf=0.7, verbose=False)
            notes = [model_curr.names[int(b.cls[0])] for b in results[0].boxes]
            if notes:
                speech_text = f"Notes: {', '.join(set(notes))}"
                
        elif current_mode == "OCR":
            speech_text = "OCR Mode active."

        # 2. Add visual feedback on the frame
        cv2.putText(frame, f"Mode: {current_mode}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 3. Compress and Encode
        resized = cv2.resize(frame, (640, 480))
        _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 40])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # 4. Package data
        data = {
            "mode": current_mode,
            "speech": speech_text,
            "image": jpg_as_text
        }
        
        yield json.dumps(data)
    cap.release()

async def stream_frames():
    """Streaming loop to broadcast to WebSocket."""
    # We use a thread-safe way to run the generator
    for frame_data in get_frame():
        await manager.broadcast(frame_data)
        await asyncio.sleep(0.05) # ~20 FPS

def start_stream_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stream_frames())

if __name__ == "__main__":
    # Run the vision streaming in a separate thread
    threading.Thread(target=start_stream_loop, daemon=True).start()
    
    # Start FastAPI Server
    print("[LAPTOP]: Starting FastAPI server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
