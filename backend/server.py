from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from backend.modules.currency import CurrencyModule
from backend.modules.objects import ObjectModule
from backend.modules.ocr_module import OCRModule
from backend.modules.llm import LLMModule

import numpy as np
import cv2
import json
import os

app = FastAPI()

import sys
import os

# Add relevant directories to PATH for model imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(os.path.join(project_root, "yolov5"))
sys.path.append(project_root)

# Configuration and Paths
MODEL_PATH_V5 = "models/yolov5_currency.pt"
MODEL_PATH_V26_CURRENCY = "models/yolo26_seg.pt"
MODEL_PATH_V26_OBJECTS = "models/yolo26s.pt"
MODEL_PATH_V5_OBJECTS = "yolov5s.pt" # Points to the pretrained weight file in the root
NVIDIA_API_KEY = "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"



# Initialize Models (PRE-LOADED for speed)
# print("[Server] Pre-loading Object Detection models...")
objects = ObjectModule(MODEL_PATH_V26_OBJECTS, MODEL_PATH_V5_OBJECTS)

# print("[Server] Pre-loading Currency models...")
currency = CurrencyModule(MODEL_PATH_V5, MODEL_PATH_V26_CURRENCY)

# print("[Server] Initializing LLM Orchestrator...")
llm = LLMModule(use_nvidia=False, nvidia_key=NVIDIA_API_KEY)

# LAZY-LOADED: OCR will be initialized inside the socket only when mode is requested
ocr = None

import time


# State tracking to avoid redundant AI calls
last_detection_state = None
last_ai_processed_time = 0
AI_DEBOUNCE_INTERVAL = 3.0 # At least 3 seconds between LLM calls unless the scene changes

@app.websocket("/vision")
async def vision_stream(websocket: WebSocket):
    global last_detection_state, last_ai_processed_time
    await websocket.accept()
    current_mode = "ObjectDetection"
    cv2.namedWindow("Server Stream", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            start_time = time.time()
            data = await websocket.receive_bytes()
            
            mode_idx = data[0]
            if mode_idx == 0: current_mode = "ObjectDetection"
            elif mode_idx == 1: current_mode = "Currency"
            elif mode_idx == 2: current_mode = "OCR"
            
            frame_data = np.frombuffer(data[1:], dtype=np.uint8)
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            
            preprocess_time = time.time() - start_time
            # print(f"\n[Server] New frame in mode: {current_mode} (Preprocess: {preprocess_time:.4f}s)")
            
            # Logic branch based on mode
            current_state = None
            result_prompt = ""
            raw_detections = []
            
            if current_mode == "ObjectDetection":
                detections, raw_detections = objects.analyze_scene(frame)
                current_state = sorted(detections)
                result_prompt = objects.get_llm_prompt(detections) if detections else None
                
            elif current_mode == "Currency":
                desc, total, raw_detections = currency.detect_and_sum(frame)
                current_state = desc if total > 0 else None
                result_prompt = currency.get_llm_prompt(desc) if total > 0 else None
                
            elif current_mode == "OCR":
                # Lazy-load OCR model on first call to save startup time
                global ocr
                if ocr is None:
                    print("[Server] Initializing OCR Module (first use)...")
                    ocr = OCRModule()
                
                text = ocr.perform_ocr(data[1:])
                current_state = text
                result_prompt = ocr.get_llm_prompt(text)

            # Draw detections on the frame for server-side visualization
            for det in raw_detections:
                bbox = det["bbox"]
                label = det["label"]
                conf = det["conf"]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Server Stream", frame)
            cv2.waitKey(1)

            inference_time = time.time() - start_time - preprocess_time
            # print(f"[Server] Inference complete: {inference_time:.4f}s")

            # DECISION: Should we trigger a new AI session (LLM + TTS)?
            now = time.time()
            if result_prompt and (current_state != last_detection_state or (now - last_ai_processed_time) > AI_DEBOUNCE_INTERVAL):
                print(f"\n[LLM Prompt]: {result_prompt}")
                print("[LLM Response]: ", end="", flush=True)
                last_detection_state = current_state
                last_ai_processed_time = now
                
                full_response = ""
                for chunk in llm.generate_streaming_response(result_prompt):
                    await websocket.send_text(json.dumps({"type": "text", "content": chunk}))
                    print(chunk, end="", flush=True)
                    full_response += chunk
                
                print("\n") # Newline after response
                await websocket.send_text(json.dumps({"type": "status", "content": "done"}))
            else:
                # Still send a heartbeat to keep the board socket alive
                await websocket.send_text(json.dumps({"type": "heartbeat"}))

    except WebSocketDisconnect:
        print("[Server] Board connection lost.")
    except Exception as e:
        print(f"[Server] Error: {e}")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
