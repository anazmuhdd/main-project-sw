from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from .modules.currency import CurrencyModule
from .modules.objects import ObjectModule
from .modules.ocr_module import OCRModule
from .modules.llm import LLMModule
import numpy as np
import cv2
import json
import os

app = FastAPI()

# Configuration and Paths
MODEL_PATH_V5 = "models/yolov5_currency.pt"
MODEL_PATH_V26_CURRENCY = "models/yolo26_seg.pt"
MODEL_PATH_V26_OBJECTS = "models/yolo26s.pt"
MODEL_PATH_V5_OBJECTS = "backend/yolov5_object_detection"
NVIDIA_API_KEY = "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"

# Initialize Modules
currency = CurrencyModule(MODEL_PATH_V5, MODEL_PATH_V26_CURRENCY)
objects = ObjectModule(MODEL_PATH_V26_OBJECTS, MODEL_PATH_V5_OBJECTS)
ocr = OCRModule()
llm = LLMModule(use_nvidia=True, nvidia_key=NVIDIA_API_KEY)

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
            print(f"\n[Server] New frame in mode: {current_mode} (Preprocess: {preprocess_time:.4f}s)")
            
            # Logic branch based on mode
            current_state = None
            result_prompt = ""
            
            if current_mode == "ObjectDetection":
                detections = objects.analyze_scene(frame)
                current_state = sorted(detections)
                result_prompt = objects.get_llm_prompt(detections) if detections else None
                
            elif current_mode == "Currency":
                desc, total = currency.detect_and_sum(frame)
                current_state = desc if total > 0 else None
                result_prompt = currency.get_llm_prompt(desc) if total > 0 else None
                
            elif current_mode == "OCR":
                # For OCR, we usually trigger once, not stream
                text = ocr.perform_ocr(data[1:])
                current_state = text # OCR is always considered unique
                result_prompt = ocr.get_llm_prompt(text)

            inference_time = time.time() - start_time - preprocess_time
            print(f"[Server] Inference complete: {inference_time:.4f}s")

            # DECISION: Should we trigger a new AI session (LLM + TTS)?
            now = time.time()
            if result_prompt and (current_state != last_detection_state or (now - last_ai_processed_time) > AI_DEBOUNCE_INTERVAL):
                print(f"[Server] Scene changed or time limit reached. Calling LLM...")
                last_detection_state = current_state
                last_ai_processed_time = now
                
                llm_start_time = time.time()
                full_response = ""
                for chunk in llm.generate_streaming_response(result_prompt):
                    await websocket.send_text(json.dumps({"type": "text", "content": chunk}))
                    full_response += chunk
                
                llm_time = time.time() - llm_start_time
                print(f"[Server] LLM Response generated in: {llm_time:.4f}s")
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
