import cv2
import asyncio
import websockets
import numpy as np
import time
import subprocess
import json
import os
from periphery import GPIO # Specific for Radxa periphery library

# --- Configuration ---
LAPTOP_IP = "192.168.1.5" # UPDATE THIS TO YOUR LAPTOP IP
SERVER_URL = f"ws://{LAPTOP_IP}:8000/vision"

# USB Camera Configuration (as requested)
CAMERA_INDEX = "/dev/video2"
CAP_DRIVER = cv2.CAP_V4L2

# GPIO Configuration (from your gpio_test.py)
GPIO_CHIP = "/dev/gpiochip0"
GPIO_LINE = 108   # PD12

# Piper TTS Configuration
PIPER_MODEL = "en_US-lessac-medium.onnx"
PIPER_EXE = "./piper/piper" # Path to your piper binary

async def board_main():
    print(f"[Board] Starting AI Vision Client...")
    
    # Initialize Camera
    print(f"[Board] Initializing camera at {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX, CAP_DRIVER)
    if not cap.isOpened():
        print(f"[Board] Error: Could not open camera at {CAMERA_INDEX}")
        return
    print("[Board] Camera initialized successfully.")

    # Initialize GPIO Button
    print(f"[Board] Initializing GPIO pin {GPIO_LINE} on {GPIO_CHIP}...")
    try:
        button = GPIO(GPIO_CHIP, GPIO_LINE, "in")
        print("[Board] GPIO initialized successfully.")
    except Exception as e:
        print(f"[Board] Error initializing GPIO: {e}")
        return

    current_mode_idx = 0 # 0: ObjectDetection, 1: Currency, 2: OCR
    modes = ["ObjectDetection", "Currency", "OCR"]
    
    # Main Streaming Loop
    try:
        print(f"[Board] Connecting to AI Server at {SERVER_URL}...")
        async with websockets.connect(SERVER_URL) as websocket:
            print("[Board] Connection established.")
            
            while cap.isOpened():
                frame_start = time.time()
                
                # 1. Read Button for Mode Switching
                btn_val = button.read()
                # Assuming button.read() returns 1 if pressed, 
                # but based on your test it could be inverted
                if btn_val != 0: 
                    current_mode_idx = (current_mode_idx + 1) % 3
                    print(f"\n[Board] *** MODE SWITCHED TO: {modes[current_mode_idx]} ***")
                    time.sleep(0.3) # Simple hardware debounce

                # 2. Capture Frame
                ret, frame = cap.read()
                if not ret:
                    print("[Board] Error: Failed to grab frame.")
                    break
                
                # Optional: Resize for speed if needed
                frame = cv2.resize(frame, (640, 480))
                
                # 3. Encode Frame
                encode_start = time.time()
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                encode_time = time.time() - encode_start
                
                # 4. Pack and Send
                # Protocol: First byte byte is mode_idx
                payload = bytes([current_mode_idx]) + buffer.tobytes()
                await websocket.send(payload)
                print(f"[Board] Sent frame (Encode: {encode_time:.4f}s)", end="\r")

                # 5. Handle Incoming AI Responses (Asynchronous)
                try:
                    # Non-blocking check for responses
                    response_raw = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    data = json.loads(response_raw)
                    
                    if data['type'] == 'text':
                        text = data['content']
                        print(f"\n[Board] Received Speech: {text}")
                        # Immediately pipe to Piper TTS
                        try:
                            # Start piper process
                            p = subprocess.Popen([PIPER_EXE, "--model", PIPER_MODEL, "--output_raw"],
                                                stdin=subprocess.PIPE,
                                                stdout=subprocess.DEVNULL)
                            p.communicate(input=text.encode())
                            print(f"[Board] Played audio chunk.")
                        except Exception as e:
                            print(f"[Board] TTS Player Error: {e}")
                            
                    elif data['type'] == 'status' and data['content'] == 'done':
                        print("[Board] Logic: End of response stream.")
                        
                except asyncio.TimeoutError:
                    # No response yet, continue streaming
                    pass
                
    except Exception as e:
        print(f"\n[Board] Connection Error: {e}")
    finally:
        cap.release()
        button.close()
        print("[Board] Client shut down.")

if __name__ == "__main__":
    try:
        asyncio.get_event_loop().run_until_complete(board_main())
    except KeyboardInterrupt:
        print("\n[Board] Stopped by user.")
