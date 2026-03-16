import asyncio
import websockets
import json
import cv2
import numpy as np
import base64
import subprocess
import time
import sys
import tty
import termios
import threading
import os

# --- CONFIGURATION ---
LAPTOP_IP = "172.20.10.x" # CHANGE THIS to your Laptop's IP
WS_URL = f"ws://{LAPTOP_IP}:8000/ws"
SPEECH_GAP = 3.0

# GLOBAL STATE
last_speak_time = 0
command_queue = asyncio.Queue()

def speak(text):
    """Speaks using espeak on the board with a rate limiter."""
    global last_speak_time
    if not text or len(text.strip()) < 2: return
    
    current_time = time.time()
    if (current_time - last_speak_time) < SPEECH_GAP:
        return
    
    print(f"\n[SPEAKER]: {text}")
    try:
        subprocess.run(['espeak', '-s', '160', '-a', '100', text], check=True)
        last_speak_time = time.time()
    except Exception as e:
        print(f"Speech error: {e}")

def get_key_blocking():
    """Captures keypresses in a thread-safe way."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

async def keyboard_listener():
    """Listens for keys and puts them in the queue."""
    loop = asyncio.get_event_loop()
    while True:
        key = await loop.run_in_executor(None, get_key_blocking)
        key = key.lower()
        if key == 'q':
            os._exit(0)
        if key in ['1', '2', '3']:
            m = {'1': "OCR", '2': "OBJECT", '3': "CURRENCY"}[key]
            speak(f"Requesting {m}")
            await command_queue.put(f"MODE:{m}")

async def main():
    print("=== RADXA SMART GLASS STATION (WebSockets) ===")
    print(f"[SYSTEM]: Connecting to {WS_URL}...")
    
    async for websocket in websockets.connect(WS_URL):
        try:
            # Task to send commands
            async def send_commands():
                while True:
                    cmd = await command_queue.get()
                    await websocket.send(cmd)
                    print(f"\n[CLIENT]: Sent {cmd}")

            # Task to receive data
            async def receive_data():
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # 1. Handle Audio
                    if data.get("speech"):
                        speak(data["speech"])
                    
                    # 2. Handle Image
                    img_data = base64.b64decode(data["image"])
                    nparr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        if os.environ.get('DISPLAY'):
                            cv2.imshow("Smart Glass Feed", frame)
                            cv2.waitKey(1)
                        else:
                            print(f"\r[STATUS] Mode: {data['mode']:<10} | Connected: OK", end="", flush=True)

            # Parallel execution
            await asyncio.gather(
                send_commands(),
                receive_data(),
                keyboard_listener()
            )
            
        except websockets.ConnectionClosed:
            print("\n[ERR]: Connection lost. Retrying in 3 seconds...")
            await asyncio.sleep(3)
            continue
        except Exception as e:
            print(f"\n[ERR]: {e}")
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
