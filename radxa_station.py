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
import os

# --- CONFIGURATION ---
LAPTOP_IP = "172.20.10.x" # CHANGE THIS to your Laptop's IP
WS_URL = f"ws://{LAPTOP_IP}:8000/ws"

# GLOBAL STATE
command_queue = asyncio.Queue()
speech_queue = asyncio.Queue()

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
            # Put a simple notification in speech queue
            await speech_queue.put(f"Switching to {m}")
            await command_queue.put(f"MODE:{m}")

async def speech_worker():
    """Consumes words from the queue and speaks them using espeak."""
    print("[SPEAKER]: Worker started.")
    while True:
        text = await speech_queue.get()
        if text:
            # We use a slightly faster rate for better flow if words are streamed
            try:
                # Use -s 170 for decent speed
                subprocess.run(['espeak', '-s', '170', '-a', '100', text], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"Speech error: {e}")
        speech_queue.task_done()

async def main():
    print("=== RADXA SMART GLASS STATION (Streaming AI) ===")
    print(f"[SYSTEM]: Connecting to {WS_URL}...")
    
    # Start speech worker
    asyncio.create_task(speech_worker())

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
                    message_str = await websocket.recv()
                    data = json.loads(message_str)
                    
                    msg_type = data.get("type")
                    
                    if msg_type == "speech_word":
                        word = data.get("content", "").strip()
                        if word:
                            # print(word, end=" ", flush=True)
                            await speech_queue.put(word)
                            
                    elif msg_type == "speech_end":
                        # print("\n[LLM]: Response finished.")
                        pass # Could put a beep here
                        
                    elif msg_type == "frame":
                        # Handle Image (Optional display on Radxa)
                        img_data = base64.b64decode(data["image"])
                        nparr = np.frombuffer(img_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        llm_status = "BUSY" if data.get("llm_busy") else "IDLE"
                        print(f"\r[STATUS] Mode: {data['mode']:<10} | LLM: {llm_status:<5} | Connected: OK", end="", flush=True)
                        
                        if frame is not None and os.environ.get('DISPLAY'):
                            cv2.imshow("Smart Glass Feed", frame)
                            cv2.waitKey(1)

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
