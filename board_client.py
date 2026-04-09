import cv2
import asyncio
import websockets
import numpy as np
import time
import subprocess
import json
import os
from periphery import GPIO # Specific for Radxa periphery library
import logging
import coloredlogs

# Configure Logging
logger = logging.getLogger("BoardClient")
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s')



# --- Configuration ---
LAPTOP_IP = "172.20.10.2"
SERVER_URL = f"ws://{LAPTOP_IP}:8000/vision"

# USB Camera Configuration (as requested)
CAMERA_INDEX = "/dev/video0"
CAP_DRIVER = cv2.CAP_V4L2

# GPIO Configuration (from your gpio_test.py)
GPIO_CHIP = "/dev/gpiochip0"
GPIO_LINE = 108   # PD12

# Piper TTS Configuration
BASE_PROJECT_DIR = "/home/radxa/Project/main-project-sw"
PIPER_EXE = f"{BASE_PROJECT_DIR}/ven/bin/piper" 
PIPER_MODEL = f"{BASE_PROJECT_DIR}/backend/tts/models/en_GB-alba-medium.onnx"

def get_sample_rate(model_path):
    json_path = model_path + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                return data.get("audio", {}).get("sample_rate", 22050)
        except Exception:
            pass
    return 22050

PIPER_SAMPLE_RATE = get_sample_rate(PIPER_MODEL)

tts_queue = asyncio.Queue()

# Removed play_voice in favor of a persistent worker in tts_worker


async def tts_worker():
    """
    Perfect-sync TTS Worker.
    Architecture:
      1. Piper runs PERSISTENTLY (warm, never re-initialized).
      2. At startup, a warm-up call is made so first real speech has ZERO delay.
      3. For each sentence: send to Piper → collect audio bytes → play with a
         dedicated `aplay` that we AWAIT → exact sync, no estimation needed.
    """
    logger.info("TTS Worker: Starting persistent Piper engine...")
    
    piper_proc = await asyncio.create_subprocess_exec(
        PIPER_EXE, "--model", PIPER_MODEL, "--output-raw",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    )

    async def synthesize_and_play(text: str):
        """Send one sentence to Piper, collect audio, play it, and WAIT for completion."""
        piper_proc.stdin.write((text + "\n").encode("utf-8"))
        await piper_proc.stdin.drain()
        
        # Collect audio bytes for this sentence.
        # 400ms timeout is safe for sentences up to ~20 words.
        # The previous 80ms was too short, causing audio to bleed into the next sentence.
        audio_chunks = []
        while True:
            try:
                chunk = await asyncio.wait_for(piper_proc.stdout.read(16384), timeout=0.4)
                if chunk:
                    audio_chunks.append(chunk)
                else:
                    break
            except asyncio.TimeoutError:
                break  # Piper stopped producing — sentence is fully generated
        
        if not audio_chunks:
            return
        
        audio_data = b"".join(audio_chunks)
        
        # Play with a fresh aplay and AWAIT it — exact sync, no estimation needed.
        aplay_proc = await asyncio.create_subprocess_exec(
            "aplay", "-r", str(PIPER_SAMPLE_RATE), "-f", "S16_LE", "-t", "raw", "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        aplay_proc.stdin.write(audio_data)
        aplay_proc.stdin.close()
        await aplay_proc.wait()  # ← TRUE SYNC: blocks until speaker is physically done

    # --- WARM-UP: Pre-heat Piper so first real sentence has no startup lag ---
    logger.info("TTS Worker: Warming up Piper engine (first-request pre-heat)...")
    try:
        await asyncio.wait_for(synthesize_and_play(" "), timeout=8.0)
        logger.info("TTS Worker: Piper ready. Zero cold-start latency from now on.")
    except Exception as e:
        logger.warning(f"TTS warm-up failed (non-critical): {e}")

    try:
        while True:
            text = await tts_queue.get()
            
            if text == "SIGNAL_READY":
                # All sentences before this marker have been spoken & awaited.
                # Drain any stale text items that might have accumulated
                # (old responses that were superseded by a newer one).
                drained = 0
                while not tts_queue.empty():
                    try:
                        stale = tts_queue.get_nowait()
                        if stale != "SIGNAL_READY":  # Don't drain other markers
                            tts_queue.task_done()
                            drained += 1
                    except asyncio.QueueEmpty:
                        break
                if drained:
                    logger.debug(f"Drained {drained} stale sentence(s) from queue.")
                tts_queue.task_done()
                continue
            
            if text and text.strip():
                try:
                    await synthesize_and_play(text)
                except Exception as e:
                    logger.error(f"TTS Playback Error: {e}")
            
            tts_queue.task_done()

    except Exception as e:
        logger.error(f"TTS Worker Loop Error: {e}")
    finally:
        try:
            piper_proc.terminate()
        except Exception:
            pass


async def board_main():
    logger.info("Starting AI Vision Client...")

    
    # Initialize Camera
    logger.info(f"Initializing camera at {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX, CAP_DRIVER)
    if not cap.isOpened():
        logger.error(f"Could not open camera at {CAMERA_INDEX}")
        return
    logger.info("Camera initialized successfully.")

    # Initialize GPIO Button
    logger.info(f"Initializing GPIO pin {GPIO_LINE} on {GPIO_CHIP}...")
    try:
        button = GPIO(GPIO_CHIP, GPIO_LINE, "in")
        logger.info("GPIO initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing GPIO: {e}")
        return


    current_mode_idx = 0 # 0: ObjectDetection, 1: Currency, 2: OCR
    modes = ["ObjectDetection", "Currency", "OCR"]
    
    # Main Streaming Loop
    try:
        logger.info(f"Connecting to AI Server at {SERVER_URL}...")
        async with websockets.connect(SERVER_URL) as websocket:
            logger.info("Connection established.")
            
            # Start TTS Worker
            tts_task = asyncio.create_task(tts_worker())
            
            tts_buffer = ""
            while cap.isOpened():
                # 1. Read Button for Mode Switching
                btn_val = button.read()
                if btn_val != 0: 
                    current_mode_idx = (current_mode_idx + 1) % 3
                    logger.warning(f"*** MODE SWITCHED TO: {modes[current_mode_idx]} ***")
                    time.sleep(0.3) # Simple hardware debounce

                # 2. Capture Frame
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to grab frame.")
                    break
                
                frame = cv2.resize(frame, (320, 240))
                
                # 3. Encode Frame
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                
                # 4. Pack and Send
                payload = bytes([current_mode_idx]) + buffer.tobytes()
                await websocket.send(payload)
                
                # 4.5 Local Visualization (Requested)
                try:
                    cv2.imshow("Board Feed (Real-time)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    # Catch-all to prevent any GUI related error from killing the client
                    if not hasattr(board_main, "_gui_error_logged"):
                        logger.warning(f"Local GUI visualization failed: {e}. Continuing in headless mode...")
                        board_main._gui_error_logged = True
                    
                print(f"[Board] Streaming {modes[current_mode_idx]}...", end="\r")

                # 5. Handle Incoming AI Responses (Asynchronous)
                try:
                    # Non-blocking check for responses
                    response_raw = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                    data = json.loads(response_raw)
                    
                    if data['type'] == 'text':
                        content = data['content']
                        print(content, end="", flush=True)
                        
                        # Fluid TTS: Buffered to group words into natural phrases
                        tts_buffer += content
                        punctuation_marks = ".!?,;:"
                        split_idx = -1
                        for i, char in enumerate(tts_buffer):
                            if char in punctuation_marks:
                                split_idx = i
                                break
                        
                        if split_idx != -1:
                            phrase = tts_buffer[:split_idx+1]
                            tts_queue.put_nowait(phrase + " ") # Space for continuity
                            tts_buffer = tts_buffer[split_idx+1:].lstrip()
                        elif len(tts_buffer) > 40 and " " in tts_buffer:
                            last_space = tts_buffer.rfind(" ")
                            phrase = tts_buffer[:last_space]
                            tts_queue.put_nowait(phrase + " ")
                            tts_buffer = tts_buffer[last_space+1:]
                            
                    elif data['type'] == 'status' and data['content'] == 'done':
                        # Final flush: send remaining buffered text (with newline to flush Piper)
                        if tts_buffer.strip():
                            tts_queue.put_nowait(tts_buffer.strip() + "\n")
                        tts_buffer = ""
                        print("\n")
                        logger.info("Response ended. Waiting for speech to finish...")
                        
                        # Sync logic: Wait for TTS to finish before signaling READY
                        tts_queue.put_nowait("SIGNAL_READY")
                        await tts_queue.join()
                        await websocket.send(json.dumps({"type": "ready"}))
                        logger.info("Board is ready for next frame.")
                        
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    # Catch-all for websocket or parsing errors in this block
                    logger.error(f"Error handling response: {e}")

                
    except Exception as e:
        logger.error(f"Connection Error: {e}")
    finally:
        cap.release()
        button.close()
        # Drain the queue to stop further speech
        while not tts_queue.empty():
            try: tts_queue.get_nowait(); tts_queue.task_done()
            except: break
        if not tts_task.done():
            tts_task.cancel()
        logger.info("Client shut down.")


if __name__ == "__main__":
    try:
        asyncio.get_event_loop().run_until_complete(board_main())
    except KeyboardInterrupt:
        print("\n[Board] Stopped by user.")
