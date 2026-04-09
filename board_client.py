import cv2
import asyncio
import websockets
import numpy as np
import time
import subprocess
import json
import os

BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
from periphery import GPIO # Specific for Radxa periphery library
import logging
import logging.handlers
import coloredlogs

# Configure Logging
LOG_DIR = os.path.join(BASE_PROJECT_DIR, "logs", "board")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "board.log")

log_fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s"
date_fmt = "%Y-%m-%d %H:%M:%S"

# 1. Create Handlers
# Stream Handler (Terminal) - Set to INFO to avoid noise
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter(log_fmt, date_fmt))

# File Handler (Detailed) - Set to DEBUG for diagnostics
file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(log_fmt, date_fmt))

# 2. Configure Root Logger
logger = logging.getLogger("BoardClient")
logger.setLevel(logging.DEBUG) # Catch everything, handlers will filter
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# 3. Add Colors to Terminal Only
import coloredlogs
coloredlogs.install(level='INFO', logger=logger, fmt=log_fmt, datefmt=date_fmt, stream=sys.stdout)

logger.info(f"=== Board Client Starting — logs → {LOG_FILE} ===")



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
    Producer-Consumer TTS — Piper and aplay stay alive forever.

    Flow:
      Main loop puts text in tts_queue (producer).
      This worker writes to Piper stdin (consumer).
      A background pipe_audio task streams Piper stdout → aplay in real-time.
      First word plays in ~200ms. No cold-start delay after warmup.

    Sync (how we know when to send READY):
      We track bytes_piped (bytes sent to aplay) and stream_start_time.
      finish_time = stream_start_time + (bytes_piped / BYTES_PER_SEC)
      We wait until that precise moment before sending READY to the server.
    """
    BYTES_PER_SEC = PIPER_SAMPLE_RATE * 2  # 16-bit mono = 2 bytes/sample

    # ── Start Piper — stays alive forever ──────────────────────────────────
    piper_proc = await asyncio.create_subprocess_exec(
        PIPER_EXE, "--model", PIPER_MODEL, "--output-raw",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    )

    # ── Start aplay — stays alive forever, receives Piper's stream ─────────
    aplay_proc = await asyncio.create_subprocess_exec(
        "aplay", "-r", str(PIPER_SAMPLE_RATE), "-f", "S16_LE", "-t", "raw", "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL
    )

    # Shared state (updated by pipe_audio, read by SIGNAL_READY handler)
    total_bytes_piped = 0
    planned_finish_time = time.time()

    async def pipe_audio():
        """Real-time bridge: Piper stdout → aplay stdin. Updates global byte counter."""
        nonlocal total_bytes_piped, planned_finish_time
        chunk_n = 0
        try:
            while True:
                chunk = await piper_proc.stdout.read(4096)
                if not chunk:
                    break
                
                # Update the timeline: when will THIS chunk finish playing?
                duration = len(chunk) / BYTES_PER_SEC
                # audio appends to current queue. If queue empty, starts playing now.
                planned_finish_time = max(time.time(), planned_finish_time) + duration
                
                if aplay_proc.stdin:
                    aplay_proc.stdin.write(chunk)
                    await aplay_proc.stdin.drain()
                    total_bytes_piped += len(chunk)
                    chunk_n += 1
                    if chunk_n % 20 == 0: # Reduce log verbosity
                        logger.debug(
                            f"[TTS►PIPE] chunk#{chunk_n} piped.  "
                            f"Current audio queue end: +{planned_finish_time - time.time():.2f}s"
                        )
        except Exception as e:
            logger.error(f"[TTS►PIPE ERROR] {e}")

    asyncio.create_task(pipe_audio())

    # ── Warm-up: prime the ONNX engine so first real narration has no lag ──
    logger.info("TTS Worker: Warming up Piper (priming ONNX engine)...")
    piper_proc.stdin.write(b" \n")
    await piper_proc.stdin.drain()
    
    # Wait for warmup audio to appear in pipe_audio
    warmup_start = time.time()
    while total_bytes_piped == 0 and time.time() - warmup_start < 3.0:
        await asyncio.sleep(0.1)
    
    logger.info("TTS Worker: Ready. Streaming mode active.")

    try:
        while True:
            text = await tts_queue.get()

            if text == "SIGNAL_READY":
                logger.debug("[TTS►SYNC] SIGNAL_READY received — waiting for audio to finish...")
                
                # 1. Wait for Piper to finish generating for everything sent so far
                stable_count = 0
                prev_bytes = total_bytes_piped
                check_start = time.time()
                
                while stable_count < 3: # 300ms of stability
                    await asyncio.sleep(0.1)
                    if total_bytes_piped == prev_bytes:
                        stable_count += 1
                    else:
                        stable_count = 0
                        prev_bytes = total_bytes_piped
                    
                    # Safety timeout: if 5s passed and still no bytes, assume idle
                    if time.time() - check_start > 5.0:
                        break
                
                # 2. Precise sync: calculate when the current timeline finishes
                wait_for = planned_finish_time - time.time() + 0.1 # Small buffer
                
                if wait_for > 0:
                    logger.info(f"[TTS►SYNC] Waiting {wait_for:.2f}s for speaker to finish...")
                    await asyncio.sleep(wait_for)
                else:
                    logger.debug(f"[TTS►SYNC] Audio already played through (wait={wait_for:.2f}s)")
                
                logger.info("[TTS►SYNC] Narrator finished. Signal sent.")
                tts_queue.task_done()
                continue

            # ── Normal text piece ──
            if text and text.strip():
                # Stream to Piper immediately
                piper_proc.stdin.write((text + " ").encode("utf-8"))
                await piper_proc.stdin.drain()

            tts_queue.task_done()



    except Exception as e:
        logger.error(f"TTS Worker Loop Error: {e}")
    finally:
        for proc in [piper_proc, aplay_proc]:
            try:
                proc.terminate()
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
                t_send = time.time()
                payload = bytes([current_mode_idx]) + buffer.tobytes()
                await websocket.send(payload)
                logger.debug(
                    f"[FRAME►TX] frame sent  mode={modes[current_mode_idx]}  "
                    f"size={len(payload)}B  t={time.strftime('%H:%M:%S')}"
                )
                
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
                        logger.debug(f"[LLM◄RX] text chunk len={len(content)} '{content[:40]}'")
                        # Stream to TTS queue IMMEDIATELY for lowest latency
                        tts_queue.put_nowait(content)
                        # Keep buffer for logging purposes
                        tts_buffer += content

                    elif data['type'] == 'status' and data['content'] == 'done':
                        print("\n")
                        response_len = len(tts_buffer)
                        word_count = len(tts_buffer.split())
                        logger.info(
                            f"[LLM◄DONE] Full response received  "
                            f"chars={response_len}  words={word_count}  "
                            f"text='{tts_buffer.strip()[:100]}{'...' if response_len>100 else ''}'"
                        )

                        tts_buffer = ""

                        logger.debug("[SYNC] Queuing SIGNAL_READY and joining queue...")
                        t_sync_start = time.time()
                        tts_queue.put_nowait("SIGNAL_READY")
                        await tts_queue.join()
                        t_sync_end = time.time()

                        logger.info(
                            f"[SYNC] TTS complete  "
                            f"total_speech_time={(t_sync_end - t_sync_start):.2f}s"
                        )

                        await websocket.send(json.dumps({"type": "ready"}))
                        logger.info("[READY►TX] Sent READY to server")

                    elif data['type'] == 'heartbeat':
                        logger.debug("[HB◄RX] Heartbeat from server")
                        
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
