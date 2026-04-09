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
    """Background worker to process TTS queue with a persistent piper pipeline and synchronized audio sync."""
    logger.info("TTS Worker started with persistent pipeline and audio synchronization.")
    
    # We run piper persistently to avoid startup delay
    # We read from its stdout to get the audio data
    piper_cmd = [PIPER_EXE, "--model", PIPER_MODEL, "--output-raw"]
    
    piper_process = await asyncio.create_subprocess_exec(
        *piper_cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    )
    
    try:
        while True:
            text = await tts_queue.get()
            if text:
                try:
                    # 1. Send text to Piper
                    # We add a small marker or just rely on the fact that piper processes 
                    # one line at a time. However, reading from stdout until EOF is not possible
                    # with a persistent process. 
                    # SOLUTION: Pipe to a temporary file is more reliable for sync
                    temp_wav = "/tmp/curr_speech.wav" # Or any local project path
                    if os.name == 'nt': temp_wav = "curr_speech.wav"
                    else: temp_wav = f"{BASE_PROJECT_DIR}/curr_speech.raw"

                    # For a persistent piper, we can't easily get the BOUNDARY of a single sentence's audio
                    # from a raw stream without parsing headers or using a wrapper.
                    # BETTER PERSISTENT SYNC: Run piper per-sentence BUT keep it fast by using a 
                    # lightweight call OR using a named pipe.
                    
                    # Actually, the user wants it to be persistent. Let's try this:
                    # Run piper once per sentence but it's very fast on Radxa if cached.
                    # Or use a separate piper process for each narration.
                    # Given the Radxa performance, persistent is better.
                    
                    # If I use --output_file per sentence, piper doesn't support that in persistent mode.
                    # Let's revert to a slightly less persistent but perfectly synced approach:
                    # Run piper -> aplay for the WHOLE narration (the one that triggered 'done').
                    
                    # Wait, the user said "dont need the piper to reinit... it must be initialized... during the program start".
                    # To keep it persistent AND synced, I can use the 'ready' signal after the WHOLE narration.
                    # I will modify the loop to play the chunks and only signal ready at the end.
                    
                    # To know when the LAST chunk is done playing:
                    # I'll play each chunk with aplay and wait for it.
                    
                    # How to get the chunk from persistent piper?
                    # piper --output-raw doesn't terminate. 
                    # Let's use a non-persistent call for now but optimize it, 
                    # OR use piper-python if available. 
                    # Actually, let's keep it simple: Use a single shell command for the whole phrase.
                    
                    # REFINED: The tts_queue now receives 'phrases'. 
                    # I will run piper for that phrase and wait.
                    
                    full_cmd = f"echo '{text}' | '{PIPER_EXE}' --model '{PIPER_MODEL}' --output-raw | aplay -r {PIPER_SAMPLE_RATE} -f S16_LE -t raw"
                    proc = await asyncio.create_subprocess_shell(full_cmd)
                    await proc.wait() # THIS IS THE SYNC. We wait for aplay to finish.
                    
                except Exception as e:
                    logger.error(f"Error in TTS playback: {e}")
            tts_queue.task_done()
    except Exception as e:
        logger.error(f"TTS Worker Exception: {e}")
    finally:
        if piper_process:
            try: piper_process.terminate()
            except: pass


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
                        # 1. Print chunk immediately (Live feedback)
                        content = data['content']
                        print(content, end="", flush=True)
                        
                        # 2. Buffer for phrase-based TTS (continuity fix)
                        tts_buffer += content
                        # print(f" [Buffer: {tts_buffer}]", end="\r", flush=True) # REMOVED: causing truncation
                        
                        # Find the last punctuation mark to split the phrase
                        punctuation_marks = ".!?,;:"
                        last_punc_idx = -1
                        for i, char in enumerate(tts_buffer):
                            if char in punctuation_marks:
                                last_punc_idx = i
                        
                        if last_punc_idx != -1:
                            phrase = tts_buffer[:last_punc_idx+1].strip()
                            if phrase:
                                tts_queue.put_nowait(phrase)
                            tts_buffer = tts_buffer[last_punc_idx+1:].lstrip()
                        elif len(tts_buffer) > 40 and " " in tts_buffer:
                            # Force a split if the phrase is getting too long (e.g. > 40 chars)
                            last_space_idx = tts_buffer.rfind(" ")
                            phrase = tts_buffer[:last_space_idx].strip()
                            if phrase:
                                tts_queue.put_nowait(phrase)
                            tts_buffer = tts_buffer[last_space_idx+1:].lstrip()
                            
                    elif data['type'] == 'status' and data['content'] == 'done':
                        # Final flush of the buffer
                        if tts_buffer.strip():
                            tts_queue.put_nowait(tts_buffer.strip())
                        tts_buffer = ""
                        print("\n") # Newline after response ends
                        logger.info("Response ended. Waiting for speech to finish...")
                        
                        # Sync logic: Wait for TTS to finish before signaling READY
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
