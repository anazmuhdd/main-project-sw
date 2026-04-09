from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from backend.modules.currency import CurrencyModule
from backend.modules.objects import ObjectModule
from backend.modules.ocr_module import OCRModule
from backend.modules.llm import LLMModule

import numpy as np
import cv2
import json
import os
import asyncio
import time
import logging
import logging.handlers

app = FastAPI()

import sys
import os

# Add relevant directories to PATH for model imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(os.path.join(project_root, "yolov5"))
sys.path.append(project_root)

# ── Logging Setup ────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(project_root, "logs", "backend")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")

log_fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s"
date_fmt = "%Y-%m-%d %H:%M:%S"

# 1. Create Handlers
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO) # Clean console
stream_handler.setFormatter(logging.Formatter(log_fmt, date_fmt))

file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG) # Full history
file_handler.setFormatter(logging.Formatter(log_fmt, date_fmt))

# 2. Configure Root Logger
logger = logging.getLogger("Server")
logger.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

logger.info(f"=== Server Starting — logs → {LOG_FILE} ===")

# ── Configuration and Paths ───────────────────────────────────────────────────
MODEL_PATH_V5 = "models/yolov5_currency.pt"
MODEL_PATH_V26_CURRENCY = "models/yolo26_seg.pt"
MODEL_PATH_V26_OBJECTS = "models/yolo26s.pt"
MODEL_PATH_V5_OBJECTS = "yolov5s.pt"
NVIDIA_API_KEY = "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"

# ── Pre-load Models ───────────────────────────────────────────────────────────
logger.info("Pre-loading ObjectDetection model...")
objects = ObjectModule(MODEL_PATH_V26_OBJECTS, MODEL_PATH_V5_OBJECTS)
logger.info("Pre-loading Currency model...")
currency = CurrencyModule(MODEL_PATH_V5, MODEL_PATH_V26_CURRENCY)
logger.info("Initializing LLM module...")
llm = LLMModule(use_nvidia=False, nvidia_key=NVIDIA_API_KEY)
logger.info("All models loaded. Server ready.")

ocr = None  # Lazy-loaded on first OCR request

# ── Global State ──────────────────────────────────────────────────────────────
last_detection_state = None
last_ai_processed_time = 0


@app.websocket("/vision")
async def vision_stream(websocket: WebSocket):
    global last_detection_state, last_ai_processed_time
    client_addr = websocket.client
    await websocket.accept()
    logger.info(f"[CONNECT] Board connected from {client_addr}")

    current_mode = "ObjectDetection"
    try:
        cv2.namedWindow("Raw Board Feed", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detection View", cv2.WINDOW_NORMAL)
    except Exception:
        pass

    latest_frame_data = {"data": None, "mode_idx": 0}
    frame_ready_event = asyncio.Event()
    board_ready_event = asyncio.Event()
    board_ready_event.set()

    # ── Counters / stats per session ─────────────────────────────────────────
    session_start = time.time()
    frames_received = 0
    frames_processed = 0
    llm_triggers = 0

    async def frame_receiver():
        """Continuously drains the websocket; keeps only the latest frame."""
        nonlocal frames_received
        try:
            while True:
                raw_data = await websocket.receive()

                if "bytes" in raw_data:
                    frames_received += 1
                    data = raw_data["bytes"]
                    latest_frame_data["mode_idx"] = data[0]
                    latest_frame_data["data"] = data[1:]
                    frame_ready_event.set()
                    logger.debug(
                        f"[FRAME►RX] frame#{frames_received} "
                        f"size={len(data)-1}B mode_idx={data[0]}"
                    )

                    # Update RAW window
                    try:
                        frame_data = np.frombuffer(data[1:], dtype=np.uint8)
                        raw_frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
                        if raw_frame is not None:
                            cv2.imshow("Raw Board Feed", raw_frame)
                            cv2.waitKey(1)
                    except Exception:
                        pass

                elif "text" in raw_data:
                    msg = json.loads(raw_data["text"])
                    if msg.get("type") == "ready":
                        elapsed = time.time() - session_start
                        logger.info(
                            f"[READY◄RX] Board sent READY  "
                            f"(session_up={elapsed:.1f}s  "
                            f"frames_rx={frames_received}  "
                            f"llm_calls={llm_triggers})"
                        )
                        board_ready_event.set()

        except WebSocketDisconnect:
            logger.warning(f"[DISCONNECT] Board disconnected from {client_addr}")
        except Exception as e:
            logger.error(f"[RX ERROR] {e}")

    receiver_task = asyncio.create_task(frame_receiver())
    logger.info("[RECEIVER] frame_receiver task started")

    try:
        while not receiver_task.done():
            # Wait for a new frame (raw feed always updated by receiver task)
            try:
                await asyncio.wait_for(frame_ready_event.wait(), timeout=0.1)
                frame_ready_event.clear()
            except asyncio.TimeoutError:
                continue

            # Skip AI branch while board is still speaking
            if not board_ready_event.is_set():
                logger.debug("[SKIP] Board not ready — skipping AI pass")
                continue

            t0 = time.time()

            mode_idx = latest_frame_data["mode_idx"]
            raw_bytes = latest_frame_data["data"]
            if raw_bytes is None:
                continue

            if mode_idx == 0:   current_mode = "ObjectDetection"
            elif mode_idx == 1: current_mode = "Currency"
            elif mode_idx == 2: current_mode = "OCR"

            frame_data = np.frombuffer(raw_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("[DECODE] Failed to decode JPEG frame — skipping")
                continue

            t_decode = time.time()
            logger.debug(f"[DECODE] Frame decoded in {(t_decode-t0)*1000:.1f}ms  mode={current_mode}")

            # ── Detection / Processing ────────────────────────────────────
            current_state = None
            result_prompt = None
            raw_detections = []

            if current_mode == "ObjectDetection":
                detections, raw_detections = objects.analyze_scene(frame)
                current_state = sorted(detections)
                t_infer = time.time()
                logger.info(
                    f"[DETECT] {current_mode}  "
                    f"detections={detections if detections else '[]'}  "
                    f"infer={( t_infer - t_decode)*1000:.1f}ms"
                )
                if detections:
                    items_str = ", ".join(detections)
                    result_prompt = f"""
        ACT AS A SENSORY SYSTEM. Your only job is to narrate the provided camera data. 
        Input Data: {items_str}
        
        Rules:
        - Describe only the items in the Input Data.
        - Be natural and spatial (left, right, front).
        - DO NOT say you are an AI.
        - DO NOT say you have no eyes.
        - Keep it to 1-2 short sentences maximum.
        
        System Narration:"""
                else:
                    result_prompt = None

            elif current_mode == "Currency":
                desc, total, raw_detections = currency.detect_and_sum(frame)
                current_state = desc if total > 0 else None
                result_prompt = currency.get_llm_prompt(desc) if total > 0 else None
                t_infer = time.time()
                logger.info(
                    f"[DETECT] Currency  total={total}  "
                    f"desc='{desc}'  infer={(t_infer-t_decode)*1000:.1f}ms"
                )

            elif current_mode == "OCR":
                global ocr
                if ocr is None:
                    logger.info("[OCR] Initializing OCR module (first use)...")
                    ocr = OCRModule()
                    logger.info("[OCR] OCR module ready")
                text = ocr.perform_ocr(raw_bytes)
                current_state = text
                result_prompt = ocr.get_llm_prompt(text)
                t_infer = time.time()
                logger.info(
                    f"[DETECT] OCR  chars={len(text)}  "
                    f"infer={(t_infer-t_decode)*1000:.1f}ms"
                )

            # ── Draw detections on Detection View window ──────────────────
            try:
                for det in raw_detections:
                    bbox, label, conf = det["bbox"], det["label"], det["conf"]
                    if conf >= 0.50:
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (bbox[0], bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Detection View", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                pass

            frames_processed += 1

            # ── LLM Trigger Decision ──────────────────────────────────────
            now = time.time()
            state_changed = (current_state != last_detection_state)
            time_elapsed = now - last_ai_processed_time

            if result_prompt and (state_changed or time_elapsed > 60.0):
                trigger_reason = "state_changed" if state_changed else "60s_reminder"
                llm_triggers += 1
                board_ready_event.clear()

                logger.info(
                    f"[LLM►TX] Triggering LLM #{llm_triggers}  "
                    f"reason={trigger_reason}  "
                    f"since_last={time_elapsed:.1f}s  "
                    f"state='{current_state}'"
                )
                logger.debug(f"[LLM►PROMPT]\n{result_prompt}")

                last_detection_state = current_state
                last_ai_processed_time = now

                t_llm_start = time.time()
                full_response = ""
                chunk_count = 0

                for chunk in llm.generate_streaming_response(result_prompt):
                    await websocket.send_text(json.dumps({"type": "text", "content": chunk}))
                    full_response += chunk
                    chunk_count += 1
                    logger.debug(f"[LLM►TX] chunk#{chunk_count} len={len(chunk)} '{chunk}'")

                await websocket.send_text(json.dumps({"type": "status", "content": "done"}))

                t_llm_end = time.time()
                logger.info(
                    f"[LLM►DONE] Response streamed  "
                    f"chunks={chunk_count}  "
                    f"total_chars={len(full_response)}  "
                    f"llm_time={(t_llm_end - t_llm_start)*1000:.0f}ms"
                )
                logger.info(f"[LLM►RESPONSE] '{full_response.strip()}'")

            else:
                skip_reason = "board_not_ready" if not board_ready_event.is_set() else (
                    "state_unchanged" if not state_changed else "no_detections"
                )
                logger.debug(f"[SKIP] No LLM trigger  reason={skip_reason}")
                await websocket.send_text(json.dumps({"type": "heartbeat"}))

            total_cycle = time.time() - t0
            logger.debug(f"[CYCLE] Total cycle time={total_cycle*1000:.1f}ms  frame#{frames_processed}")

    except Exception as e:
        logger.error(f"[MAIN LOOP ERROR] {e}", exc_info=True)
    finally:
        elapsed = time.time() - session_start
        logger.info(
            f"[SESSION END] uptime={elapsed:.1f}s  "
            f"frames_rx={frames_received}  "
            f"frames_processed={frames_processed}  "
            f"llm_triggers={llm_triggers}"
        )
        if not receiver_task.done():
            receiver_task.cancel()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
