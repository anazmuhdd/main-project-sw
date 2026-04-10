"""
=============================================================================
 PIPER TTS WORKER — REFERENCE IMPLEMENTATION
=============================================================================
 Source: board_client.py  (main board-side client)
 Copied: 2026-04-10
 Status: ✅ CONFIRMED WORKING (verified in logs, two clean responses spoken)

 PURPOSE OF THIS FILE
 --------------------
 This file is a REFERENCE COPY of the Piper TTS worker that is embedded
 inside board_client.py.  Keep it here so that:
   1. Future agents can restore the correct logic if board_client.py changes.
   2. Developers can understand how the Producer-Consumer TTS pipeline works
      without reading the full client file.

 DO NOT run this file directly — it depends on globals (logger, tts_queue,
 PIPER_EXE, PIPER_MODEL, PIPER_SAMPLE_RATE) defined in board_client.py.

=============================================================================
 HOW IT WORKS — ARCHITECTURE
=============================================================================

  ┌─────────────┐   put(text)    ┌─────────────┐  stdin write  ┌───────────┐
  │  Main Loop  │ ─────────────► │  tts_queue  │ ────────────► │  Piper    │
  │ (producer)  │   put(SIGNAL)  │  asyncio Q  │               │  process  │
  └─────────────┘                └─────────────┘               └─────┬─────┘
                                                                      │ stdout (raw PCM)
                                                               ┌──────▼──────┐
                                                               │  pipe_audio │
                                                               │  (task)     │
                                                               └──────┬──────┘
                                                                      │ stdin write
                                                               ┌──────▼──────┐
                                                               │    aplay    │
                                                               │  (speaker)  │
                                                               └─────────────┘

  Key design principle:
    • Piper and aplay are started ONCE and stay alive for the entire session.
    • No cold-start penalty after the initial ONNX warmup (~1.4s first sentence).
    • Sync is done via byte counting, NOT process wait() — because aplay
      consumes a buffer, not a file, so wait() would never return.

=============================================================================
 SYNC PROTOCOL (3-STEP, RACE-CONDITION FREE)
=============================================================================

  When the main loop has finished accumulating the full LLM response it puts:
    tts_queue.put(full_text)       ← the sentence to speak
    tts_queue.put("SIGNAL_READY")  ← the sync marker

  The worker processes them in order:
    Step 0 — text item: write to Piper stdin → audio flows to aplay
    Step 1 — SIGNAL_READY: gate here until bytes_piped > 0
              (Piper takes ~1.4s to start producing the first byte —
               the old bug was releasing the gate before this happened)
    Step 2 — wait for bytes_piped to stop growing (300ms stable)
              → Piper has finished generating the whole sentence
    Step 3 — compute finish_time = stream_start_time + (bytes_piped / BPS)
              sleep until that moment (+150ms buffer)
              → speaker has physically finished

  After Step 3, send {"type": "ready"} to the server to trigger next frame.

=============================================================================
 CONFIGURATION CONSTANTS (defined in board_client.py)
=============================================================================

  BASE_PROJECT_DIR  = "/home/radxa/Project/main-project-sw"
  PIPER_EXE         = f"{BASE_PROJECT_DIR}/ven/bin/piper"
  PIPER_MODEL       = f"{BASE_PROJECT_DIR}/backend/tts/models/en_GB-alba-medium.onnx"
  PIPER_SAMPLE_RATE = 22050   (read from .onnx.json automatically)
  BYTES_PER_SEC     = PIPER_SAMPLE_RATE * 2   # 16-bit mono = 44100 B/s

=============================================================================
"""

import asyncio
import time
import os
import json


# ---------------------------------------------------------------------------
# Helper — reads sample_rate from the model's companion .json file
# ---------------------------------------------------------------------------
def get_sample_rate(model_path: str) -> int:
    json_path = model_path + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                return data.get("audio", {}).get("sample_rate", 22050)
        except Exception:
            pass
    return 22050


# ---------------------------------------------------------------------------
# TTS Worker — paste this function exactly into board_client.py
# ---------------------------------------------------------------------------
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
    bytes_piped = 0
    stream_start_time = None

    async def pipe_audio():
        """Real-time bridge: Piper stdout → aplay stdin. Updates byte counters."""
        nonlocal bytes_piped, stream_start_time
        chunk_n = 0
        try:
            while True:
                chunk = await piper_proc.stdout.read(4096)
                if not chunk:
                    break
                if stream_start_time is None:
                    stream_start_time = time.time()
                    logger.debug("[TTS►PIPE] First audio chunk received — aplay stream started")
                if aplay_proc.stdin:
                    aplay_proc.stdin.write(chunk)
                    await aplay_proc.stdin.drain()
                    bytes_piped += len(chunk)
                    chunk_n += 1
                    logger.debug(
                        f"[TTS►PIPE] chunk#{chunk_n} +{len(chunk)}B  "
                        f"total={bytes_piped}B  "
                        f"~{bytes_piped/BYTES_PER_SEC:.2f}s audio queued to aplay"
                    )
        except Exception as e:
            logger.error(f"[TTS►PIPE ERROR] {e}")

    asyncio.create_task(pipe_audio())

    # ── Warm-up: prime the ONNX engine so first real narration has no lag ──
    logger.info("TTS Worker: Warming up Piper (priming ONNX engine)...")
    piper_proc.stdin.write(b" \n")
    await piper_proc.stdin.drain()
    await asyncio.sleep(2.0)       # Wait for warmup audio to play through
    stream_start_time = None       # Reset — warmup doesn't count
    bytes_piped = 0
    logger.info("TTS Worker: Ready. Streaming mode active.")

    try:
        while True:
            text = await tts_queue.get()

            if text == "SIGNAL_READY":
                logger.debug("[TTS►SYNC] SIGNAL_READY received — waiting for Piper to START producing audio...")

                # ── STEP 1: Wait until Piper has actually started producing audio ──
                # FIX for race condition:
                # SIGNAL_READY can arrive before Piper generates any bytes.
                # We gate here until bytes_piped > 0 (up to 8s timeout).
                wait_start = time.time()
                while bytes_piped == 0:
                    if time.time() - wait_start > 8.0:
                        logger.warning("[TTS►SYNC] Timeout waiting for Piper audio — no bytes after 8s")
                        break
                    await asyncio.sleep(0.05)

                logger.debug(f"[TTS►SYNC] Piper started — initial bytes: {bytes_piped}B (waited {time.time()-wait_start:.2f}s)")

                # ── STEP 2: Wait for Piper to STOP producing (byte count stable) ──
                stable_count = 0
                prev_bytes = bytes_piped
                while stable_count < 3:  # 300ms of no new bytes = generation done
                    await asyncio.sleep(0.1)
                    if bytes_piped == prev_bytes:
                        stable_count += 1
                    else:
                        stable_count = 0
                        prev_bytes = bytes_piped

                logger.debug(f"[TTS►SYNC] Piper settled — total audio piped: {bytes_piped}B (~{bytes_piped/BYTES_PER_SEC:.2f}s)")

                # ── STEP 3: Wait until aplay physically finishes playing ──────────
                if stream_start_time is not None and bytes_piped > 0:
                    finish_time = stream_start_time + (bytes_piped / BYTES_PER_SEC)
                    wait_for = finish_time - time.time() + 0.15
                    if wait_for > 0:
                        logger.info(
                            f"[TTS►SYNC] Waiting {wait_for:.2f}s for speaker to finish  "
                            f"(bytes={bytes_piped}  duration={bytes_piped/BYTES_PER_SEC:.2f}s)"
                        )
                        await asyncio.sleep(wait_for)
                    else:
                        logger.debug(f"[TTS►SYNC] aplay already done (wait_for={wait_for:.2f}s)")

                logger.info("[TTS►SYNC] Speaker finished. Resetting counters.")
                # ── Reset for next narration ───────────────────────────────
                stream_start_time = None
                bytes_piped = 0

                tts_queue.task_done()
                continue

            # ── Normal text — write to persistent Piper immediately ────────
            if text and text.strip():
                logger.info(f"[TTS►PIPER] Sending text ({len(text.split())} words, {len(text)} chars): '{text[:80]}{'...' if len(text)>80 else ''}'")
                piper_proc.stdin.write((text + "\n").encode("utf-8"))
                await piper_proc.stdin.drain()
                logger.debug("[TTS►PIPER] Text written to Piper stdin")

            tts_queue.task_done()

    except Exception as e:
        logger.error(f"TTS Worker Loop Error: {e}")
    finally:
        for proc in [piper_proc, aplay_proc]:
            try:
                proc.terminate()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# How the main loop feeds the worker (from board_client.py board_main())
# ---------------------------------------------------------------------------
"""
USAGE IN MAIN LOOP
==================

1. Start the worker as a task (done once after WebSocket connects):

    tts_task = asyncio.create_task(tts_worker())

2. On receiving each LLM text chunk, accumulate it:

    tts_buffer += content

3. When 'done' status arrives from server, queue for TTS and sync:

    if tts_buffer.strip():
        logger.info(f"[TTS►QUEUE] Queuing full response for TTS")
        tts_queue.put_nowait(tts_buffer.strip())
    tts_buffer = ""

    # SIGNAL_READY tells the worker to wait for speech to finish
    tts_queue.put_nowait("SIGNAL_READY")
    await tts_queue.join()           # blocks main loop until speech is done

    # Now safe to tell server we're ready for the next frame
    await websocket.send(json.dumps({"type": "ready"}))
"""
