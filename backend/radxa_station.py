import asyncio
import websockets
import json
import subprocess
import sys
import tty
import termios
import os
import base64

# --- CONFIGURATION ---
LAPTOP_IP = "192.168.137.91"
WS_URL = f"ws://{LAPTOP_IP}:8000/ws"


def get_key_blocking():
    """Captures keypresses."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return ch


async def keyboard_listener(command_queue, speech_queue):
    """Listens for keyboard input."""
    loop = asyncio.get_running_loop()

    while True:
        key = await loop.run_in_executor(None, get_key_blocking)
        key = key.lower()

        if key == 'q':
            os._exit(0)

        if key in ['1', '2', '3']:
            mode = {'1': "OCR", '2': "OBJECT", '3': "CURRENCY"}[key]

            await speech_queue.put(f"Switching to {mode}")
            await command_queue.put(f"MODE:{mode}")


async def speech_worker(speech_queue):
    """Speaks text using espeak."""
    print("[SPEAKER]: Worker started.")
    loop = asyncio.get_running_loop()

    while True:
        text = await speech_queue.get()

        if text:
            try:
                # Use executor to avoid blocking the main async loop
                await loop.run_in_executor(None, 
                    lambda: subprocess.run(['espeak', '-s', '170', '-a', '100', text], 
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            except Exception as e:
                print(f"Speech error: {e}")

        speech_queue.task_done()


async def send_commands(websocket, command_queue):
    """Send commands to server."""
    while True:
        cmd = await command_queue.get()
        await websocket.send(cmd)

        print(f"\n[CLIENT]: Sent {cmd}")

        command_queue.task_done()


async def receive_data(websocket, speech_queue):
    """Receive AI responses and status."""
    while True:
        message_str = await websocket.recv()
        data = json.loads(message_str)

        msg_type = data.get("type")

        if msg_type == "speech_word":
            word = data.get("content", "").strip()
            if word:
                await speech_queue.put(word)

        elif msg_type == "speech_end":
            pass

        elif msg_type == "frame":
            # We don't decode image on Radxa (not needed)
            llm_status = "BUSY" if data.get("llm_busy") else "IDLE"

            print(
                f"\r[STATUS] Mode: {data['mode']:<10} | LLM: {llm_status:<5} | Connected: OK",
                end="",
                flush=True
            )


async def main():
    print("=== RADXA SMART GLASS STATION (Streaming AI) ===")
    print(f"[SYSTEM]: Connecting to {WS_URL}...")

    command_queue = asyncio.Queue()
    speech_queue = asyncio.Queue()

    asyncio.create_task(speech_worker(speech_queue))
    asyncio.create_task(keyboard_listener(command_queue, speech_queue))

    while True:
        try:
            async with websockets.connect(WS_URL) as websocket:

                print("[SYSTEM]: Connected.")

                await asyncio.gather(
                    send_commands(websocket, command_queue),
                    receive_data(websocket, speech_queue)
                )

        except websockets.ConnectionClosed:
            print("\n[ERR]: Connection lost. Retrying in 3 seconds...")
            await asyncio.sleep(3)

        except Exception as e:
            print(f"\n[ERR]: {e}")
            await asyncio.sleep(3)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")