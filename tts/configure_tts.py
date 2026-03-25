import pyttsx3
import os

# --- CONFIGURATION ---
VOICE_INDEX = 19       # English (Great Britain)
RATE = 220           
VOLUME = 0.5  
TEXT_TO_SPEAK = "This is a test recording to a wave file. Let's see if this plays better."
SAVE_TO_FILE = "test.wav" # Changed from .mp3 to .wav for better Linux/board compatibility
# ----------------------

def run_tts():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, SAVE_TO_FILE)
        
        engine = pyttsx3.init()
        
        # Select voice
        voices = engine.getProperty('voices')
        if VOICE_INDEX < len(voices):
            engine.setProperty('voice', voices[VOICE_INDEX].id)
            print(f"Using: {voices[VOICE_INDEX].name}")
        
        engine.setProperty('rate', RATE)
        engine.setProperty('volume', VOLUME)
        
        # IMPORTANT: On some Linux systems, you cannot do 'say' and 'save_to_file' 
        # in the same runAndWait() call effectively. 
        # We will ONLY save to file first to ensure the file is perfect.
        
        print(f"Saving audio to: {save_path} ...")
        if os.path.exists(save_path):
            os.remove(save_path)
            
        engine.save_to_file(TEXT_TO_SPEAK, save_path)
        engine.runAndWait() # This ensures the file is written
        
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            print(f"SUCCESS: File saved ({os.path.getsize(save_path)} bytes)")
            
            # Now we try to PLAY it live using the system's 'aplay' command
            # This is often more reliable on boards than pyttsx3's live 'say'
            print("\nAttempting to play live via 'aplay'...")
            os.system(f"aplay {save_path}")
        else:
            print("ERROR: File was not created properly.")

        engine.stop()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_tts()
