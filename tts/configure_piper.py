import subprocess
import os

# --- DEFAULT SETTINGS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_MODEL = os.path.join(BASE_DIR, "models/en_GB-alba-medium.onnx")
# -------------------------

def speak(text, output_file="piper_output.wav", play_live=True):
    """
    Generates speech using Piper and optionally plays it via aplay.
    """
    try:
        if not os.path.exists(VOICE_MODEL):
            print(f"Error: Model not found at {VOICE_MODEL}")
            return False

        output_path = os.path.join(BASE_DIR, output_file)
        
        # Piper command: echo "TEXT" | piper --model MODEL --output_file OUTPUT
        command = [
            "piper", 
            "--model", VOICE_MODEL, 
            "--output_file", output_path
        ]
        
        process = subprocess.Popen(
            command, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=text)

        if process.returncode == 0 and os.path.exists(output_path):
            print(f"DONE: Audio generated at {output_path}")
            
            if play_live:
                print(f"Playing...")
                os.system(f"aplay {output_path}")
            return True
        else:
            print(f"ERROR: Piper failed (Exit code {process.returncode})")
            print(f"Stderr: {stderr}")
            return False

    except Exception as e:
        print(f"An error occurred in Piper: {e}")
        return False

if __name__ == "__main__":
    # Test execution
    test_text = "The Piper system is now modular and ready to be used in other scripts."
    speak(test_text)
