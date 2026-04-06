import subprocess
import os
import time
import json

# --- DEFAULT SETTINGS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_MODEL = os.path.join(BASE_DIR, "models/en_GB-alba-medium.onnx")
# -------------------------

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

def speak(text, output_file="piper_output.wav", play_live=True, stream=True):
    """
    Generates speech using Piper and optionally plays it via aplay.
    """
    try:
        if not os.path.exists(VOICE_MODEL):
            print(f"Error: Model not found at {VOICE_MODEL}")
            return False

        output_path = os.path.join(BASE_DIR, output_file)
        
        # Piper command: echo "TEXT" | piper --model MODEL --output_file OUTPUT
        piper_path = "/home/radxa/Project/main-project-sw/ven/bin/piper"
        sample_rate = get_sample_rate(VOICE_MODEL)
        
        if stream and play_live:
            print("Streaming live to audio device...")
            # Use --output-raw and pipe to aplay for zero latency
            piper_cmd = [
                piper_path,
                "--model", VOICE_MODEL,
                "--output-raw"
            ]
            aplay_cmd = [
                "aplay",
                "-r", str(sample_rate),
                "-f", "S16_LE",
                "-t", "raw"
            ]
            
            p_piper = subprocess.Popen(
                piper_cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            p_aplay = subprocess.Popen(
                aplay_cmd, 
                stdin=p_piper.stdout
            )
            
            # Allow p_piper.stdout to be closed in this process after it is used by p_aplay
            p_piper.stdout.close()
            
            start_time = time.time()
            # Send text to piper
            if isinstance(text, str):
                text = text.encode('utf-8')
            
            p_piper.stdin.write(text)
            p_piper.stdin.close()
            
            # Wait for aplay to finish first
            p_aplay.wait()
            
            # Check for errors in piper
            err = p_piper.stderr.read()
            p_piper.wait()
            if p_piper.returncode != 0:
                print(f"Piper error: {err.decode('utf-8', 'replace')}")
            
            end_time = time.time()
            print(f"Total stream duration: {end_time - start_time:.4f} seconds")
            return p_piper.returncode == 0




        # Fallback to generation-then-playback mode if not streaming
        command = [
            piper_path, 
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
        
        start_time = time.time()
        stdout, stderr = process.communicate(input=text)
        end_time = time.time()
        generation_time = end_time - start_time

        if process.returncode == 0 and os.path.exists(output_path):
            print(f"DONE: Audio generated at {output_path}")
            print(f"Time Taken to Generate: {generation_time:.4f} seconds")
            
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
