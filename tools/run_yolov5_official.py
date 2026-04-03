import subprocess
import os
import sys
import cv2
import time

def run_segmentation():
    # 1. Paths
    python_exe = os.path.join('..', 'myenv', 'Scripts', 'python.exe')
    predict_script = os.path.join('yolov5', 'segment', 'predict.py')
    model_path = os.path.join('models', 'yolov5_currency.pt')
    source_video = 'currency_video.mp4'
    project_name = 'runs/predict-seg'
    run_name = 'currency_inference'

    # 2. Check existence
    if not os.path.exists(predict_script):
        print(f"❌ Error: YOLOv5 repo script not found at {predict_script}")
        return
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return

    # 3. Build Command
    # Use absolute paths and ensure they are quoted correctly for Windows with spaces
    command = [
        f'"{python_exe}"',
        f'"{predict_script}"',
        '--weights', f'"{model_path}"',
        '--source', f'"{source_video}"',
        '--imgsz', '640',
        '--conf-thres', '0.25',
        '--project', f'"{project_name}"',
        '--name', f'"{run_name}"',
        '--exist-ok'
    ]

    cmd_str = " ".join(command)
    print("=" * 50)
    print("🚀 Running Official YOLOv5 Segmentation Inference")
    print("-" * 50)
    print(f"📦 Model: {model_path}")
    print(f"🎞️ Video: {source_video}")
    print(f"⏳ Executing command: {cmd_str}")
    print("-" * 50)

    try:
        # Run command using shell=True to handle the quoted command string correctly on Windows
        process = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode == 0:
            output_dir = os.path.join(project_name, run_name)
            print("-" * 50)
            print("✅ SUCCESS!")
            print(f"📂 Results are saved in: {os.path.abspath(output_dir)}")
            
            # --- VIDEO FIX STEP ---
            video_result = os.path.join(output_dir, source_video)
            if os.path.exists(video_result):
                print("-" * 50)
                print("🔧 Fixing video playback compatibility...")
                fixed_video = os.path.join(output_dir, "currency_segmented_FIXED.mp4")
                
                cap = cv2.VideoCapture(video_result)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Use a very safe codec for Windows (XVID in .mp4 or MJPG)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(fixed_video, fourcc, fps, (w, h))
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        out.write(frame)
                    
                    cap.release()
                    out.release()
                    print(f"✨ Playable video generated: {os.path.basename(fixed_video)}")
                    print(f"💡 Please open 'currency_segmented_FIXED.mp4' to check the results.")
            # ----------------------
            print("=" * 50)
        else:
            print(f"❌ Error: Process finished with return code {process.returncode}")
            
    except Exception as e:
        print(f"❌ Failed to run inference script: {e}")

if __name__ == '__main__':
    run_segmentation()
