import time
import cv2
import torch
import numpy as np
import yaml
from pathlib import Path
from executorch.runtime import Runtime

def preprocess(frame, imgsz):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz))
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)

def main(model_path, video_path, output_path=None):
    # 1. Load Metadata
    metadata_path = Path(model_path).parent / 'metadata.yaml'
    if not metadata_path.exists():
        print(f"Error: metadata.yaml not found at {metadata_path}")
        return

    with open(metadata_path, 'r') as f:
        meta = yaml.safe_load(f)
    
    imgsz = meta['imgsz'][0]
    task = meta['task']
    print(f"🎬 Testing {task} model: {model_path}")
    print(f"📹 Video: {video_path}")

    # 2. Load Model into ExecuTorch Runtime
    print(f"Loading model {model_path}...")
    runtime = Runtime.get()
    program = runtime.load_program(model_path)
    method = program.load_method('forward')

    # 3. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps_list = []
    frame_count = 0
    max_frames = 100 # Test on first 100 frames for speed benchmark

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        input_tensor = preprocess(frame, imgsz)

        # Inference
        t1 = time.time()
        outputs = method.execute([input_tensor])
        t2 = time.time()

        inf_time = (t2 - t1) * 1000
        fps = 1.0 / (t2 - t1)
        fps_list.append(fps)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Frame {frame_count}/{max_frames} | Inference: {inf_time:.2f}ms | FPS: {fps:.2f}")

    cap.release()
    
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    avg_lat = 1000 / avg_fps if avg_fps > 0 else 0
    
    print(f"\n--- Results for {Path(model_path).name} ---")
    print(f"Average Inference Latency: {avg_lat:.2f} ms")
    print(f"Average FPS: {avg_fps:.2f}")
    print("------------------------------------------")

if __name__ == "__main__":
    import sys
    # Example: python tools/video_inference_executorch.py <model> <video>
    if len(sys.argv) < 3:
        print("Usage: python video_inference_executorch.py <model.pte> <video.mp4>")
    else:
        main(sys.argv[1], sys.argv[2])
