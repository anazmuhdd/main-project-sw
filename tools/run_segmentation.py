import cv2
import os
import time
import torch
import sys
from ultralytics import YOLO

# Add yolov5 repo to path
REPO_PATH = os.path.abspath('yolov5')
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

def run_inference():
    # Paths
    model_path = os.path.join('models', 'yolov5_currency.pt')
    video_path = 'currency_video.mp4'
    output_path = 'currency_segmented_output.mp4'

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Load model
    print(f"Loading model: {model_path}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Use YOLO from ultralytics
        model = YOLO(model_path)
        print("Model loaded successfully with Ultralytics.")
    except Exception as e:
        print(f"Warning: Could not load with Ultralytics directly: {e}")
        print("Attempting to use torch.hub as fallback...")
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            model.to(device)
            print("Model loaded successfully via Torch Hub.")
            is_ultralytics = False
        except Exception as e2:
            print(f"Error: Failed to load model: {e2}")
            return
    else:
        is_ultralytics = True

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path}...")
    print(f"Output will be saved to: {output_path}")

    frame_count = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if is_ultralytics:
                # Run inference with Ultralytics on GPU
                results = model.predict(frame, conf=0.25, device=device)
                annotated_frame = results[0].plot()
            else:
                # Run inference with YOLOv5 torch.hub
                results = model(frame)
                annotated_frame = results.render()[0]

            # Write the frame
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
    finally:
        # Cleanup
        cap.release()
        out.release()
    
    end_time = time.time()
    duration = end_time - start_time
    print("-" * 30)
    print(f"Finished! Total frames: {frame_count}")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Average FPS: {frame_count/duration:.2f}")
    print(f"Result saved to: {os.path.abspath(output_path)}")

if __name__ == '__main__':
    run_inference()
