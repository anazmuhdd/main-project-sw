import cv2
from ultralytics import YOLO
import sys
import os
import argparse
import time
import torch

def main():
    parser = argparse.ArgumentParser(description="YOLO Currency Detection - Optimized for RTX 4050")
    parser.add_argument("--source", type=str, default=r'c:\Users\anasm\OneDrive\Documents\Main Project\main-project-sw\currency_video.mp4', 
                        help="Path to video file or '0' for webcam")
    parser.add_argument("--model", type=str, default=r'c:\Users\anasm\OneDrive\Documents\Main Project\main-project-sw\Currency_Model_partial.pt', 
                        help="Path to .pt model file")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    args = parser.parse_args()

    # 1. Path to model
    model_path = args.model
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # 2. Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 3. Load the YOLO model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    model.to(device)
    
    # 4. Open video source
    source = args.source
    if source.isdigit():
        source = int(source)
        print(f"Opening webcam source: {source}")
        is_webcam = True
    else:
        print(f"Opening video file: {source}")
        if not os.path.exists(source):
            print(f"Error: Video file not found at {source}")
            return
        is_webcam = False

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return

    # Get original video FPS to sync playback speed
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or is_webcam:
        # Default for webcams or if unknown (common is 30)
        target_fps = 30 
    else:
        target_fps = video_fps
    
    target_frame_time = 1.0 / target_fps
    print(f"Target Playback Speed: {target_fps:.2f} FPS")

    # Window settings
    window_name = "Currency Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Press 'q' to quit.")
    
    prev_time = time.time()
    
    while cap.isOpened():
        loop_start = time.time()
        
        success, frame = cap.read()
        if not success:
            if is_webcam:
                print("Failed to read frame from webcam.")
            else:
                print("Video ended.")
            break

        # 5. Run inference
        # Optimized for RTX 4050 (half=True)
        results = model.predict(
            source=frame, 
            device=device, 
            imgsz=args.imgsz, 
            half=(device == 'cuda'), 
            conf=0.25, 
            verbose=False
        )

        # 6. Visualize
        annotated_frame = results[0].plot()

        # Calculate actual processing time and FPS for display
        curr_time = time.time()
        duration = curr_time - loop_start
        actual_fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # small overlay
        cv2.putText(annotated_frame, f"FPS: {actual_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 7. Display
        cv2.imshow(window_name, annotated_frame)

        # 8. Sync Playback
        # If it's a video file, wait enough time to match original speed
        if not is_webcam:
            # Calculate remaining time to wait (ms)
            # wait_ms = (target - processing) * 1000
            wait_ms = int((target_frame_time - duration) * 1000)
            if wait_ms < 1:
                wait_ms = 1
        else:
            wait_ms = 1 # Run at max speed for live camera

        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detection finished.")

if __name__ == "__main__":
    main()
