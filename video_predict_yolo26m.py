import cv2
from ultralytics import YOLO
import time
import os

# Load the YOLO26m model
model = YOLO('yolo26m.pt')

# Input video
video_file = '/home/radxa/test/test_video.mp4'

if not os.path.exists(video_file):
    print(f"Error: {video_file} not found.")
    exit(1)

cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print(f"Error: Could not open video {video_file}")
    exit(1)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video
out = cv2.VideoWriter('output_inference.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"Processing video: {video_file} ({width}x{height} at {fps} fps)")
print("Running on CPU - This will be slow (approx 1-3 seconds per frame)")

frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO26 inference
    results = model(frame, verbose=False)
    
    # Draw results on frame
    annotated_frame = results[0].plot()
    
    # Save frame
    out.write(annotated_frame)
    
    frame_count += 1
    if frame_count % 10 == 0:
        elapsed = time.time() - start_time
        print(f"Processed {frame_count} frames. Average speed: {elapsed/frame_count:.2f} s/frame")
    
    # Limit for testing - avoid huge processing time if video is long
    if frame_count > 50:
        print("Stopping after 50 frames for testing...")
        break

cap.release()
out.release()
print(f"Finished processing. Output saved as output_inference.mp4")
