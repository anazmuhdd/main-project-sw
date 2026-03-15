import cv2
from ultralytics import YOLO
import os

# Load the YOLO26m model (COCO pre-trained)
model = YOLO('yolo26m.pt')

video_path = '/home/radxa/test/test_video.mp4'
if not os.path.exists(video_path):
    print(f"Error: {video_path} not found.")
    exit(1)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if ret:
    results = model(frame)
    results[0].save(filename='yolo26m_video_test_frame.jpg')
    print("Inference on the first frame of the video successful! Result saved.")
else:
    print("Could not read frame from video.")

cap.release()
