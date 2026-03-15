import cv2
import time
import os

def test_pipeline(video_index):
    # This pipeline is specifically tuned for the A7Z ISP
    pipeline = (
        f"v4l2src device=/dev/video{video_index} ! "
        "video/x-raw,format=NV12,width=1280,height=720 ! "
        "videoconvert ! appsink"
    )
    
    print(f"\n--- Testing /dev/video{video_index} ---")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print(f"ERROR: Could not initialize GStreamer pipeline for video{video_index}")
        return False

    print("Pipeline started. Waiting 3 seconds for ISP to calibrate...")
    time.sleep(3)
    
    # Grab a few frames to clear the buffer
    for _ in range(5):
        cap.read()
        
    ret, frame = cap.read()
    if ret:
        filepath = f"captured_video{video_index}.jpg"
        cv2.imwrite(filepath, frame)
        print(f"SUCCESS! Frame captured and saved to: {os.path.abspath(filepath)}")
    else:
        print(f"FAILED: Could not read frame from video{video_index}")
        
    cap.release()
    return ret

# Try the most likely nodes
test_pipeline(0)
test_pipeline(1)
