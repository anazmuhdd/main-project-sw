import cv2
import torch
from ultralytics import YOLO
import time

def main():
    # --- CONFIGURATION ---
    MODEL_PATH = "yolo26s.pt"
    SOURCE = 0 # 0 for default webcam
    CONF_THRESHOLD = 0.5
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load Model
    print(f"Loading model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize Webcam
    print(f"Opening webcam source: {SOURCE}")
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Live Object Detection Started. Press 'q' to quit.")
    
    prev_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Run Inference
            # stream=True is more memory efficient for video
            results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
            
            # Draw results on the frame
            annotated_frame = results[0].plot()
            
            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow("YOLOv8 Live Object Detection", annotated_frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped. Resources released.")

if __name__ == "__main__":
    main()
