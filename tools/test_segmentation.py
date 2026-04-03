import cv2
import os
import time
import torch
import numpy as np
import sys

# Add the cloned YOLOv5 repository to the path so its internal modules (models, utils) are discoverable
REPO_PATH = os.path.abspath('yolov5')
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

from ultralytics import YOLO
from ultralytics.nn.tasks import BaseModel

# Patch: Fix internal Ultralytics compatibility bug where fuse() gets unexpected 'verbose' arg
_original_fuse = BaseModel.fuse
def patched_fuse(self, *args, **kwargs):
    # Remove 'verbose' if current version of fuse doesn't support it or if it causes issues
    kwargs.pop('verbose', None)
    return _original_fuse(self, *args, **kwargs)
BaseModel.fuse = patched_fuse

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = r'models/yolov5_currency.pt'
VIDEO_PATH = r'currency_video.mp4'
OUTPUT_PATH = r'currency_segmentation_test.mp4'
CONF_THRESHOLD = 0.3
IMG_SIZE = 640
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # 1. Environment Check
    print("=" * 50)
    print("🚀 YOLOv5 Currency Segmentation Test Script")
    print("=" * 50)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video not found at {VIDEO_PATH}")
        return

    # 2. Loading Model
    # We use Ultralytics as it's the modern standard for YOLO models.
    # If this is an older YOLOv5 repo-style model, it might require the yolov5 codebase to be on PYTHONPATH.
    print(f"📦 Loading Model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Model loaded successfully via Ultralytics.")
    except Exception as e:
        print(f"⚠️ Warning: Directly loading with Ultralytics failed: {str(e)[:100]}...")
        print("💡 This appears to be a legacy YOLOv5 repository model.")
        print("🔄 Attempting to load via torch.hub...")
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=False)
            model.to(DEVICE) # Move to GPU
            print(f"✅ Model loaded successfully via Torch Hub on {DEVICE}.")
            is_legacy = True
        except Exception as hub_e:
            print(f"❌ Error: Failed to load model. Please ensure you have the YOLOv5 repository if this is a legacy model.")
            print(f"Detailed Error: {hub_e}")
            return
    else:
        # For Ultralytics, we don't need to manually call .to(DEVICE)
        # as it's passed during predict()
        print(f"✅ Model loaded successfully via Ultralytics.")
        is_legacy = False

    # 3. Setup Video Processing
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {VIDEO_PATH}")
        return

    # Get video properties for Output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0: orig_fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Universal codec for .mp4
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, orig_fps, (width, height))

    print(f"🎞️ Processing Video: {VIDEO_PATH} ({width}x{height} @ {orig_fps}fps)")
    print(f"💾 Saving Output to: {OUTPUT_PATH}")
    print(f"⚡ GPU Acceleration: {'Enabled' if DEVICE == 'cuda' else 'Disabled'}")
    print("⏳ Please wait, processing frames...")

    frame_count = 0
    start_time = time.time()
    
    try:
        # 4. Processing Loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            loop_start = time.time()
            
            # Performance Inference
            if not is_legacy:
                # Ultralytics API
                # Removed 'verbose' to avoid compatibility issues with certain versions
                results = model.predict(source=frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, device=DEVICE)
                annotated_frame = results[0].plot()
            else:
                # Legacy YOLOv5 API
                results = model(frame, size=IMG_SIZE)
                annotated_frame = results.render()[0]

            # Add custom premium overlay
            process_time = time.time() - loop_start
            fps = 1.0 / process_time if process_time > 0 else 0
            
            # Status Bar at top
            cv2.rectangle(annotated_frame, (0, 0), (width, 40), (45, 45, 45), -1)
            cv2.putText(annotated_frame, f"Model: {os.path.basename(MODEL_PATH)} | Device: {DEVICE.upper()} | FPS: {fps:.1f}", 
                        (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write output
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"🛠️ Processed {frame_count} frames...")
    finally:
        # 5. Cleanup - GUARANTEED closure to prevent video corruption
        cap.release()
        out.release()
        print("🔒 Video streams released and file finalized.")
    
    total_time = time.time() - start_time
    print("=" * 50)
    print("🏁 INFERENCE COMPLETE")
    print("-" * 30)
    print(f"⏱️ Total Time: {total_time:.2f}s")
    print(f"📈 Avg FPS: {frame_count/total_time:.2f}")
    print(f"📂 Result Saved: {os.path.abspath(OUTPUT_PATH)}")
    print("=" * 50)

if __name__ == "__main__":
    main()
