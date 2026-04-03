"""
🚀 ExecuTorch Inference Test Script for Radxa Cubie A7Z
This script is designed to run on the Radxa board.
Requirements: 
1. pip install executorch opencv-python
2. Copy the .pte file and metadata.yaml to the board.
"""

import time
import cv2
import torch
import numpy as np
import yaml
from pathlib import Path
from executorch.runtime import Runtime

def preprocess(image_path, imgsz):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz))
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0), img

def main(model_path, image_path):
    # 1. Load Metadata
    metadata_path = Path(model_path).parent / 'metadata.yaml'
    with open(metadata_path, 'r') as f:
        meta = yaml.safe_load(f)
    
    imgsz = meta['imgsz'][0]
    names = meta['names']
    print(f"Loaded model metadata: {meta['task']} task, {imgsz}x{imgsz} input.")

    # 2. Load Model into ExecuTorch Runtime
    print(f"Loading model {model_path}...")
    runtime = Runtime(model_path)

    # 3. Preprocess Input
    input_tensor, _ = preprocess(image_path, imgsz)

    # 4. Inference
    print("Running inference...")
    start_time = time.time()
    outputs = runtime.forward([input_tensor])
    end_time = time.time()

    print(f"Inference took: {(end_time - start_time)*1000:.2f} ms")
    
    # 5. Simple Post-processing (Top-1)
    # Note: For YOLO, you'd usually have a more complex NMS here, 
    # but the exported .pte may already have NMS embedded if 'nms=True' was used.
    # Check outputs
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")

if __name__ == "__main__":
    # Example usage:
    # python inference_test_a7z.py yolo26_seg.pte test_image.jpg
    import sys
    if len(sys.argv) < 3:
        print("Usage: python inference_test_a7z.py <model.pte> <image.jpg>")
    else:
        main(sys.argv[1], sys.argv[2])
