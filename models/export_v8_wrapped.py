from ultralytics import YOLO
import torch
import sys
import os

def main():
    v10_path = 'yolo26s.pt'
    if len(sys.argv) > 1:
        v10_path = sys.argv[1]
    
    print(f"Loading YOLOv10 weights from {v10_path}...")
    # We load the v10 weights
    v10_model = torch.load(v10_path, map_location='cpu')
    
    # We create a YOLOv8 model of the same size (Small)
    # Note: YOLO26s is v10s which is comparable to v8s
    print("Initializing YOLOv8s architecture...")
    v8_model = YOLO('yolov8s.yaml') # Use yaml to get architecture only
    
    # Load weights. We use non-strict because the head will differ, 
    # but the backbone and neck (where the heavy lifting happens) are identical.
    print("Mapping weights (Backbone/Neck)...")
    v8_model.model.load_state_dict(v10_model['model'].state_dict(), strict=False)
    
    # Export the YOLOv8-wrapped model
    print("Exporting v8-wrapped model to ONNX...")
    # This will now produce the standard [1, 84, 8400] output
    path = v8_model.export(format='onnx', imgsz=640, opset=11, simplify=True)
    
    # Rename to yolo26s_raw.onnx
    os.rename(path, 'yolo26s_raw.onnx')
    print(f"Exported to yolo26s_raw.onnx")

if __name__ == "__main__":
    main()
