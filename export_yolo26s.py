from ultralytics import YOLO
import torch

# Load the YOLO26s model (Small)
# This is usually the best balance for real-time NPU performance (20-30 FPS)
model = YOLO('yolo26s.pt')

# Export to ONNX with Opset 17 for NPU compatibility
success = model.export(format='onnx', imgsz=640, simplify=True, opset=17)

print(f"Export successful: {success}")
