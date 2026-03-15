from ultralytics import YOLO
import torch

# Load a YOLO26 model
model = YOLO('yolo26n.pt')

# Export the model to ONNX format
# We specify imgsz=640 to match the usual input size
success = model.export(format='onnx', imgsz=640)
print(f"Export successful: {success}")
