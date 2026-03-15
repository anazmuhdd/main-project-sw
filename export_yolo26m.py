from ultralytics import YOLO
import torch

# Load the YOLO26m model (Medium)
model = YOLO('yolo26m.pt')

# Export to ONNX
# We fix simplify=True to make it more light on the NPU side
success = model.export(format='onnx', imgsz=640, simplify=True)

print(f"Export successful: {success}")

# Check output shape
import onnxruntime as ort
session = ort.InferenceSession("yolo26m.onnx")
inputs = session.get_inputs()
outputs = session.get_outputs()
print(f"Input: {inputs[0].name}, Shape: {inputs[0].shape}")
print(f"Output: {outputs[0].name}, Shape: {outputs[0].shape}")
