from ultralytics import YOLO
import torch

# Load the YOLO26m model (Medium)
model = YOLO('yolo26m.pt')

# Export to ONNX with a more stable opset (17) for compatibility
# Opset 17 is generally well-supported by NPU toolchains
success = model.export(format='onnx', imgsz=640, simplify=True, opset=17)

print(f"Export successful: {success}")

# Check output shape
import onnx
model_onnx = onnx.load("yolo26m.onnx")
for input in model_onnx.graph.input:
    print(f"Input: {input.name}, Shape: {input.type.tensor_type.shape}")
for output in model_onnx.graph.output:
    print(f"Output: {output.name}, Shape: {output.type.tensor_type.shape}")
