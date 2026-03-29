from ultralytics import YOLO
import ultralytics.nn.modules.head as head
import torch
import sys

# Patch the v10Detect forward to return raw features instead of the slow NMS-free head
def patched_v10_forward(self, x):
    # v10 has two heads: one2many (v8 style) and one2one (v10 style)
    # When training is false, it usually decodes one2one. 
    # We return the raw convolutions from the one2one head.
    return [self.one2one_cv[i](x[i]) for i in range(self.nl)]

# Apply the patch
head.v10Detect.forward = patched_v10_forward

def main():
    model_path = 'yolo26s.pt'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    print(f"Loading model from {model_path} with PATCHED head...")
    model = YOLO(model_path)
    
    # Export the model to ONNX format
    print("Exporting PATCHED model to ONNX...")
    # This should now produce 3 raw output tensors (one for each scale)
    path = model.export(format='onnx', imgsz=640, opset=11, simplify=True)
    print(f"Exported to {path}")

if __name__ == "__main__":
    main()
