from ultralytics import YOLO
import sys

def main():
    model_path = 'yolo26s.pt'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Export the model to ONNX format
    # We use imgsz=640 (standard for YOLO) and format='onnx'
    # We ensure it's a static graph by not using dynamic shapes
    print("Exporting to ONNX...")
    path = model.export(format='onnx', imgsz=640, opset=12, simplify=True)
    print(f"Exported to {path}")

if __name__ == "__main__":
    main()
