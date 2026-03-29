from ultralytics import YOLO
import sys

def main():
    model_name = 'yolov8s.pt'
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    
    # Export to ONNX with standard raw outputs
    # imgsz=640, format='onnx', opset=12, simplify=True
    # We don't use end2end=True here as we want raw outputs
    print("Exporting to ONNX...")
    path = model.export(format='onnx', imgsz=640, opset=11, simplify=True)
    print(f"Exported to {path}")

if __name__ == "__main__":
    main()
