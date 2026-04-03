import os
import yaml
from ultralytics import YOLO

def main():
    # 1. Setup paths
    base_dir = os.getcwd()
    models_dir = os.path.join(base_dir, 'models')
    calib_images_dir = os.path.join(base_dir, 'ai-sdk', 'models', 'yolo26', 'images')
    
    # 2. Create a temporary calibration YAML
    # ExecuTorch INT8 export needs a dataset to calibrate quantization
    calib_yaml_path = os.path.join(base_dir, 'tools', 'calib.yaml')
    calib_data = {
        'path': base_dir,
        'train': 'ai-sdk/models/yolo26/images', # relative to path
        'val': 'ai-sdk/models/yolo26/images',
        'names': {0: 'item'} # Generic class name
    }
    
    with open(calib_yaml_path, 'w') as f:
        yaml.dump(calib_data, f)
    
    print(f"Created calibration YAML at {calib_yaml_path}")

    # 3. Export Detection Model
    det_model_path = os.path.join(models_dir, 'yolo26s.pt')
    if os.path.exists(det_model_path):
        print(f"--- Exporting Detection Model: {det_model_path} ---")
        model = YOLO(det_model_path)
        # Note: imgsz=640 is standard, ExecuTorch uses XNNPACK for CPU optimization
        model.export(format='executorch', simplify=True)
    else:
        print(f"Error: {det_model_path} not found.")

    # 4. Export Segmentation Model
    seg_model_path = os.path.join(models_dir, 'yolo26_seg.pt')
    if os.path.exists(seg_model_path):
        print(f"--- Exporting Segmentation Model: {seg_model_path} ---")
        model = YOLO(seg_model_path)
        model.export(format='executorch', simplify=True)
    else:
        print(f"Error: {seg_model_path} not found.")

    # 5. Cleanup
    if os.path.exists(calib_yaml_path):
        os.remove(calib_yaml_path)
        print("Cleaned up temporary calibration YAML.")

if __name__ == "__main__":
    main()
