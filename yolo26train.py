import roboflow
from ultralytics import YOLO
from roboflow import Roboflow
import yaml
import os
import glob

def main():
    print("--- Setting up YOLO26 Training ---")
    
    # rf = Roboflow(api_key="868TaqkwHhPtIstulvnK")
    # project = rf.workspace("anas-mohammed").project("detect-indian-currency-ldmie")
    # version = project.version(1)
    # dataset = version.download("yolo26")
                
    # print("\n--- Dataset Analysis ---")
    dataset_path = "Detect-Indian-Currency-2"
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        print(f"Num Classes: {data_config.get('nc', 'Unknown')}")
        print(f"Class Names: {data_config.get('names', 'Unknown')}")
        
        for split in ['train', 'valid', 'test']:
            img_dir = os.path.join(dataset_path, split, 'images')
            if os.path.exists(img_dir):
                count = len(glob.glob(os.path.join(img_dir, '*')))
                print(f"  - {split.ljust(6)} images: {count}")
            else:
                print(f"  - {split.ljust(6)} images: 0 (Directory not found)")
    else:
        print(f"WARNING: data.yaml not found at {yaml_path}")

    # 3. Initialize Model
    model = YOLO("yolo26m.pt")

    # 4. Hyperparameter Tuning (Find best settings)
    print("\n--- Starting Hyperparameter Tuning ---")
    best_hps = model.tune(
        data=yaml_path,
        epochs=30,        # Number of epochs per iteration
        iterations=30,    # Number of different hyperparameter combinations to try
        optimizer='AdamW',
        plots=True,       # Save tuning plots
        save=True,        # Save best hyperparameters
        val=True
    )
    print("Hyperparameter Tuning Complete. Applying best parameters to final training...")

    # 5. Train Model with Tuned Parameters
    print("\n--- Starting YOLO26 Training with Tuned Hyperparameters ---")
    
    # We pass the best hyperparameters discovered during tuning to the train call
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        device=0,      
        batch=16,      
        exist_ok=True, 
        project="yolo26_currency_run",
        **best_hps  # Apply the tuned hyperparameters here
    )
    print("Training Complete. Results saved to 'yolo26_currency_run'.")

if __name__ == '__main__':
    main()
