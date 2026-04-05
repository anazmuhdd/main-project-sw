import torch
from ultralytics import YOLO
import numpy as np

class ObjectModule:
    """
    Handles object detection, mapping, and spatial orientation for the visually impaired.
    """
    def __init__(self, model_path_v26, model_path_v5):
        try:
            self.model = YOLO(model_path_v26)
        except Exception:
            # Fallback for YOLOv5
            self.model = YOLO(model_path_v5)
        
    def analyze_scene(self, frame):
        """
        Detects objects and calculates spatial orientation.
        """
        results = self.model(frame)[0]
        img_width = frame.shape[1]
        img_height = frame.shape[0]
        
        objects_detected = []
        for box in results.boxes:
            label = results.names[int(box.cls)]
            conf = float(box.conf)
            if conf < 0.3: continue # Filter low confidence
            
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            center_x = (x_min + x_max) / 2
            
            # Divide frame into three horizontal zones for spatial orientation
            if center_x < img_width / 3:
                position = "on your left"
            elif center_x < (2 * img_width / 3):
                position = "directly in front of you"
            else:
                position = "on your right"
                
            objects_detected.append(f"{label} {position}")
            
        return objects_detected

    def get_llm_prompt(self, objects_data):
        """
        Generate a scene explanation prompt.
        """
        scene_info = ", ".join(objects_data) if objects_data else "no clear objects"
        return f"""
        Role: Intelligent scene narrator for a visually impaired user.
        Information Detected: {scene_info}
        Objective:
        1. Summarize the immediate surroundings clearly.
        2. Specifically use spatial words like "left", "right", and "directly in front".
        3. If no objects are found, tell the user the area is clear.
        4. Organize the information from closest to furthest or left to right.
        Example Output: "In your view, there is a door on your left and a chair directly in front of you. Be careful as you walk."
        Your Response:"""
