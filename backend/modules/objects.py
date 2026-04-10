import os
from ultralytics import YOLO

class ObjectModule:
    """
    Handles object detection using Ultralytics YOLO (yolo26s.pt).
    Detects objects, assigns spatial position (left/center/right), and
    generates an LLM prompt for natural assistive narration.
    """
    def __init__(self, model_path_v26, model_path_v5_ignored=None):
        path = model_path_v26
        if not os.path.isabs(path):
            # Resolve relative to project root (3 levels up from this file)
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = os.path.join(base, path)

        print(f"[Objects] Loading YOLO v26 model: {path}")
        self.model = YOLO(path)
        self.conf_threshold = 0.50
        self.names = self.model.names
        print(f"[Objects] Model loaded. Classes: {len(self.names)}")

    def analyze_scene(self, frame):
        """
        Runs YOLO inference on the frame.
        Returns:
            objects_detected: list[str]  — e.g. ["person on your left", "chair in front"]
            raw_detections:   list[dict] — bbox, label, conf for visualization
        """
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)

        img_width = frame.shape[1]
        objects_detected = []
        raw_detections = []

        if results and len(results) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = self.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                center_x = (xyxy[0] + xyxy[2]) / 2
                if center_x < img_width / 3:
                    position = "on your left"
                elif center_x < (2 * img_width / 3):
                    position = "directly in front of you"
                else:
                    position = "on your right"

                print(f"[Objects] Detected: {label} ({conf:.2f}) {position}")
                objects_detected.append(f"{label} {position}")
                raw_detections.append({
                    "bbox": [int(x) for x in xyxy],
                    "label": label,
                    "conf": conf
                })

        return objects_detected, raw_detections

    def get_llm_prompt(self, objects_data):
        scene_info = ", ".join(objects_data) if objects_data else "no clear objects"
        return f"""
You are a helpful vision assistant for a visually impaired person.

Detected objects: {scene_info}

TASK:
Describe what is around the user in 1-2 natural sentences.

RULES:
- Be spatial: use "on your left", "in front of you", "on your right"
- Group duplicates (e.g. "two people in front of you")
- DO NOT mention AI, detection, or confidence scores
- Keep it short and clear

Response:"""
