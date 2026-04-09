import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path

# --- Dependency Setup ---
FILE = Path(__file__).resolve()
BACKEND_ROOT = FILE.parents[1]
PROJECT_ROOT = BACKEND_ROOT.parents[0]
YOLOV5_ROOT = PROJECT_ROOT / "yolov5"

if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

class ObjectModule:
    """
    Handles high-quality object detection and spatial awareness using YOLOv5.
    """
    def __init__(self, model_path_v26_ignored, model_path_v5):
        # Try GPU, fallback to CPU
        try:
            self.device = select_device("0")
        except Exception:
            self.device = select_device("cpu")
            
        self.half = self.device.type != "cpu"

        
        if not os.path.isabs(model_path_v5):
            # Check if it's in the root or in a specific folder
            root_path = str((PROJECT_ROOT / model_path_v5).resolve())
            if os.path.exists(root_path):
                model_path_v5 = root_path
            else:
                # Try finding it in the root directly
                model_path_v5 = str((PROJECT_ROOT / "yolov5s.pt").resolve())

        # print(f"[Objects] Loading model: {model_path_v5} on {self.device}...")
        self.model = DetectMultiBackend(model_path_v5, device=self.device, dnn=False, fp16=self.half)
        self.stride = int(self.model.stride)
        self.names = self.model.names
        self.pt = self.model.pt
        self.imgsz = check_img_size(640, s=self.stride)
        
        self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))
        # print(f"[Objects] Model loaded. Classes: {len(self.names)}")
        
    def preprocess(self, frame):
        img = letterbox(frame, self.imgsz, stride=self.stride, auto=self.pt)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).half() if self.half else torch.from_numpy(img).to(self.device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def analyze_scene(self, frame):
        """
        Detects objects and calculates spatial orientation.
        Returns: (objects_detected, raw_detections)
        """
        img = self.preprocess(frame)
        pred = self.model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.50, 0.45, classes=None, agnostic=False, max_det=1000)
        
        img_width = frame.shape[1]
        objects_detected = []
        raw_detections = []
        
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    label = self.names[int(cls)]
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    
                    if center_x < img_width / 3:
                        position = "on your left"
                    elif center_x < (2 * img_width / 3):
                        position = "directly in front of you"
                    else:
                        position = "on your right"
                        
                    print(f"[Objects] Detected: {label} ({conf:.2f}) {position}")
                    objects_detected.append(f"{label} {position}")
                    
                    # Store raw detection for server-side visualization
                    raw_detections.append({
                        "bbox": [int(x) for x in xyxy],
                        "label": label,
                        "conf": float(conf)
                    })
            
        return objects_detected, raw_detections

    def get_llm_prompt(self, objects_data):
        scene_info = ", ".join(objects_data) if objects_data else "no clear objects"
        return f"""
        Role: You are a helpful, professional AI vision assistant for a blind or visually impaired person.
        Context: The user is holding a camera that detects objects and their relative positions.
        Input Data: {scene_info}
        
        Task:
        1. Describe the surroundings in a natural, conversational, and spatial way (e.g., "There is a person directly ahead of you and a chair on your left").
        2. Group multiple identical objects together (e.g., "I see three people standing in front of you"), if multiple items are presnet in the input data.
        3. Be concise but descriptive. Avoid listing technical details like confidence scores.

        IMPORTANT:
        - Output ONLY the natural speech for the user.
        - DO NOT output any internal reasoning, chain-of-thought, or <thought> tags.
        - Do not say things like "I am an AI" or "Based on the data". Just narrate.
        - Dont tell anything additional, be consice and accurate, only give the sentence according to the input data.
        
        Response:"""

