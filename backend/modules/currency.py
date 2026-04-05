import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path

# --- Dependency Setup ---
FILE = Path(__file__).resolve()
BACKEND_ROOT = FILE.parents[1] # c:/Users/anasm/.../backend
PROJECT_ROOT = BACKEND_ROOT.parents[0] # c:/Users/anasm/.../
YOLOV5_ROOT = PROJECT_ROOT / "yolov5"

# Add yolov5 to sys.path if not there
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))

# YOLOv5 Imports from local source
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.segment.general import process_mask
from utils.torch_utils import select_device

class CurrencyModule:
    """
    Handles high-quality currency segmentation and summation using YOLOv5 source code.
    """
    def __init__(self, model_path_v5, model_path_v26_ignored):
        # We'll use the user specified YOLOv5 currency model
        # Try GPU, fallback to CPU
        try:
            self.device = select_device("0")
        except Exception:
            self.device = select_device("cpu")
            
        self.half = self.device.type != "cpu"
        
        # Adjust weight path to be absolute or relative to project root
        if not os.path.isabs(model_path_v5):
            model_path_v5 = str((PROJECT_ROOT / model_path_v5).resolve())

        # print(f"[Currency] Loading model: {model_path_v5} on {self.device}...")
        self.model = DetectMultiBackend(model_path_v5, device=self.device, dnn=False, fp16=self.half)

        self.stride = int(self.model.stride)
        self.names = self.model.names                  # Output: {0:'n10', 1:'n100', ...}
        self.pt = self.model.pt
        self.imgsz = check_img_size(640, s=self.stride)
        
        # Warmup
        self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))
        # print(f"[Currency] Model loaded successfully. Classes: {list(self.names.values())}")
        
    def preprocess(self, frame):
        """Standard YOLOv5 letterbox preprocessing."""
        img = letterbox(frame, self.imgsz, stride=self.stride, auto=self.pt)[0]
        img = img.transpose((2, 0, 1))[::-1]        # HWC → CHW, BGR → RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).half() if self.half else torch.from_numpy(img).to(self.device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect_and_sum(self, frame):
        """
        Runs inference and sums note values using segmentation logic.
        """
        img = self.preprocess(frame)
        
        # 1. Inference
        out = self.model(img, augment=False, visualize=False)
        pred = out[0]        # (1, N, 5+nc+nm)
        proto = out[1]       # (1, 32, 160, 160)
        
        # 2. NMS (nm=32 for segmentation)
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False, max_det=1000, nm=32)
        
        detected_notes = []
        counts = {}
        total = 0
        
        # 3. Process detections
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes to original frame size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                
                for *xyxy, conf, cls in reversed(det[:, :6]):
                    raw_label = self.names[int(cls)]
                    # Handle 'n' prefix
                    label = raw_label[1:] if raw_label.startswith('n') else raw_label
                    print(f"[Currency] Detected: {raw_label} ({conf:.2f}) -> Path: {label}")
                    
                    try:
                        val = int(label)
                        total += val
                        detected_notes.append(val)
                        counts[val] = counts.get(val, 0) + 1
                    except ValueError:
                        # print(f"[Currency] Info: Skipped non-integer label '{label}'")
                        pass
                
        # 4. Result formatting
        if not detected_notes:
            print("[Currency] No currency found.")
            return "No currency detected.", 0
            
        summary_parts = []
        for val in sorted(counts.keys()):
            count = counts[val]
            summary_parts.append(f"{count} {val} rupee note{'s' if count > 1 else ''}")
            
        summary = ", ".join(summary_parts)
        print(f"[Currency] Sum: {total} | Breakdown: {summary}")
        return f"You are holding {summary}. The total amount is {total} rupees.", total

    def get_llm_prompt(self, description):
        """Constructs a prompt for a visually impaired user."""
        return f"""
        Role: You are a warm and helpful personal assistant for a visually impaired person.
        Context: The user is holding currency notes that have been detected by a camera.
        Input Data: {description}
        
        Task:
        1. Tell the user how much money they are holding in a clear and friendly manner.
        2. Specifically state the total amount first, then the breakdown of notes if relevant.
        3. Keep it conversational (e.g., "You have three fifty-rupee notes, totaling 150 rupees").

        IMPORTANT:
        - Output ONLY the natural speech for the user.
        - DO NOT output any internal reasoning, chain-of-thought, or <thought> tags.
        - Avoid technical phrases like "Detection Summary" or "Object detected".
        
        Response:"""

