import os
import cv2
import logging
from ultralytics import YOLO

# Denomination mapping
CLASS_TO_VAL = {
    'n10': 10,
    'n20': 20,
    'n50': 50,
    'n100': 100,
    'n200': 200,
    'n500': 500
}

class CurrencyModule:
    """
    Handles currency detection and summation using Ultralytics YOLO (v8/v26).
    """
    def __init__(self, model_path_v5_ignored=None, model_path_v26=None):
        # Prefer v26 as requested by the user
        path = model_path_v26 if model_path_v26 else model_path_v5_ignored
        if not path:
            raise ValueError("No model path provided for CurrencyModule")
            
        print(f"[Currency] Loading model: {path}")
        self.model = YOLO(path)
        self.conf_threshold = 0.5
        self.names = self.model.names

    def detect_and_sum(self, frame):
        """
        Runs inference and returns a summary for the server.
        Matches the interface expected by server.py: (description, total, raw_detections)
        """
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        
        counts = {}
        raw_detections = []
        total_sum = 0
        
        if len(results) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                name = self.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                
                val = CLASS_TO_VAL.get(name, 0)
                if val > 0:
                    counts[name] = counts.get(name, 0) + 1
                    total_sum += val
                    
                raw_detections.append({
                    "bbox": [int(x) for x in xyxy],
                    "label": name,
                    "conf": conf,
                    "value": val
                })
        
        if not counts:
            return "No currency detected.", 0, []

        details = []
        # Sort by denomination for cleaner output
        sorted_names = sorted(counts.keys(), key=lambda x: CLASS_TO_VAL[x], reverse=True)
        for name in sorted_names:
            count = counts[name]
            val = CLASS_TO_VAL[name]
            label = f"{val} rupee note" if count == 1 else f"{val} rupee notes"
            details.append(f"{count} {label}")
            
        details_str = ", ".join(details)
        # We return details_str as the 'description' for get_llm_prompt
        return details_str, total_sum, raw_detections

    def get_llm_prompt(self, details_str, total=None):
        """
        Constructs the high-quality prompt for the LLM.
        Note: server.py currently calls result_prompt = currency.get_llm_prompt(desc)
        where desc is the string returned by detect_and_sum.
        """
        # If total isn't passed, we try to extract it from context or just rely on details_str
        # However, for the new prompt, we want both. 
        # I'll update the server.py call as well to be more robust, 
        # or I can embed the total into the description if needed.
        
        # For now, let's assume details_str is the breakdown.
        # If total is None, we'll try to calculate it again or just omit from prompt (not ideal).
        
        # Actually, I will modify currency.py to be compatible with the current server.py 
        # but I'll also modify server.py to pass the total correctly.
        
        # If total is None, let's compute it from details_str for backward compatibility if possible
        # but better to just fix the call site.
        
        t_val = total if total is not None else "Unknown"

        return f"""
You are a real-time assistive AI helping a visually impaired person understand money they are holding.

GOAL:
Speak like a helpful human assistant describing what the user is holding.

IMPORTANT STYLE INSTRUCTIONS:
- Speak directly to the user.
- Use natural phrases like:
  "You are holding...", "You have...", "You are carrying..."
- Make it feel personal and immediate.
- Do NOT sound robotic or report-like.

STRICT RULES:
- Maximum 2 sentences.
- First sentence MUST include total amount.
- Second sentence MUST describe denominations.
- NEVER change the numbers.
- NEVER convert numbers into words (720 must stay 720).
- NEVER guess or add/remove notes.
- DO NOT mention detection, AI, or system details.
- Keep it simple, clear, and friendly.

FEW-SHOT EXAMPLES (FOR STYLE UNDERSTANDING ONLY — DO NOT COPY EXACTLY):

Example 1:
Input: Total 150 rupees, Breakdown: 1 100 rupee note, 1 50 rupee note
Output: You are holding 150 rupees, including one 100 note and one 50 note.

Example 2:
Input: Total 270 rupees, Breakdown: 1 200 rupee note, 1 50 rupee note, 1 20 rupee note
Output: You have 270 rupees with one 200 note, one 50 note, and one 20 note.

Example 3:
Input: Total 500 rupees, Breakdown: 1 500 rupee note
Output: You are carrying 500 rupees as a single 500 note.

NOTE:
- These are examples to understand tone and style.
- Do NOT copy sentences exactly.
- Always generate a fresh sentence based on input, not on the rough example.

INPUT:
Total: {t_val} rupees
Breakdown: {details_str}

OUTPUT:
"""
