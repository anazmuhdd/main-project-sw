import time
import os
import json
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

class CurrencyDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        # We handle model loading outside or inside, but YOLO handles path checks
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.names = self.model.names

    def process_frame(self, frame):
        """
        Runs inference and returns a summary of detections.
        """
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        
        counts = {}
        processed_results = []
        
        if len(results) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                name = self.names[cls_id]
                conf = float(box.conf[0])
                
                counts[name] = counts.get(name, 0) + 1
                processed_results.append({
                    "class": name,
                    "confidence": conf,
                    "value": CLASS_TO_VAL.get(name, 0)
                })
            
        return counts, processed_results

    def get_summary_text(self, counts):
        """
        Generates the raw data for the LLM.
        """
        if not counts:
            return "No currency detected."
            
        total_sum = sum(CLASS_TO_VAL[name] * count for name, count in counts.items())
        
        details = []
        # Sort by denomination for cleaner output
        sorted_names = sorted(counts.keys(), key=lambda x: CLASS_TO_VAL[x], reverse=True)
        for name in sorted_names:
            count = counts[name]
            val = CLASS_TO_VAL[name]
            label = f"{val} rupee note" if count == 1 else f"{val} rupee notes"
            details.append(f"{count} {label}")
            
        return {
            "total": total_sum,
            "counts": counts,
            "details_str": ", ".join(details)
        }

def generate_llm_prompt(summary_data):
    """
    Constructs the prompt for the LLM based on detection data.
    """
    if isinstance(summary_data, str):
        return f"User is holding the camera. Output which matches: {summary_data}"

    total = summary_data['total']
    details = summary_data['details_str']
    
    # SYSTEM PROMPT (Strict rules as requested)
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
Total: {total} rupees
Breakdown: {details}

OUTPUT:
"""

if __name__ == "__main__":
    import sys
    # Paths relative to script or absolute
    BASE_DIR = r"c:\Users\anasm\OneDrive\Documents\Main Project\main-project-sw"
    
    # Add project root to path for imports
    sys.path.append(BASE_DIR)
    from backend.modules.llm import LLMModule
    
    MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo26_seg.pt")
    TEST_IMAGE = os.path.join(BASE_DIR, "Testimages", "WhatsApp Image 2026-04-06 at 2.40.58 PM.jpeg")
    
    print(f"Loading model from: {MODEL_PATH}")
    detector = CurrencyDetector(MODEL_PATH)
    
    if os.path.exists(TEST_IMAGE):
        print(f"Testing with image: {TEST_IMAGE}")
        frame = cv2.imread(TEST_IMAGE)
        counts, results = detector.process_frame(frame)
        summary = detector.get_summary_text(counts)
        prompt = generate_llm_prompt(summary)
        
        print("\n--- DETECTION SUMMARY ---")
        print(json.dumps(summary, indent=2))
        print("\n--- FINAL PROMPT ---")
        print(prompt)
        
        print("\n--- LLM NARRATION ---")
        llm = LLMModule()
        print("Generating...", flush=True)
        for chunk in llm.generate_streaming_response(prompt):
            print(chunk, end="", flush=True)
        print("\n")
    else:
        print(f"Image not found at {TEST_IMAGE}.")
