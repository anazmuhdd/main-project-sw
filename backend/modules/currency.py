import torch
from ultralytics import YOLO
import numpy as np

class CurrencyModule:
    """
    Handles currency detection, summation, and formatting for the visually impaired.
    """
    def __init__(self, model_path_v5, model_path_v26):
        # We'll use the better-performing model for summation
        # Using YOLOv5 by default as specified (or YOLOv26)
        self.model = YOLO(model_path_v5)
        
    def detect_and_sum(self, frame):
        """
        Runs inference and sums note values.
        """
        results = self.model(frame)[0]
        detected_notes = []
        counts = {}
        total = 0
        
        for box in results.boxes:
            raw_label = results.names[int(box.cls)]
            # Fix: Handle 'n10', 'n50' prefix
            label = raw_label[1:] if raw_label.startswith('n') else raw_label
            print(f"[Currency] Detected: {raw_label} -> Parsed as: {label}")
            
            try:
                val = int(label)
                total += val
                detected_notes.append(val)
                counts[val] = counts.get(val, 0) + 1
            except ValueError:
                print(f"[Currency] Warning: Could not parse label '{label}' as integer.")
                pass
                
        # Create a detailed description
        if not detected_notes:
            print("[Currency] No currency found in frame.")
            return "No currency detected.", 0
            
        summary = ", ".join([f"{count} {val} rupee note{'s' if count > 1 else ''}" for val, count in counts.items()])
        print(f"[Currency] Final Sum: {total} | Breakdown: {summary}")
        return f"You are holding {summary}. The total amount is {total} rupees.", total


    def get_llm_prompt(self, description):
        """
        Constructs a high-quality prompt for a visually impaired user.
        """
        return f"""
        Role: Helpful assistant for a visually impaired person.
        Task: Describe the currency they are holding based on the detection result.
        Detection Result: {description}
        Guidelines:
        1. Be concise but warm and helpful.
        2. Specifically mention the total amount clearly.
        3. If there are multiple notes, mention each set.
        Example Output: "Hello, you have two 50 rupee notes and one 100 rupee note. The total amount is 200 rupees."
        Your Response:"""
