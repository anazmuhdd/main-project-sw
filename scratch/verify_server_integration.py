import sys
import os
import cv2
import json

# Add project root to path
BASE_DIR = r"c:\Users\anasm\OneDrive\Documents\Main Project\main-project-sw"
sys.path.append(BASE_DIR)

from backend.modules.currency import CurrencyModule
from backend.modules.llm import LLMModule

def test_currency_integration():
    MODEL_PATH_V26 = os.path.join(BASE_DIR, "models", "yolo26_seg.pt")
    TEST_IMAGE = os.path.join(BASE_DIR, "Testimages", "WhatsApp Image 2026-04-06 at 2.40.58 PM.jpeg")
    
    print(f"--- TESTING CURRENCY INTEGRATION ---")
    print(f"Model: {MODEL_PATH_V26}")
    print(f"Image: {TEST_IMAGE}")
    
    if not os.path.exists(TEST_IMAGE):
        print(f"ERROR: Image not found at {TEST_IMAGE}")
        return

    # Initialize Module (matching server.py style)
    currency = CurrencyModule(None, MODEL_PATH_V26)
    
    # Process Image (matching server.py vision_stream loop)
    frame = cv2.imread(TEST_IMAGE)
    desc, total, raw_detections = currency.detect_and_sum(frame)
    
    print("\n--- DETECTION RESULTS ---")
    print(f"Total: {total}")
    print(f"Description: {desc}")
    print(f"Raw Detections Count: {len(raw_detections)}")
    
    # Generate Prompt (matching server.py vision_stream loop)
    if total > 0:
        prompt = currency.get_llm_prompt(desc, total)
        print("\n--- GENERATED PROMPT ---")
        print(prompt)
        
        # Test LLM (optional but good for verification)
        print("\n--- LLM RESPONSE ---")
        llm = LLMModule()
        for chunk in llm.generate_streaming_response(prompt):
            print(chunk, end="", flush=True)
        print("\n")
    else:
        print("No currency detected to process.")

if __name__ == "__main__":
    test_currency_integration()
