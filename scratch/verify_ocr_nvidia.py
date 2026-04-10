import sys
import os
import json
import io

# Fix encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
BASE_DIR = r"c:\Users\anasm\OneDrive\Documents\Main Project\main-project-sw"
sys.path.append(BASE_DIR)

from backend.modules.ocr_module import OCRModule
from backend.modules.llm import LLMModule

# User's provided API Key
NVIDIA_API_KEY = "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"

def test_ocr_nvidia_pipeline():
    # 1. Initialize Modules
    ocr_module = OCRModule()
    llm = LLMModule(use_nvidia=True, nvidia_key=NVIDIA_API_KEY)
    
    # 2. Pick a test image
    test_img_path = os.path.join(BASE_DIR, "9ube35adqzf21.jpg")
    
    if not os.path.exists(test_img_path):
        print(f"Error: Test image not found at {test_img_path}")
        return

    print(f"--- TESTING OCR + NVIDIA NIM PIPELINE ---")
    print(f"Image: {test_img_path}")

    # 3. Perform OCR (Extraction)
    with open(test_img_path, "rb") as f:
        img_bytes = f.read()
    
    print("\n[Step 1] Performing Structured OCR...")
    ocr_result = ocr_module.perform_ocr(img_bytes)
    
    print("--- RAW OCR RESULT ---")
    print(ocr_result)
    print("----------------------")

    # Step 2: Parser Logic
    obj, text = ocr_module.parse_ocr_output(ocr_result)
    # Step 4: Clean Text
    cleaned_text = ocr_module.clean_text(text)
    
    print(f"Object Identified: {obj}")
    print(f"Cleaned Text: {cleaned_text[:100]}...")

    # Step 3: Decision Logic (Critical)
    if not cleaned_text or cleaned_text.upper() == "NO_TEXT":
        print("\nNarration Result: I cannot see any readable text.")
        return

    # 4. Generate Narration (LLM)
    print("\n[Step 2] Generating Narration via NVIDIA...")
    prompt = ocr_module.get_llm_prompt(obj, cleaned_text)
    
    print("Narration Result: ", end="", flush=True)
    for chunk in llm.generate_streaming_response(prompt):
        print(chunk, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    test_ocr_nvidia_pipeline()
