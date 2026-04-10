import ollama
import os
import base64
import sys

# Add project root to path
BASE_DIR = r"c:\Users\anasm\OneDrive\Documents\Main Project\main-project-sw"
sys.path.append(BASE_DIR)

def test_ocr_pipeline(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"--- testing OCR with model: glm-ocr:latest ---")
    print(f"Image: {image_path}")

    # Read image
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # The user wants both identification and extraction in one natural response.
    # GLM-OCR is a VLM, so we can ask it directly.
    
    prompt = """
You are an OCR extraction system.

TASK:
1. Identify the object type (one word only): book, currency, label, sign, document, or unknown.
2. Extract ONLY visible text from the image.

STRICT RULES:
- DO NOT explain anything.
- DO NOT summarize.
- DO NOT add extra words.
- If no text is visible, return: NO_TEXT
- Keep output structured exactly like this:

OUTPUT FORMAT:
OBJECT: <object_type>
TEXT:
<raw extracted text>

"""

    try:
        response = ollama.chat(
            model='glm-ocr:latest',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_bytes]
            }],
        )
        
        print("\n--- OCR RESPONSE ---")
        print(response['message']['content'])
        print("\n")
        
    except Exception as e:
        print(f"Error calling Ollama: {e}")

if __name__ == "__main__":
    # Using one of the currency images as a placeholder for text testing
    test_img = os.path.join(BASE_DIR, "", "9ube35adqzf21.jpg")
    test_ocr_pipeline(test_img)
