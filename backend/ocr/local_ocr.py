import ollama
import os
import sys

def run_local_ocr(image_path, model="deepseek-ocr:latest"):
    """
    Performs OCR on an image using a local Ollama model.
    """
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return

    print(f"--- Local OCR Inference ---")
    print(f"Model: {model}")
    
    try:
        # Request OCR from Ollama
        # Note: Using 'num_ctx' to ensure stable performance for vision models
        response = ollama.chat(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': 'Extract all text from this image exactly as it appears.',
                    'images': [image_path]
                }
            ],
            options={'num_ctx': 8192}
        )
        
        extracted_text = response['message']['content']
        print("\n--- Extracted Text ---")
        print(extracted_text)
        print("----------------------")
        return extracted_text

    except Exception as e:
        print(f"Error during inference: {e}")
        if "not found" in str(e).lower():
            print(f"Tip: Ensure the model '{model}' is pulled in Ollama (ollama pull {model})")
        return None

if __name__ == "__main__":
    # Default image path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_image = os.path.join(script_dir, "9ube35adqzf21.jpg")
    
    # Check for command line argument
    img_to_process = sys.argv[1] if len(sys.argv) > 1 else default_image
    
    # Check for model argument
    selected_model = sys.argv[2] if len(sys.argv) > 2 else "glm-ocr:latest"
    
    run_local_ocr(img_to_process, selected_model)
