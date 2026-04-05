import ollama
import os
import io
import base64
from PIL import Image

class OCRModule:
    """
    Handles OCR inference and text summarization for the visually impaired.
    """
    def __init__(self, model="glm-ocr:latest"):
        self.model = model
        
    def perform_ocr(self, frame_bytes):
        """
        Runs OCR on a given frame using Ollama.
        """
        try:
            # Convert frame to base64 if it's already bytes, otherwise encode
            # assuming it's passed as encoded JPEG bytes
            img_b64 = base64.b64encode(frame_bytes).decode()
            
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': 'Extract all text from this image exactly as it appears.',
                    'images': [frame_bytes]
                }],
                options={'num_ctx': 8192}
            )
            return response['message']['content']
        except Exception as e:
            print(f"OCR Error: {e}")
            return None

    def get_llm_prompt(self, ocr_text):
        """
        Prompt for the LLM to summarize the OCR text.
        """
        if not ocr_text or len(ocr_text.strip()) == 0:
            return "No readable text was found in the image."
            
        return f"""
        Role: Intelligent reader for a visually impaired person.
        OCR Text Detected: {ocr_text}
        Action:
        1. Summarize the text found based on the object (e.g., if it's a menu, a sign, or a label).
        2. Specifically tell the user what they are looking at and read the most important parts.
        3. Keep the most helpful information (like expiry dates or warnings) prominent.
        Example Output: "This is a food label. It says it's a pack of bread, and the best before date is October 10th."
        Your Response:"""
