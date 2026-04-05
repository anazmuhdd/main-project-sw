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
        Prompt for the LLM to summarize the OCR text for a visually impaired user.
        """
        if not ocr_text or len(ocr_text.strip()) == 0:
            return "I am sorry, but I cannot find any readable text in front of you."
            
        return f"""
        Role: You are an intelligent and clear-voiced reader for a visually impaired person.
        Context: The user is pointing a camera at a document, sign, or object with text.
        Extracted Text: {ocr_text}
        
        Task:
        1. Summarize the text detected. Identify what type of object it is (e.g., menu, medicine bottle, street sign).
        2. Keep the information well-structured. Read only the most critical parts first (like the product name, price, or warning).
        3. Be conversational and helpful.

        IMPORTANT:
        - Output ONLY the natural speech for the user.
        - DO NOT output any internal reasoning, chain-of-thought, or <thought> tags.
        - Avoid technical terms like "OCR Text Detected" or "Action".
        
        Response:"""
