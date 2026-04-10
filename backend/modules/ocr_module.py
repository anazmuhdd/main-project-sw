import ollama
import base64

class OCRModule:
    """
    Handles OCR extraction (Ollama/GLM-OCR) and structured parsing.
    Follows strict internal instructions for decision logic and text cleaning.
    """
    def __init__(self, model="glm-ocr:latest"):
        self.model = model

    def perform_ocr(self, frame_bytes):
        """
        STEP 1 — OCR PROMPT (STRICT EXTRACTION)
        """
        prompt_ocr = """
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

OBJECT: <object_type>
TEXT:
<raw extracted text>
"""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt_ocr.strip(),
                    'images': [frame_bytes]
                }],
                options={'num_ctx': 4096}
            )
            return response['message']['content']
        except Exception as e:
            print(f"[OCR Module Error]: {e}")
            return "OBJECT: unknown\nTEXT: NO_TEXT"

    def parse_ocr_output(self, ocr_text):
        """
        STEP 2 — PARSER LOGIC (MANDATORY)
        Extracts OBJECT and TEXT. Robust fallback if VLM skips headers.
        """
        object_type = "unknown"
        text_content = ""
        lines = ocr_text.splitlines()
        
        # Check if the output follows the structured format
        has_headers = "OBJECT:" in ocr_text and "TEXT:" in ocr_text
        
        if not has_headers:
            # Fallback: Treat the whole thing as text if it doesn't look like a structured output
            # but check for object keywords just in case they appeared elsewhere
            text_content = ocr_text
            for line in lines:
                if "OBJECT:" in line.upper():
                    object_type = line.upper().replace("OBJECT:", "").strip().lower()
                    # Remove the OBJECT line from text content if it was found
                    text_content = text_content.replace(line, "").strip()
        else:
            capture_text = False
            for line in lines:
                line = line.strip()
                if line.startswith("OBJECT:"):
                    object_type = line.replace("OBJECT:", "").strip().lower()
                elif line.startswith("TEXT:"):
                    capture_text = True
                    continue
                elif capture_text:
                    text_content += line + " "

        # One final check: if text_content is just "NO_TEXT", clear it
        if text_content.strip().upper() == "NO_TEXT":
            text_content = ""

        return object_type, text_content.strip()

    def clean_text(self, text):
        """
        STEP 4 — CLEAN TEXT
        Removes duplicates and noise before passing to LLM.
        """
        if not text or text.upper() == "NO_TEXT":
            return ""
        
        # Deduplication of lines/phrases
        phrases = text.split("  ")  # Sometimes VLM separates with double spaces
        if len(phrases) <= 1:
            phrases = text.split(". ")
            
        unique_phrases = list(dict.fromkeys(p.strip() for p in phrases if p.strip()))
        return " ".join(unique_phrases)

    def get_llm_prompt(self, object_type, text):
        """
        STEP 5 — LLM PROMPT (FINAL NARRATION)
        Generates the natural spoken response via NVIDIA NIM.
        """
        return f"""
You are an assistive AI helping a visually impaired person.

INPUT:
Object: {object_type}
Text: {text}

TASK:
Generate a natural spoken response.

RULES:
- Start with: "You are holding..." or "In front of you is..."
- First identify the object clearly
- Then speak the important content
- Maximum 2 sentences
- Keep it simple and natural
- DO NOT mention OCR or extraction
- DO NOT repeat unnecessary text
- If text is long → summarize in one sentence
- If text is short → read it clearly

EXAMPLES (style only):

Input:
Object: currency
Text: RESERVE BANK OF INDIA 500
Output:
You are holding currency notes. I can see a 500 rupee note.

Input:
Object: book
Text: This chapter explains thermodynamics...
Output:
You are holding a book page. It explains the basics of thermodynamics.

FINAL OUTPUT:
"""
