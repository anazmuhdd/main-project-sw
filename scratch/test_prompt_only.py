
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
You are a real-time assistive AI helping a visually impaired person understand money.

TASK:
Convert structured currency data into a short, clear spoken response.

STRICT RULES:
- Speak in a natural human tone.
- Maximum 2 sentences.
- First sentence MUST state the total amount clearly.
- Second sentence MUST describe the denominations.
- DO NOT mention detection, AI, model, confidence, or technical terms.
- DO NOT add extra explanations.
- DO NOT guess or hallucinate.
- Keep wording consistent and simple.
- Output must be ready for text-to-speech.

STYLE:
- Friendly, calm, and confident.
- Example: "You have 270 rupees, including one 200 note and one 50 note and one 20 note."

INPUT:
Total: {total} rupees
Breakdown: {details}

OUTPUT:
"""

if __name__ == "__main__":
    test_data = {
        "total": 350,
        "details_str": "one 200 rupee note, one 100 rupee note, one 50 rupee note"
    }
    print(generate_llm_prompt(test_data))
