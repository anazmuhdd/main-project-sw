import requests
import json
import ollama
import os

class LLMModule:
    """
    Handles LLM processing and response streaming for the visually impaired.
    """
    def __init__(self, use_nvidia=False, nvidia_key=None):
        self.use_nvidia = use_nvidia
        self.nvidia_key = nvidia_key
        self.local_model = "qwen3.5:2b"  # User has this model locally
        self.nvidia_url = "https://integrate.api.nvidia.com/v1/chat/completions"

    def generate_streaming_response(self, prompt):
        """
        Generates a streaming response for faster audio playback on the board.
        Supports local Ollama and NVIDIA Cloud.
        """
        if self.use_nvidia and self.nvidia_key:
            # NVIDIA Implementation... (omitted for brevity in this view, keeping it unchanged)
            headers = {
                "Authorization": f"Bearer {self.nvidia_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "nvidia/qwen-2.5-7b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
                "stream": True
            }
            
            try:
                response = requests.post(self.nvidia_url, headers=headers, json=payload, stream=True)
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            data = json.loads(decoded_line[6:])
                            if 'choices' in data:
                                chunk = data['choices'][0].get('delta', {}).get('content', '')
                                if chunk:
                                    yield chunk
            except Exception as e:
                yield f"[NVIDIA Error]: {e}"
                                
        else:
            # Use local Ollama
            try:
                stream = ollama.chat(
                    model=self.local_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=True,
                    think=False
                )
                for chunk in stream:
                    content = chunk['message']['content']
                    # Simple filter to skip common thinking tags if they appear despite prompt instructions
                    if not any(tag in content.lower() for tag in ["<thought>", "</thought>", "thinking..."]):
                        yield content
            except Exception as e:
                yield f"[Local LLM Error]: {e}. Please ensure Ollama is running (`ollama serve`)."
