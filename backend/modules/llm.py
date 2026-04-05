import requests
import json
import ollama
import os

class LLMModule:
    """
    Handles LLM processing and response streaming for the visually impaired.
    """
    def __init__(self, use_nvidia=True, nvidia_key=None):
        self.use_nvidia = use_nvidia
        self.nvidia_key = nvidia_key
        self.local_model = "qwen3.5:2b"
        self.nvidia_url = "https://integrate.api.nvidia.com/v1/chat/completions"

    def generate_streaming_response(self, prompt):
        """
        Generates a streaming response for faster audio playback on the board.
        """
        if self.use_nvidia and self.nvidia_key:
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
                "stream": True # Stream the response as requested
            }
            
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
                                
        else:
            # Use local Ollama
            stream = ollama.chat(
                model=self.local_model,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
            )
            for chunk in stream:
                yield chunk['message']['content']
