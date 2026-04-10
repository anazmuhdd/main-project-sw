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
                "model": "qwen/qwen3.5-122b-a10b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 1024,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False}
            }
            
            try:
                response = requests.post(self.nvidia_url, headers=headers, json=payload, stream=True)
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            content_str = decoded_line[6:].strip()
                            if content_str == "[DONE]" or not content_str:
                                continue
                            try:
                                data = json.loads(content_str)
                                if 'choices' in data:
                                    chunk = data['choices'][0].get('delta', {}).get('content', '')
                                    if chunk:
                                        yield chunk
                            except json.JSONDecodeError:
                                continue
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
                in_think_block = False
                buffer = ""
                for chunk in stream:
                    content = chunk['message']['content']
                    buffer += content
                    # Stateful filter: drop everything inside <think>...</think> blocks
                    # These can span multiple streaming chunks, so we track state
                    while True:
                        if in_think_block:
                            end = buffer.find("</think>")
                            if end != -1:
                                buffer = buffer[end + len("</think>"):]
                                in_think_block = False
                            else:
                                buffer = ""  # discard entire chunk, still inside block
                                break
                        else:
                            start = buffer.find("<think>")
                            if start != -1:
                                # Yield text before the think block
                                before = buffer[:start]
                                if before:
                                    yield before
                                buffer = buffer[start + len("<think>"):]
                                in_think_block = True
                            else:
                                # No think tag — yield entire buffer
                                if buffer:
                                    yield buffer
                                buffer = ""
                                break
            except Exception as e:
                yield f"[Local LLM Error]: {e}. Please ensure Ollama is running (`ollama serve`)."
