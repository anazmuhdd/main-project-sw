import httpx
import json
import asyncio

async def test_ollama_openai_api():
    url = "http://localhost:11434/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # Payload following OpenAI V1 format
    # We include 'options' or specific provider flags to try and disable thinking
    payload = {
        "model": "qwen3:4b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Give extremely short answers."},
            {"role": "user", "content": "I see a laptop. Describe it in one short sentence."}
        ],
        "stream": True,
        
        # Ollama often accepts 'options' in the OpenAI compatible endpoint for model-specific tweaks
        "options": {
            "num_predict": 50,
            "temperature": 0.1
        }
    }

    print(f"--- Testing Ollama OpenAI V1 Endpoint ---")
    print(f"URL: {url}")
    print(f"Model: {payload['model']}")
    print("-" * 40)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    print(await response.aread())
                    return

                print("Stream Output:")
                full_content = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data_str)
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                print(content, end="", flush=True)
                                full_content += content
                        except Exception:
                            continue
                
                print("\n" + "-" * 40)
                print("Finished Verification.")
                
                # Check if common 'think' indicators are in the output
                think_indicators = ["<think>", "I think", "Let me see", "Reasoning:"]
                found_think = [ind for ind in think_indicators if ind.lower() in full_content.lower()]
                
                if found_think:
                    print(f"WARNING: Potential thinking detected in output: {found_think}")
                else:
                    print("SUCCESS: No obvious thinking patterns detected.")

    except Exception as e:
        print(f"Connection Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama_openai_api())
