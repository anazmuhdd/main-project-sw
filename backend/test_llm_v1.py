from openai import OpenAI
import json

def test_openai_client_with_ollama():
    # Initialize the OpenAI client pointing to Ollama
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Ollama doesn't require a real key, but the client needs something
    )

    model_name = "qwen3.5:2b"
    
    print(f"--- Testing OpenAI Client with Ollama ---")
    print(f"Model: {model_name}")
    print("-" * 40)

    try:
        # Using the Chat Completions template as requested
        # We try to pass model-specific flags/messages to verify thinking is disabled
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Provide direct answers without any internal reasoning or <think> tags."},
                {"role": "user", "content": "I see a laptop. Describe it in one short sentence."}
            ],
            stream=True,
            reasoning_effort="low",
            extra_body={
                "think": False # Passing the custom flag in extra_body for the provider to see
            }
        )

        print("Stream Output:")
        full_content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_content += content
        
        print("\n" + "-" * 40)
        print("Finished Verification.")
        
        # Check for thinking artifacts
        think_indicators = ["<think>", "</think>"]
        found_think = [ind for ind in think_indicators if ind in full_content]
        
        if found_think:
            print(f"FAILED: Thinking tags were found in the output: {found_think}")
        else:
            print("SUCCESS: No thinking tags detected in the streamed output.")

    except Exception as e:
        print(f"Error using OpenAI client: {e}")

if __name__ == "__main__":
    test_openai_client_with_ollama()
