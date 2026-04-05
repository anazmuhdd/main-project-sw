import requests, base64, io
import os
from PIL import Image

invoke_url = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1"

def get_resized_b64(file_path, limit=180_000):
    """Encodes an image to base64, resizing it iteratively if it exceeds character limit."""
    with open(file_path, "rb") as f:
        img_data = f.read()
    
    image_b64 = base64.b64encode(img_data).decode()
    
    if len(image_b64) < limit:
        print(f"Image fits within limit ({len(image_b64)} chars).")
        return image_b64
    
    print(f"Image too large ({len(image_b64)} chars). Resizing...")
    img = Image.open(io.BytesIO(img_data))
    
    # Iteratively reduce dimensions until the base64 string fits
    scale = 0.9
    while len(image_b64) >= limit:
        w, h = img.size
        new_size = (int(w * scale), int(h * scale))
        if new_size[0] < 10 or new_size[1] < 10:
            break
            
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        # Save as JPEG with optimized quality to reduce file size further
        img.save(buffered, format="JPEG", quality=85, optimize=True)
        image_b64 = base64.b64encode(buffered.getvalue()).decode()
        print(f"Resized to {new_size}. New length: {len(image_b64)}")
        
    return image_b64

# Get the base64 encoded image (resized if necessary)
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "9ube35adqzf21.jpg")
image_b64 = get_resized_b64(image_path)


# Ensure it's under the limit or warn
if len(image_b64) >= 180_000:
    print("Warning: Image still exceeds limit after resizing. API may fail.")

headers = {
  "Authorization": f"Bearer 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",
  "Accept": "application/json"
}

payload = {
  "input": [
    {
      "type": "image_url",
      "url": f"data:image/jpeg;base64,{image_b64}"
    }
  ]
}

print("Invoking NVIDIA OCR API...")
response = requests.post(invoke_url, headers=headers, json=payload)
print(response.json())

