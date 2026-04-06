import os
import sys
from pathlib import Path

# Paths
FILE = Path(__file__).resolve()
OCR_DIR = FILE.parents[0]
IMAGE_DIR = OCR_DIR / 'images'

# Add OCR dir to sys.path to import local_ocr
if str(OCR_DIR) not in sys.path:
    sys.path.append(str(OCR_DIR))

from local_ocr import run_local_ocr

def main():
    print(f"Starting Batch OCR Processing...")
    print(f"Searching for images in: {IMAGE_DIR}")
    
    # Supported image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(IMAGE_DIR.glob(ext)))
        
    if not image_files:
        print("No images found in the target directory.")
        return

    print(f"Found {len(image_files)} images for processing.")
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\n--- Processing Image {i}/{len(image_files)} ---")
        
        # Run OCR
        extracted = run_local_ocr(str(img_path))
        
        if extracted:
            print(f"Extracted Text Successful for: {img_path.name}")
        else:
            print(f"Failed to extract text for: {img_path.name}")

    print(f"\nBatch OCR Processing Complete.")

if __name__ == '__main__':
    main()
