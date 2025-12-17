"""
OCR using TrOCR (Transformer-based OCR) from Microsoft
=======================================================
TrOCR is a state-of-the-art OCR model using Vision Transformer (ViT) 
and Text Transformer architecture.

For Bengali, we use the multilingual model or fine-tuned Bengali model.

Usage:
    python ocr_trocr.py image.jpg
    python ocr_trocr.py image.jpg --output result.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from PIL import Image
import torch

# Check for transformers
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:
    print("Installing transformers...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def classify_text(text: str) -> str:
    """Classify text type."""
    bangla_count = len(re.findall(r'[\u0980-\u09FF]', text))
    english_count = len(re.findall(r'[a-zA-Z]', text))
    digit_count = len(re.findall(r'[0-9]', text))
    
    total = len(text.replace(" ", ""))
    if total == 0:
        return "unknown"
    
    if digit_count > total * 0.5:
        return "number"
    if bangla_count > english_count:
        return "bangla"
    if english_count > bangla_count:
        return "english"
    return "mixed"


class BengaliOCR:
    """Bengali OCR using TrOCR or similar transformer models."""
    
    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        """
        Initialize the OCR model.
        
        Available models:
        - microsoft/trocr-base-printed (English printed text)
        - microsoft/trocr-base-handwritten (English handwritten)
        - For Bengali, we'll use detection + recognition pipeline
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        print(f"Loading TrOCR model: {model_name}...")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def recognize_crop(self, image: Image.Image) -> str:
        """Recognize text from a cropped image region."""
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_length=64)
        
        # Decode
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    
    def ocr_with_detection(self, image_path: str) -> list:
        """
        Full OCR pipeline: detect text regions, then recognize each.
        Uses EasyOCR for detection and TrOCR for recognition.
        """
        import easyocr
        import cv2
        import numpy as np
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read {image_path}")
        
        # Use EasyOCR for detection only
        print("Detecting text regions...")
        reader = easyocr.Reader(['bn', 'en'], gpu=torch.cuda.is_available())
        detections = reader.readtext(img, detail=1)
        
        items = []
        pil_img = Image.open(image_path).convert('RGB')
        
        print(f"Recognizing {len(detections)} text regions...")
        for bbox, easyocr_text, conf in detections:
            # Get bounding box
            pts = [[int(p[0]), int(p[1])] for p in bbox]
            x_min = max(0, min(p[0] for p in pts) - 5)
            y_min = max(0, min(p[1] for p in pts) - 5)
            x_max = min(img.shape[1], max(p[0] for p in pts) + 5)
            y_max = min(img.shape[0], max(p[1] for p in pts) + 5)
            
            # Crop region
            crop = pil_img.crop((x_min, y_min, x_max, y_max))
            
            # Try TrOCR recognition
            try:
                trocr_text = self.recognize_crop(crop)
                # Use TrOCR if it gives reasonable result, else fall back to EasyOCR
                if trocr_text and len(trocr_text) > 0:
                    text = trocr_text
                else:
                    text = easyocr_text
            except:
                text = easyocr_text
            
            if text:
                items.append({
                    "text": text,
                    "type": classify_text(text),
                    "easyocr": easyocr_text,  # Keep original for comparison
                })
        
        return items


def run_simple_ocr(image_path: str) -> list:
    """
    Simple OCR using just EasyOCR with optimized settings.
    This is the most reliable for Bengali NID cards.
    """
    import easyocr
    import cv2
    
    print("Loading EasyOCR with Bengali + English...")
    reader = easyocr.Reader(['bn', 'en'], gpu=torch.cuda.is_available())
    
    # Read and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")
    
    # Upscale small images
    h, w = img.shape[:2]
    if w < 1500:
        scale = 2
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("Running OCR...")
    results = reader.readtext(
        img_rgb,
        detail=1,
        paragraph=True,  # Merge paragraphs
        contrast_ths=0.1,
        adjust_contrast=0.5,
        text_threshold=0.5,
    )
    
    items = []
    for bbox, text, conf in results:
        text = text.strip()
        if text:
            items.append({
                "text": text,
                "type": classify_text(text),
                "confidence": round(conf, 3)
            })
    
    return items


def main():
    parser = argparse.ArgumentParser(description="Bengali OCR using TrOCR/EasyOCR")
    parser.add_argument("image", help="Image file to process")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    parser.add_argument("--method", choices=["simple", "trocr"], default="simple",
                       help="OCR method: simple (EasyOCR) or trocr (hybrid)")
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    print(f"Processing {args.image}...")
    print("="*50)
    
    try:
        if args.method == "trocr":
            ocr = BengaliOCR()
            items = ocr.ocr_with_detection(args.image)
        else:
            items = run_simple_ocr(args.image)
        
        output = {"items": items}
        
        # Pretty print
        output_str = json.dumps(output, ensure_ascii=False, indent=2)
        print("\nResults:")
        print(output_str)
        
        # Save if requested
        if args.output:
            args.output.write_text(output_str, encoding="utf-8")
            print(f"\nSaved to {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
