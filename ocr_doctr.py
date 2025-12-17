"""
OCR using docTR (Document Text Recognition)
============================================
docTR is a modern OCR library with excellent accuracy.
It uses deep learning models for both detection and recognition.

Usage:
    python ocr_doctr.py image.jpg
    python ocr_doctr.py image.jpg --output result.json
"""

import argparse
import json
import re
import sys
from pathlib import Path


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


def run_doctr_ocr(image_path: str) -> list:
    """
    Run docTR OCR on an image.
    """
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    
    print("Loading docTR models...")
    
    # Create OCR predictor
    # Uses DBNet for detection and CRNN/ViTSTR for recognition
    model = ocr_predictor(pretrained=True)
    
    # Load image
    print(f"Processing {image_path}...")
    doc = DocumentFile.from_images(image_path)
    
    # Run OCR
    result = model(doc)
    
    # Parse results
    items = []
    
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                # Combine words in the line
                words = [word.value for word in line.words]
                text = " ".join(words).strip()
                
                if text:
                    # Get average confidence
                    confidences = [word.confidence for word in line.words]
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0
                    
                    items.append({
                        "text": text,
                        "type": classify_text(text),
                        "confidence": round(avg_conf, 3)
                    })
    
    return items


def main():
    parser = argparse.ArgumentParser(description="docTR OCR for documents")
    parser.add_argument("image", help="Image file to process")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    print(f"Processing {args.image} with docTR...")
    print("="*50)
    
    try:
        items = run_doctr_ocr(args.image)
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
