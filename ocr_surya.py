"""
OCR using Surya - Modern ML-based OCR
======================================
Surya is a state-of-the-art multilingual OCR with excellent accuracy.
It supports 90+ languages including Bangla.

Features:
- Text detection (where is text)
- Text recognition (what does it say)
- Layout analysis
- Reading order detection
- Post-processing corrections for common Bangla OCR errors

Usage:
    python ocr_surya.py image.jpg
    python ocr_surya.py image.jpg --output result.json
    python ocr_surya.py image.jpg --no-correct  # Disable corrections
"""

import argparse
import json
import re
import sys
from pathlib import Path
from PIL import Image

# Import correction system
try:
    from bangla_corrections import BanglaOCRCorrector
    CORRECTOR_AVAILABLE = True
except ImportError:
    CORRECTOR_AVAILABLE = False


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


def run_surya_ocr(image_path: str, languages: list = None) -> list:
    """
    Run Surya OCR on an image.
    
    Args:
        image_path: Path to image file
        languages: List of language codes (e.g., ['bn', 'en']) - not used in v0.17+
    
    Returns:
        List of detected text items with type classification
    """
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    import cv2
    import numpy as np
    
    print("Loading Surya models...")
    
    # Initialize foundation predictor first (required for recognition)
    foundation = FoundationPredictor()
    
    # Initialize predictors
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation)
    
    # Load and preprocess image for better OCR
    print("Preprocessing image...")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")
    
    # Upscale small images
    h, w = img.shape[:2]
    if w < 1500:
        scale = 2 if w >= 800 else 3
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        print(f"Upscaled image {scale}x")
    
    # Convert BGR to RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_rgb)
    
    print("Running OCR...")
    
    # Run OCR with task_names for Bengali
    rec_results = rec_predictor(
        [image],
        det_predictor=det_predictor,
        sort_lines=True,
        math_mode=False,  # Disable math mode for plain text
    )
    
    # Parse results
    items = []
    
    if rec_results and len(rec_results) > 0:
        result = rec_results[0]
        
        for line in result.text_lines:
            text = line.text.strip()
            # Clean up HTML tags that Surya sometimes adds
            text = text.replace("<br>", " ").replace("<br/>", " ")
            text = re.sub(r'<[^>]+>', '', text)  # Remove any HTML tags
            text = " ".join(text.split())  # Normalize whitespace
            
            if text:
                conf = line.confidence if hasattr(line, 'confidence') else 0
                items.append({
                    "text": text,
                    "type": classify_text(text),
                    "confidence": round(conf, 3)
                })
    
    return items


def apply_corrections(items: list) -> list:
    """Apply Bangla OCR corrections to results."""
    if not CORRECTOR_AVAILABLE:
        print("Warning: Correction module not available")
        return items
    
    corrector = BanglaOCRCorrector()
    corrected_items = []
    
    for item in items:
        original = item['text']
        corrected = corrector.correct(original)
        new_item = item.copy()
        new_item['text'] = corrected
        if corrected != original:
            new_item['original'] = original
            new_item['corrected'] = True
        corrected_items.append(new_item)
    
    return corrected_items


def main():
    parser = argparse.ArgumentParser(
        description="Surya OCR for NID cards (Bangla + English)")
    parser.add_argument("image", help="Image file to process")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    parser.add_argument("--languages", nargs="+", default=["bn", "en"],
                        help="Languages to detect (default: bn en)")
    parser.add_argument("--no-correct", action="store_true",
                        help="Disable post-processing corrections")

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    print(f"Processing {args.image} with Surya OCR...")
    print("="*50)

    try:
        items = run_surya_ocr(args.image, args.languages)
        
        # Apply corrections unless disabled
        if not args.no_correct:
            items = apply_corrections(items)
            print("✓ Applied Bangla corrections")
        
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
