"""
Combined OCR for Bengali NID Cards
===================================
Uses multiple OCR engines and combines results for better accuracy.

Usage:
    python ocr_combined.py image.jpg
    python ocr_combined.py image.jpg --output result.json
"""

import argparse
import json
import re
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


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


def preprocess_image(image_path: str, method: str = "default") -> np.ndarray:
    """
    Advanced image preprocessing for NID cards.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")
    
    # Upscale small images
    h, w = img.shape[:2]
    if w < 1500:
        scale = 3 if w < 800 else 2
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    if method == "default":
        return img
    
    elif method == "grayscale":
        # Convert to grayscale with enhanced contrast
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE for adaptive contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR for consistency
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    elif method == "binarize":
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 11, 75, 75)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            filtered, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21, 10
        )
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    elif method == "sharpen":
        # Sharpen the image
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        return sharpened
    
    return img


def run_easyocr(img_bgr: np.ndarray) -> list:
    """Run EasyOCR on image."""
    import easyocr
    
    reader = easyocr.Reader(['bn', 'en'], gpu=False)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    results = reader.readtext(
        img_rgb,
        detail=1,
        paragraph=False,
        contrast_ths=0.1,
        adjust_contrast=0.5,
        text_threshold=0.5,
    )
    
    items = []
    for bbox, text, conf in results:
        text = text.strip()
        if text and len(text) > 1:
            items.append({
                "text": text,
                "confidence": conf,
                "source": "easyocr"
            })
    
    return items


def run_surya(img_bgr: np.ndarray) -> list:
    """Run Surya OCR on image."""
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    
    foundation = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation)
    
    # Convert BGR to RGB PIL Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_rgb)
    
    rec_results = rec_predictor(
        [image],
        det_predictor=det_predictor,
        sort_lines=True,
        math_mode=False,
    )
    
    items = []
    if rec_results and len(rec_results) > 0:
        for line in rec_results[0].text_lines:
            text = line.text.strip()
            text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
            text = " ".join(text.split())
            
            if text and len(text) > 1:
                conf = line.confidence if hasattr(line, 'confidence') else 0
                items.append({
                    "text": text,
                    "confidence": conf,
                    "source": "surya"
                })
    
    return items


def is_valid_bengali(text: str) -> bool:
    """Check if text contains valid Bengali characters (not Hindi)."""
    # Hindi/Devanagari range: U+0900 to U+097F
    # Bengali range: U+0980 to U+09FF
    
    hindi_count = len(re.findall(r'[\u0900-\u097F]', text))
    bengali_count = len(re.findall(r'[\u0980-\u09FF]', text))
    
    # If there's Hindi but no Bengali, it's likely misrecognized
    if hindi_count > 0 and bengali_count == 0:
        return False
    
    return True


def combine_results(easyocr_results: list, surya_results: list) -> list:
    """
    Combine results from multiple OCR engines.
    Prefer Bengali text from EasyOCR (explicit Bengali support).
    Prefer English/numbers from Surya (better accuracy).
    """
    combined = []
    
    # Process Surya results first
    for item in surya_results:
        text = item["text"]
        
        # Skip if it's Hindi (Surya sometimes confuses Bengali with Hindi)
        if not is_valid_bengali(text):
            continue
        
        text_type = classify_text(text)
        
        # For English/numbers, prefer Surya
        if text_type in ["english", "number"]:
            combined.append({
                "text": text,
                "type": text_type,
                "confidence": item["confidence"],
                "source": "surya"
            })
        # For Bengali, check if EasyOCR has better result
        elif text_type == "bangla":
            # Find matching EasyOCR result by position/similarity
            found_better = False
            for easy_item in easyocr_results:
                easy_type = classify_text(easy_item["text"])
                if easy_type == "bangla" and easy_item["confidence"] > 0.5:
                    # EasyOCR has Bengali - add it
                    if easy_item["text"] not in [c["text"] for c in combined]:
                        combined.append({
                            "text": easy_item["text"],
                            "type": "bangla",
                            "confidence": easy_item["confidence"],
                            "source": "easyocr"
                        })
                        found_better = True
            
            if not found_better:
                combined.append({
                    "text": text,
                    "type": text_type,
                    "confidence": item["confidence"],
                    "source": "surya"
                })
    
    # Add any Bengali from EasyOCR not already included
    for item in easyocr_results:
        text_type = classify_text(item["text"])
        if text_type == "bangla":
            if item["text"] not in [c["text"] for c in combined]:
                combined.append({
                    "text": item["text"],
                    "type": text_type,
                    "confidence": item["confidence"],
                    "source": "easyocr"
                })
    
    return combined


def run_combined_ocr(image_path: str) -> list:
    """
    Run combined OCR with multiple preprocessing methods.
    """
    print("Running combined OCR pipeline...")
    
    all_results = []
    
    # Try different preprocessing methods
    methods = ["default", "grayscale", "binarize"]
    
    for method in methods:
        print(f"\n--- Preprocessing: {method} ---")
        
        try:
            img = preprocess_image(image_path, method)
            
            # Run EasyOCR (better for Bengali)
            print("Running EasyOCR...")
            easy_results = run_easyocr(img)
            print(f"  Found {len(easy_results)} items")
            
            # Run Surya (better for English/layout)
            print("Running Surya...")
            surya_results = run_surya(img)
            print(f"  Found {len(surya_results)} items")
            
            # Combine results
            combined = combine_results(easy_results, surya_results)
            all_results.extend(combined)
            
        except Exception as e:
            print(f"  Error with {method}: {e}")
    
    # Deduplicate and sort by confidence
    seen = set()
    unique_results = []
    for item in sorted(all_results, key=lambda x: x["confidence"], reverse=True):
        if item["text"] not in seen:
            seen.add(item["text"])
            unique_results.append(item)
    
    return unique_results


def main():
    parser = argparse.ArgumentParser(description="Combined OCR for Bengali NID")
    parser.add_argument("image", help="Image file to process")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    parser.add_argument("--simple", action="store_true", 
                       help="Use simple mode (EasyOCR only)")
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    print(f"Processing {args.image}...")
    print("="*50)
    
    try:
        if args.simple:
            img = preprocess_image(args.image, "grayscale")
            items = run_easyocr(img)
            for item in items:
                item["type"] = classify_text(item["text"])
        else:
            items = run_combined_ocr(args.image)
        
        output = {"items": items}
        
        # Pretty print
        output_str = json.dumps(output, ensure_ascii=False, indent=2)
        print("\n" + "="*50)
        print("Final Results:")
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
