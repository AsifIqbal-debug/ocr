"""
OCR using Tesseract with Bangla support
========================================
Tesseract is free, offline, and has good Bangla language support.

Setup on Windows:
1. Download Tesseract installer from:
   https://github.com/UB-Mannheim/tesseract/wiki
   
2. Install with "Additional language data" - select Bengali

3. Add to PATH or set TESSERACT_CMD environment variable

4. Install Python wrapper:
   pip install pytesseract pillow

Usage:
    python ocr_tesseract.py image.jpg
    python ocr_tesseract.py image.jpg --output result.json
"""

import argparse
import json
import re
import sys
import subprocess
from pathlib import Path

# Check for pytesseract
try:
    import pytesseract
    from PIL import Image
except ImportError:
    print("Installing pytesseract and pillow...")
    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "pytesseract", "pillow"])
    import pytesseract
    from PIL import Image

# Try to find Tesseract
import shutil
tesseract_paths = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    shutil.which("tesseract")
]

for path in tesseract_paths:
    if path and Path(path).exists():
        pytesseract.pytesseract.tesseract_cmd = path
        break


def check_tesseract():
    """Check if Tesseract is installed and has Bengali support."""
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")

        # Check for Bengali language
        langs = pytesseract.get_languages()
        has_bengali = 'ben' in langs or 'ben_old' in langs

        if not has_bengali:
            print("\nWarning: Bengali language pack not found!")
            print("Available languages:", langs)
            print("\nTo install Bengali support:")
            print("1. Re-run Tesseract installer")
            print("2. Select 'Additional language data'")
            print("3. Check 'Bengali' and 'Bengali (old)'")
            return False

        print(f"Bengali support: {'ben' in langs}")
        return True

    except Exception as e:
        print(f"Error: Tesseract not found - {e}")
        print("\nPlease install Tesseract:")
        print("https://github.com/UB-Mannheim/tesseract/wiki")
        return False


def preprocess_image(image_path: str) -> Image.Image:
    """Preprocess image for better OCR results."""
    import cv2
    import numpy as np

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    # Upscale for small images
    h, w = img.shape[:2]
    if w < 1500:
        scale = 2
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter (preserves edges)
    filtered = cv2.bilateralFilter(gray, 11, 75, 75)

    # Adaptive thresholding for better text contrast
    thresh = cv2.adaptiveThreshold(
        filtered, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 10
    )

    # Convert back to PIL
    return Image.fromarray(thresh)


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


def run_tesseract_ocr(image_path: str, preprocess: bool = True) -> list:
    """
    Run Tesseract OCR with Bengali + English.
    """
    if preprocess:
        try:
            img = preprocess_image(image_path)
        except Exception as e:
            print(f"Preprocessing failed, using original: {e}")
            img = Image.open(image_path)
    else:
        img = Image.open(image_path)

    # Configure Tesseract
    # PSM 6 = Assume uniform block of text
    # PSM 3 = Fully automatic page segmentation (default)
    custom_config = r'--oem 3 --psm 3'

    # Run OCR with Bengali + English
    try:
        text = pytesseract.image_to_string(
            img, lang='ben+eng', config=custom_config)
    except Exception as e:
        print(f"Bengali OCR failed, trying English only: {e}")
        text = pytesseract.image_to_string(
            img, lang='eng', config=custom_config)

    # Parse lines
    items = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if line and len(line) > 1:  # Filter noise
            items.append({
                "text": line,
                "type": classify_text(line)
            })

    return items


def main():
    parser = argparse.ArgumentParser(
        description="Tesseract OCR for NID cards (Bangla + English)")
    parser.add_argument("image", nargs="?", help="Image file to process")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip image preprocessing")
    parser.add_argument("--check", action="store_true",
                        help="Check Tesseract installation")

    args = parser.parse_args()

    if args.check or not args.image:
        check_tesseract()
        if not args.image:
            print("\nUsage: python ocr_tesseract.py image.jpg")
        return

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    if not check_tesseract():
        sys.exit(1)

    print(f"\nProcessing {args.image}...")

    items = run_tesseract_ocr(args.image, preprocess=not args.no_preprocess)
    output = {"items": items}

    # Pretty print
    output_str = json.dumps(output, ensure_ascii=False, indent=2)
    print(output_str)

    # Save if requested
    if args.output:
        args.output.write_text(output_str, encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
