import argparse
import json
from pathlib import Path
from typing import List

import cv2
import easyocr
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EasyOCR on a Bangla/English image and print the detected text."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="Mijja.jpeg",
        help="Path to the image file (default: %(default)s)",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable built-in denoise/threshold preprocessing",
    )
    parser.add_argument(
        "--save-preprocessed",
        type=Path,
        help="Optional path to save the enhanced grayscale image for inspection",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output2.txt"),
        help="File path to write extracted text (default: %(default)s)",
    )
    parser.add_argument(
        "--engine",
        choices=["easyocr", "paddle"],
        default="easyocr",
        help="OCR backend to invoke (default: %(default)s)",
    )
    parser.add_argument(
        "--paddle-det",
        type=Path,
        help="Optional path to PaddleOCR detection inference model directory",
    )
    parser.add_argument(
        "--paddle-rec",
        type=Path,
        help="Optional path to PaddleOCR recognition inference model directory",
    )
    parser.add_argument(
        "--paddle-dict",
        type=Path,
        help="Character dictionary to match the fine-tuned Paddle recognition model",
    )
    parser.add_argument(
        "--paddle-angle",
        action="store_true",
        help="Enable direction classifier when using PaddleOCR",
    )
    args = parser.parse_args()
    args.image = Path(args.image).expanduser().resolve()
    if not args.image.exists():
        parser.error(f"Image not found: {args.image}")
    if args.save_preprocessed:
        args.save_preprocessed = args.save_preprocessed.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    return args


def enhance_image(img_path: Path, enable: bool, save_path: Path | None) -> np.ndarray:
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image data from {img_path}")
    if not enable:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Pipeline: Upscale -> Grayscale -> CLAHE -> Denoise
    # This avoids binary thresholding which can lose details in complex backgrounds

    # 1. Upscale (2x) to improve recognition of small text
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. Denoise (Non-local means is good for removing grain while keeping edges)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), denoised)

    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)


import re

def classify_text(text: str) -> str:
    # Count character types
    bangla_count = len(re.findall(r'[\u0980-\u09FF]', text))
    english_count = len(re.findall(r'[a-zA-Z]', text))
    digit_count = len(re.findall(r'[0-9]', text))
    
    total = len(text.replace(" ", ""))
    if total == 0:
        return "unknown"
    
    # Determine dominant type
    if digit_count > total * 0.5:
        return "number"
    if bangla_count > english_count:
        return "bangla"
    if english_count > bangla_count:
        return "english"
    return "mixed"

def write_output(lines: List[str], destination: Path) -> None:
    items = []
    for line in lines:
        items.append({
            "text": line,
            "type": classify_text(line)
        })
    
    payload = {"items": items}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    processed_image = enhance_image(
        args.image, enable=not args.no_enhance, save_path=args.save_preprocessed)

    # Load OCR reader for Bangla + English
    reader = easyocr.Reader(['bn', 'en'])

    # Run OCR
    result = reader.readtext(processed_image, detail=0)
    write_output(result, args.output)

    # Print output
    print("\n--- Extracted Text ---")
    for line in result:
        print(line)

    print(f"\nSaved {len(result)} lines to {args.output}")


if __name__ == "__main__":
    main()
