import argparse
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
        default="test.jpg",
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
        default=Path("output.txt"),
        help="File path to write extracted text (default: %(default)s)",
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

    # Pipeline: Upscale -> Grayscale -> CLAHE -> Denoise -> Sharpen

    # 1. Upscale (3x) to improve recognition of small text
    image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. Denoise (Non-local means is good for removing grain while keeping edges)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

    # 5. Sharpening to define edges better after upscaling/denoising
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), sharpened)

    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)


def write_output(lines: List[str], destination: Path) -> None:
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
