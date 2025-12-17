"""
OCR using Google Cloud Vision API
==================================
Google Vision has excellent Bangla support and is much more accurate
than EasyOCR for NID cards.

Setup:
1. Create a Google Cloud project: https://console.cloud.google.com
2. Enable Vision API: https://console.cloud.google.com/apis/library/vision.googleapis.com
3. Create a service account and download JSON key
4. Set environment variable: GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json

Or use API key method (simpler):
1. Create API key: https://console.cloud.google.com/apis/credentials
2. Pass --api-key YOUR_KEY

Usage:
    python ocr_google.py image.jpg --api-key YOUR_API_KEY
    python ocr_google.py image.jpg --output result.json
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


def ocr_with_api_key(image_path: str, api_key: str) -> dict:
    """
    Call Google Cloud Vision API using API key (simpler setup).
    """
    # Read and encode image
    with open(image_path, "rb") as f:
        image_content = base64.b64encode(f.read()).decode("utf-8")

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

    payload = {
        "requests": [
            {
                "image": {"content": image_content},
                "features": [
                    {"type": "TEXT_DETECTION"},
                    {"type": "DOCUMENT_TEXT_DETECTION"}
                ],
                "imageContext": {
                    "languageHints": ["bn", "en"]  # Bangla and English
                }
            }
        ]
    }

    response = requests.post(url, json=payload)

    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")

    return response.json()


def parse_google_response(response: dict) -> list:
    """
    Parse Google Vision API response into structured text items.
    """
    items = []

    if "responses" not in response or not response["responses"]:
        return items

    result = response["responses"][0]

    # Get full text annotation (better structure)
    if "fullTextAnnotation" in result:
        full_text = result["fullTextAnnotation"]["text"]

        # Split by lines and process
        for line in full_text.strip().split("\n"):
            line = line.strip()
            if line:
                items.append({
                    "text": line,
                    "type": classify_text(line)
                })

    # Fallback to text annotations
    elif "textAnnotations" in result:
        # First annotation is the full text, rest are individual words
        if result["textAnnotations"]:
            full_text = result["textAnnotations"][0]["description"]
            for line in full_text.strip().split("\n"):
                line = line.strip()
                if line:
                    items.append({
                        "text": line,
                        "type": classify_text(line)
                    })

    return items


def classify_text(text: str) -> str:
    """Classify text type based on content."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Google Cloud Vision OCR for NID cards")
    parser.add_argument("image", help="Image file to process")
    parser.add_argument("--api-key", required=True,
                        help="Google Cloud API key")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    parser.add_argument("--raw", action="store_true",
                        help="Output raw API response")

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    print(f"Processing {args.image} with Google Cloud Vision...")

    try:
        response = ocr_with_api_key(args.image, args.api_key)

        if args.raw:
            output = response
        else:
            items = parse_google_response(response)
            output = {"items": items}

        # Pretty print
        output_str = json.dumps(output, ensure_ascii=False, indent=2)
        print(output_str)

        # Save if requested
        if args.output:
            args.output.write_text(output_str, encoding="utf-8")
            print(f"\nSaved to {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
