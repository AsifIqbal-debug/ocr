"""
Create Training Labels for Bangla NID OCR
==========================================
This script helps you create accurate Bangla training labels
for fine-tuning Surya OCR on NID cards.

It displays each cropped region and lets you enter the correct Bangla/English text.
"""

import os
import sys
from pathlib import Path
import json


def load_existing_labels(labels_file: Path) -> dict:
    """Load existing labels from file."""
    labels = {}
    if labels_file.exists():
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        labels[parts[0]] = parts[1]
    return labels


def save_labels(labels: dict, labels_file: Path):
    """Save labels to file."""
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_file, 'w', encoding='utf-8') as f:
        for img_name, text in sorted(labels.items()):
            f.write(f"{img_name}\t{text}\n")


def display_image_terminal(img_path: Path):
    """Display image path for reference."""
    print(f"\n📷 Image: {img_path.name}")
    print(f"   Path: {img_path}")
    print(f"   Open in image viewer to see the text")


def get_ocr_prediction(img_path: Path) -> str:
    """Get OCR prediction from Surya for reference."""
    try:
        from PIL import Image
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        
        foundation = FoundationPredictor()
        rec_predictor = RecognitionPredictor(foundation)
        
        image = Image.open(img_path)
        results = rec_predictor([image], ["bn", "en"])
        
        if results and len(results) > 0:
            text = " ".join([line.text for line in results[0].text_lines])
            return text.strip()
    except Exception as e:
        pass
    
    return ""


def label_images_interactive(crops_dir: Path, labels_file: Path):
    """
    Interactive labeling of cropped images.
    """
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = sorted([
        f for f in crops_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ])
    
    if not images:
        print(f"No images found in {crops_dir}")
        return
    
    # Load existing labels
    labels = load_existing_labels(labels_file)
    
    print("\n" + "="*60)
    print("Bangla NID OCR Training Data Labeler")
    print("="*60)
    print(f"\nTotal images: {len(images)}")
    print(f"Already labeled: {len(labels)}")
    print(f"\nCommands:")
    print("  Enter text  → Save label")
    print("  (blank)     → Skip this image")
    print("  q           → Quit and save")
    print("  b           → Go back to previous")
    print("  d           → Delete this label")
    print("  o           → Open image in default viewer")
    print("="*60)
    
    i = 0
    while i < len(images):
        img_path = images[i]
        img_name = img_path.name
        
        display_image_terminal(img_path)
        
        # Show existing label if any
        existing = labels.get(img_name, "")
        if existing:
            print(f"   Current label: {existing}")
        
        # Get OCR prediction for reference
        # (Disabled by default to save time, uncomment if needed)
        # ocr_text = get_ocr_prediction(img_path)
        # if ocr_text:
        #     print(f"   OCR suggestion: {ocr_text}")
        
        # Get user input
        prompt = f"[{i+1}/{len(images)}] Enter Bangla/English text (q=quit, b=back, o=open): "
        try:
            user_input = input(prompt).strip()
        except EOFError:
            break
        
        if user_input.lower() == 'q':
            print("\nQuitting...")
            break
        elif user_input.lower() == 'b':
            i = max(0, i - 1)
            continue
        elif user_input.lower() == 'o':
            # Open image in default viewer
            os.startfile(str(img_path))
            continue
        elif user_input.lower() == 'd':
            if img_name in labels:
                del labels[img_name]
                print(f"   Deleted label for {img_name}")
            i += 1
            continue
        elif user_input:
            labels[img_name] = user_input
            print(f"   ✓ Saved: {user_input}")
        else:
            print("   Skipped")
        
        i += 1
        
        # Auto-save every 10 images
        if i % 10 == 0:
            save_labels(labels, labels_file)
            print(f"   [Auto-saved {len(labels)} labels]")
    
    # Final save
    save_labels(labels, labels_file)
    print(f"\n✓ Saved {len(labels)} labels to {labels_file}")


def batch_label_from_file(crops_dir: Path, labels_file: Path, input_file: Path):
    """
    Load labels from a pre-prepared file.
    
    Input file format (one per line):
    filename.jpg\tবাংলা টেক্সট
    """
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return
    
    labels = load_existing_labels(labels_file)
    
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    img_name, text = parts
                    if (crops_dir / img_name).exists():
                        labels[img_name] = text
                        count += 1
    
    save_labels(labels, labels_file)
    print(f"Imported {count} labels from {input_file}")


def create_sample_bangla_labels(labels_file: Path):
    """
    Create a sample labels file with common Bangla NID fields.
    """
    sample_labels = {
        "# Bangla NID Common Fields": "",
        "# নাম (Name):": "নামঃ",
        "# পিতা (Father):": "পিতাঃ",  
        "# মাতা (Mother):": "মাতাঃ",
        "# স্বামী (Husband):": "স্বামীঃ",
        "# জন্ম তারিখ (DOB):": "জন্ম তারিখঃ",
        "# ঠিকানা (Address):": "ঠিকানাঃ",
        "# ID No:": "ID No:",
        "# Example names:": "",
        "example_name_1": "নামঃ মোঃ শিয়াম উদ্দিন",
        "example_name_2": "পিতাঃ মোঃ আব্দুল করিম",
        "example_name_3": "মাতাঃ মোসাঃ রাহেলা বেগম",
        "example_dob": "জন্ম তারিখঃ ০১ জানু ১৯৯০",
    }
    
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(labels_file.parent / "sample_labels_reference.txt", 'w', encoding='utf-8') as f:
        f.write("# Bangla NID OCR Training - Label Reference\n")
        f.write("# ==========================================\n\n")
        f.write("# Common field prefixes:\n")
        f.write("# নাম (Name)\n")
        f.write("# পিতা (Father)\n")
        f.write("# মাতা (Mother)\n")
        f.write("# স্বামী (Husband)\n")
        f.write("# জন্ম তারিখ (Date of Birth)\n\n")
        f.write("# Common suffixes:\n")
        f.write("# ঃ (visarga - used after labels like নামঃ)\n\n")
        f.write("# Numbers (Bangla):\n")
        f.write("# ০১২৩৪৫৬৭৮৯\n\n")
        f.write("# Example entries (filename<TAB>text):\n")
        f.write("crop_001.jpg\tনামঃ মোঃ আরিফুল ইসলাম\n")
        f.write("crop_002.jpg\tপিতাঃ মোঃ করিম উদ্দিন\n")
        f.write("crop_003.jpg\tমাতাঃ মোসাঃ ফাতেমা বেগম\n")
        f.write("crop_004.jpg\tজন্ম তারিখঃ ০১ জানু ১৯৯০\n")
        f.write("crop_005.jpg\tID No: 1234567890123\n")
    
    print(f"Sample reference created: {labels_file.parent / 'sample_labels_reference.txt'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create training labels for Bangla NID OCR")
    parser.add_argument("--crops-dir", type=Path, 
                       default=Path("training/data/raw"),
                       help="Directory containing cropped images")
    parser.add_argument("--labels-file", type=Path,
                       default=Path("training/data/annotations/labels.txt"),
                       help="Output labels file")
    parser.add_argument("--import-file", type=Path, default=None,
                       help="Import labels from existing file")
    parser.add_argument("--create-sample", action="store_true",
                       help="Create sample labels reference file")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_bangla_labels(args.labels_file)
        return
    
    if args.import_file:
        batch_label_from_file(args.crops_dir, args.labels_file, args.import_file)
        return
    
    # Interactive labeling
    if not args.crops_dir.exists():
        print(f"Crops directory not found: {args.crops_dir}")
        print("\nFirst, run auto_crop.py to extract text regions:")
        print("  python training/scripts/auto_crop.py <nid_image.jpg>")
        sys.exit(1)
    
    label_images_interactive(args.crops_dir, args.labels_file)


if __name__ == "__main__":
    main()
