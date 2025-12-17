"""
Auto-crop NID text regions using EasyOCR detection
==================================================
This script uses EasyOCR to detect text regions and auto-crop them,
then you just need to verify/correct the ground truth.

Much faster than manual cropping!
"""

import cv2
import easyocr
import argparse
import uuid
from pathlib import Path


def auto_crop(image_path: str, output_dir: str = "training/data/raw", 
              min_confidence: float = 0.3):
    """
    Auto-detect and crop text regions from NID card.
    Uses EasyOCR for detection, saves crops for labeling.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    labels_file = output_path.parent / "annotations" / "labels.txt"
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read {image_path}")
        return
    
    # Run EasyOCR with detail mode
    print("Running text detection...")
    reader = easyocr.Reader(['bn', 'en'], gpu=False)
    results = reader.readtext(img, detail=1)
    
    print(f"\nFound {len(results)} text regions")
    print("="*60)
    
    saved_count = 0
    
    for i, (bbox, text, conf) in enumerate(results):
        if conf < min_confidence:
            continue
        
        # Get bounding box
        pts = [[int(p[0]), int(p[1])] for p in bbox]
        x_min = max(0, min(p[0] for p in pts) - 5)
        y_min = max(0, min(p[1] for p in pts) - 5)
        x_max = min(img.shape[1], max(p[0] for p in pts) + 5)
        y_max = min(img.shape[0], max(p[1] for p in pts) + 5)
        
        # Crop
        crop = img[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue
        
        # Show crop and OCR result
        print(f"\n[{i+1}] OCR detected: '{text}' (conf: {conf:.2f})")
        
        # Display crop (optional)
        # cv2.imshow(f"Region {i+1}", crop)
        # cv2.waitKey(100)
        
        # Ask user for correction
        print("Enter correct text (or 'skip' to skip, 'quit' to stop):")
        correct_text = input(f"[{text}] -> ").strip()
        
        if correct_text.lower() == 'quit':
            break
        elif correct_text.lower() == 'skip' or not correct_text:
            correct_text = text  # Use OCR result if no correction
        
        # Classify field type from content
        if any(c.isdigit() for c in correct_text) and len(correct_text) > 8:
            field_type = "id"
        elif "birth" in correct_text.lower() or any(m in correct_text.lower() for m in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
            field_type = "dob"
        elif correct_text.isupper() and correct_text.isascii():
            field_type = "name_en"
        else:
            field_type = "text"
        
        # Save crop
        uid = str(uuid.uuid4())[:8]
        filename = f"{field_type}_{uid}.jpg"
        filepath = output_path / filename
        
        cv2.imwrite(str(filepath), crop)
        
        # Save label
        with open(labels_file, "a", encoding="utf-8") as f:
            f.write(f"{filename}\t{correct_text}\n")
        
        saved_count += 1
        print(f"✓ Saved: {filename}")
    
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"Done! Saved {saved_count} crops to {output_path}")
    print(f"Labels appended to {labels_file}")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Review labels in {labels_file}")
    print(f"2. Run: python training/scripts/prepare_data.py prepare")
    print(f"3. Run: python training/scripts/train_easyocr.py")


def auto_crop_batch(image_paths: list, output_dir: str = "training/data/raw",
                    min_confidence: float = 0.3, no_prompt: bool = False):
    """
    Process multiple images in batch mode.
    
    Args:
        image_paths: List of image paths to process
        output_dir: Output directory for crops
        min_confidence: Minimum OCR confidence threshold
        no_prompt: If True, save crops without asking for corrections
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    labels_file = output_path.parent / "annotations" / "labels.txt"
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize EasyOCR once
    print("Loading EasyOCR (Bangla + English)...")
    reader = easyocr.Reader(['bn', 'en'], gpu=False)
    
    total_saved = 0
    
    for image_path in image_paths:
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print('='*60)
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Cannot read {image_path}")
            continue
        
        # Run detection
        results = reader.readtext(img, detail=1)
        print(f"Found {len(results)} text regions")
        
        for i, (bbox, text, conf) in enumerate(results):
            if conf < min_confidence:
                continue
            
            # Get bounding box
            pts = [[int(p[0]), int(p[1])] for p in bbox]
            x_min = max(0, min(p[0] for p in pts) - 5)
            y_min = max(0, min(p[1] for p in pts) - 5)
            x_max = min(img.shape[1], max(p[0] for p in pts) + 5)
            y_max = min(img.shape[0], max(p[1] for p in pts) + 5)
            
            # Crop
            crop = img[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue
            
            if no_prompt:
                # Save without prompting
                correct_text = text
            else:
                print(f"\n[{i+1}] OCR: '{text}' (conf: {conf:.2f})")
                correct_text = input(f"Correct text (Enter=accept, skip/quit): ").strip()
                
                if correct_text.lower() == 'quit':
                    return total_saved
                elif correct_text.lower() == 'skip':
                    continue
                elif not correct_text:
                    correct_text = text
            
            # Save crop
            uid = str(uuid.uuid4())[:8]
            src_name = Path(image_path).stem[:10]
            filename = f"{src_name}_{uid}.jpg"
            filepath = output_path / filename
            
            cv2.imwrite(str(filepath), crop)
            
            # Save label
            with open(labels_file, "a", encoding="utf-8") as f:
                f.write(f"{filename}\t{correct_text}\n")
            
            total_saved += 1
            if no_prompt:
                print(f"  [{i+1}] Saved: {filename} -> {text[:30]}...")
    
    print(f"\n{'='*60}")
    print(f"Total saved: {total_saved} crops")
    print(f"Labels: {labels_file}")
    print(f"\nNext steps:")
    print(f"1. Review/correct labels: python training/scripts/create_labels.py")
    print(f"2. Train model: python training/train_surya_bangla.py")
    return total_saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-crop NID text regions")
    parser.add_argument("images", nargs="+", help="NID card image(s) to process")
    parser.add_argument("--output", default="training/data/raw",
                       help="Output directory for crops")
    parser.add_argument("--min-conf", type=float, default=0.3,
                       help="Minimum confidence threshold")
    parser.add_argument("--no-prompt", action="store_true",
                       help="Save all crops without prompting for corrections")
    
    args = parser.parse_args()
    
    if args.no_prompt:
        auto_crop_batch(args.images, args.output, args.min_conf, no_prompt=True)
    elif len(args.images) == 1:
        auto_crop(args.images[0], args.output, args.min_conf)
    else:
        auto_crop_batch(args.images, args.output, args.min_conf)
