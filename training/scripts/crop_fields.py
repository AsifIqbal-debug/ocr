"""
NID Card Text Field Cropper
===========================
Helps you create training data by cropping text fields from NID card images.
Each field (name, father, mother, DOB, ID) is saved separately with ground truth.

Usage:
    python crop_fields.py input_image.jpg

This will open an interactive window where you can:
1. Draw rectangles around text fields
2. Enter the correct ground truth text
3. Save cropped images with labels
"""

import cv2
import argparse
import json
from pathlib import Path
from datetime import datetime
import uuid

# Global state for mouse callback
drawing = False
ix, iy = -1, -1
rectangles = []
current_rect = None


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, current_rect
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_rect = (ix, iy, x, y)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if abs(x - ix) > 10 and abs(y - iy) > 10:  # Minimum size
            rectangles.append((min(ix, x), min(iy, y), max(ix, x), max(iy, y)))
        current_rect = None


def crop_fields(image_path: str, output_dir: str = "training/data/raw"):
    """Interactive field cropping tool"""
    global rectangles, current_rect
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read {image_path}")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    labels_file = output_path.parent / "annotations" / "labels.txt"
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Resize for display if too large
    display_scale = 1.0
    h, w = img.shape[:2]
    if w > 1400:
        display_scale = 1400 / w
    
    window_name = "NID Field Cropper - Draw rectangles, press 's' to save, 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n" + "="*60)
    print("NID Card Field Cropper")
    print("="*60)
    print("Instructions:")
    print("  - Draw rectangle around each text field")
    print("  - Press 's' after each rectangle to save and label")
    print("  - Press 'u' to undo last rectangle")
    print("  - Press 'q' to quit")
    print("="*60 + "\n")
    
    field_types = ["name_bn", "name_en", "father", "mother", "dob", "id_no", "other"]
    saved_count = 0
    
    while True:
        display = img.copy()
        
        # Draw saved rectangles
        for rect in rectangles:
            cv2.rectangle(display, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        
        # Draw current rectangle being drawn
        if current_rect:
            cv2.rectangle(display, (current_rect[0], current_rect[1]), 
                         (current_rect[2], current_rect[3]), (0, 0, 255), 2)
        
        # Resize for display
        if display_scale != 1.0:
            display = cv2.resize(display, None, fx=display_scale, fy=display_scale)
        
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
            
        elif key == ord('u') and rectangles:
            rectangles.pop()
            print("Undone last rectangle")
            
        elif key == ord('s') and rectangles:
            rect = rectangles[-1]
            
            # Crop the region
            crop = img[rect[1]:rect[3], rect[0]:rect[2]]
            
            # Show crop preview
            cv2.imshow("Cropped Region", crop)
            cv2.waitKey(100)
            
            # Get field type
            print("\nSelect field type:")
            for i, ft in enumerate(field_types):
                print(f"  {i+1}. {ft}")
            
            try:
                field_idx = int(input("Enter number (1-7): ")) - 1
                field_type = field_types[field_idx] if 0 <= field_idx < len(field_types) else "other"
            except:
                field_type = "other"
            
            # Get ground truth
            print(f"\nEnter the CORRECT text for this {field_type} field:")
            ground_truth = input("Ground truth: ").strip()
            
            if ground_truth:
                # Generate filename
                uid = str(uuid.uuid4())[:8]
                filename = f"{field_type}_{uid}.jpg"
                filepath = output_path / filename
                
                # Save crop
                cv2.imwrite(str(filepath), crop)
                
                # Append to labels file
                with open(labels_file, "a", encoding="utf-8") as f:
                    f.write(f"{filename}\t{ground_truth}\n")
                
                saved_count += 1
                print(f"✓ Saved: {filename} -> '{ground_truth}'")
                print(f"  Total saved: {saved_count}")
            else:
                print("Skipped (no ground truth provided)")
            
            cv2.destroyWindow("Cropped Region")
    
    cv2.destroyAllWindows()
    print(f"\n{'='*60}")
    print(f"Done! Saved {saved_count} field crops to {output_path}")
    print(f"Labels saved to {labels_file}")
    print(f"{'='*60}\n")


def batch_label(raw_dir: str = "training/data/raw"):
    """Label existing cropped images"""
    raw_path = Path(raw_dir)
    labels_file = raw_path.parent / "annotations" / "labels.txt"
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing labels
    existing = set()
    if labels_file.exists():
        for line in labels_file.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                existing.add(line.split("\t")[0])
    
    images = list(raw_path.glob("*.jpg")) + list(raw_path.glob("*.png")) + list(raw_path.glob("*.jpeg"))
    unlabeled = [img for img in images if img.name not in existing]
    
    print(f"\nFound {len(unlabeled)} unlabeled images")
    
    for i, img_path in enumerate(unlabeled):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        cv2.imshow(f"Image {i+1}/{len(unlabeled)}: {img_path.name}", img)
        cv2.waitKey(100)
        
        print(f"\n[{i+1}/{len(unlabeled)}] {img_path.name}")
        ground_truth = input("Enter ground truth (or 'skip' / 'quit'): ").strip()
        
        cv2.destroyAllWindows()
        
        if ground_truth.lower() == 'quit':
            break
        elif ground_truth.lower() == 'skip' or not ground_truth:
            continue
        
        with open(labels_file, "a", encoding="utf-8") as f:
            f.write(f"{img_path.name}\t{ground_truth}\n")
        print(f"✓ Labeled: {img_path.name}")
    
    print("\nDone labeling!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NID Card Field Cropper")
    parser.add_argument("image", nargs="?", help="Input NID card image")
    parser.add_argument("--batch-label", action="store_true", 
                       help="Label existing crops in training/data/raw/")
    parser.add_argument("--output", default="training/data/raw",
                       help="Output directory for crops")
    
    args = parser.parse_args()
    
    if args.batch_label:
        batch_label(args.output)
    elif args.image:
        crop_fields(args.image, args.output)
    else:
        print("Usage:")
        print("  python crop_fields.py <nid_image.jpg>  - Interactive cropping")
        print("  python crop_fields.py --batch-label    - Label existing crops")
