"""
Prepare Dataset for Surya Fine-tuning
=====================================
This script prepares your NID images and labels for fine-tuning Surya OCR.

Usage:
    python prepare_dataset.py --images-dir data/images --labels-dir data/labels
"""

import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import random


def create_line_crops(image_path: str, label_path: str, output_dir: Path):
    """
    Create individual line crops from an NID image with ground truth labels.
    
    For Surya fine-tuning, we need:
    - Cropped text line images
    - Corresponding ground truth text
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Cannot read {image_path}")
        return []
    
    # Read labels (one line per text region)
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    # Use EasyOCR to detect text regions
    import easyocr
    reader = easyocr.Reader(['bn', 'en'], gpu=False)
    detections = reader.readtext(img, detail=1)
    
    samples = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Match detections to labels (by position/order)
    for i, (bbox, detected_text, conf) in enumerate(detections):
        if i >= len(lines):
            break
        
        # Get bounding box
        pts = np.array(bbox, dtype=np.int32)
        x_min = max(0, pts[:, 0].min() - 5)
        y_min = max(0, pts[:, 1].min() - 5)
        x_max = min(img.shape[1], pts[:, 0].max() + 5)
        y_max = min(img.shape[0], pts[:, 1].max() + 5)
        
        # Crop region
        crop = img[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue
        
        # Save crop
        crop_name = f"{Path(image_path).stem}_{i:03d}.jpg"
        crop_path = output_dir / crop_name
        cv2.imwrite(str(crop_path), crop)
        
        samples.append({
            "image": str(crop_path),
            "text": lines[i],
            "detected": detected_text,
        })
    
    return samples


def prepare_manual_crops(crops_dir: Path, labels_file: Path, output_dir: Path):
    """
    Prepare dataset from manually cropped images with labels.
    
    Labels file format (tab-separated):
    image_name.jpg\tground_truth_text
    """
    samples = []
    
    if not labels_file.exists():
        print(f"Labels file not found: {labels_file}")
        return samples
    
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            
            img_name, text = parts
            img_path = crops_dir / img_name
            
            if img_path.exists():
                # Copy to output with augmentation
                img = cv2.imread(str(img_path))
                if img is not None:
                    out_path = output_dir / img_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), img)
                    
                    samples.append({
                        "image": str(out_path),
                        "text": text,
                    })
    
    return samples


def augment_image(img: np.ndarray) -> list:
    """
    Create augmented versions of an image for training.
    """
    augmented = []
    
    # Original
    augmented.append(("orig", img))
    
    # Brightness variations
    for factor in [0.8, 1.2]:
        bright = cv2.convertScaleAbs(img, alpha=factor, beta=0)
        augmented.append((f"bright_{factor}", bright))
    
    # Slight rotation
    h, w = img.shape[:2]
    for angle in [-2, 2]:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
        augmented.append((f"rot_{angle}", rotated))
    
    # Blur
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    augmented.append(("blur", blurred))
    
    return augmented


def create_surya_dataset(samples: list, output_dir: Path, augment: bool = True):
    """
    Create dataset in Surya fine-tuning format.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_samples = []
    val_samples = []
    
    # Shuffle and split 90/10
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.9)
    
    for i, sample in enumerate(samples):
        img = cv2.imread(sample["image"])
        if img is None:
            continue
        
        is_train = i < split_idx
        subset = "train" if is_train else "val"
        subset_dir = output_dir / subset / "images"
        subset_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(sample["image"]).stem
        
        if augment and is_train:
            # Create augmented versions
            for aug_name, aug_img in augment_image(img):
                img_name = f"{base_name}_{aug_name}.jpg"
                img_path = subset_dir / img_name
                cv2.imwrite(str(img_path), aug_img)
                
                entry = {
                    "image": str(img_path.relative_to(output_dir / subset)),
                    "text": sample["text"],
                }
                train_samples.append(entry)
        else:
            img_name = f"{base_name}.jpg"
            img_path = subset_dir / img_name
            cv2.imwrite(str(img_path), img)
            
            entry = {
                "image": str(img_path.relative_to(output_dir / subset)),
                "text": sample["text"],
            }
            if is_train:
                train_samples.append(entry)
            else:
                val_samples.append(entry)
    
    # Save manifests
    with open(output_dir / "train" / "manifest.json", 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "val" / "manifest.json", 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    
    print(f"Created dataset:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    
    return train_samples, val_samples


def main():
    parser = argparse.ArgumentParser(description="Prepare Surya fine-tuning dataset")
    parser.add_argument("--crops-dir", type=Path, default=Path("training/data/raw"),
                       help="Directory with cropped text images")
    parser.add_argument("--labels-file", type=Path, 
                       default=Path("training/data/annotations/labels.txt"),
                       help="Labels file (tab-separated: filename\\ttext)")
    parser.add_argument("--output-dir", type=Path, 
                       default=Path("training/finetune_surya/dataset"),
                       help="Output directory for processed dataset")
    parser.add_argument("--no-augment", action="store_true",
                       help="Disable data augmentation")
    
    args = parser.parse_args()
    
    print("Preparing dataset...")
    print(f"  Crops: {args.crops_dir}")
    print(f"  Labels: {args.labels_file}")
    
    # Load samples from manual crops
    samples = prepare_manual_crops(
        args.crops_dir, 
        args.labels_file,
        args.output_dir / "processed"
    )
    
    if not samples:
        print("\nNo samples found! Please create training data first:")
        print("1. Run: python training/scripts/auto_crop.py <nid_image.jpg>")
        print("2. Correct the labels in training/data/annotations/labels.txt")
        print("3. Run this script again")
        return
    
    print(f"\nLoaded {len(samples)} samples")
    
    # Create Surya-format dataset
    create_surya_dataset(
        samples, 
        args.output_dir,
        augment=not args.no_augment
    )


if __name__ == "__main__":
    main()
