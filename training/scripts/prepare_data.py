"""
EasyOCR Fine-Tuning for Bangla NID Cards
========================================
This script fine-tunes EasyOCR's recognition model on your NID card dataset.

EasyOCR uses a CRNN (CNN + RNN + CTC) architecture. We can fine-tune the
recognition model while keeping the detection model frozen.

Prerequisites:
    pip install easyocr torch torchvision pillow lmdb

Dataset Format:
    training/data/raw/          - Cropped text field images
    training/data/annotations/labels.txt - Tab-separated: filename<TAB>ground_truth
"""

import os
import sys
import argparse
import random
import lmdb
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import io


def create_lmdb_dataset(image_dir: Path, label_file: Path, output_dir: Path, 
                        check_valid: bool = True):
    """
    Convert image + label pairs to LMDB format for training.
    
    LMDB is efficient for random access during training.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read labels
    samples = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            filename, label = line.split("\t", 1)
            img_path = image_dir / filename
            if img_path.exists():
                samples.append((str(img_path), label))
            else:
                print(f"Warning: Image not found: {img_path}")
    
    print(f"Found {len(samples)} valid samples")
    
    if not samples:
        print("No samples found! Check your labels file and image directory.")
        return
    
    # Create LMDB
    env = lmdb.open(str(output_dir), map_size=1099511627776)  # 1TB max
    
    cache = {}
    cnt = 1
    
    for img_path, label in samples:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot read {img_path}")
            continue
        
        # Convert to bytes
        _, img_bytes = cv2.imencode('.jpg', img)
        img_bytes = img_bytes.tobytes()
        
        # Store with keys
        image_key = f'image-{cnt:09d}'.encode()
        label_key = f'label-{cnt:09d}'.encode()
        
        cache[image_key] = img_bytes
        cache[label_key] = label.encode('utf-8')
        
        if cnt % 100 == 0:
            with env.begin(write=True) as txn:
                for k, v in cache.items():
                    txn.put(k, v)
            cache = {}
            print(f"Processed {cnt}/{len(samples)}")
        
        cnt += 1
    
    # Write remaining
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)
        txn.put('num-samples'.encode(), str(cnt - 1).encode())
    
    env.close()
    print(f"Created LMDB dataset with {cnt - 1} samples at {output_dir}")


def build_character_list(label_file: Path, output_file: Path):
    """
    Extract unique characters from labels to build character dictionary.
    """
    chars = set()
    
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                _, label = line.strip().split("\t", 1)
                chars.update(label)
    
    # Sort characters: digits, English lowercase, English uppercase, Bangla, symbols
    def char_sort_key(c):
        if c.isdigit():
            return (0, c)
        elif c.isascii() and c.isalpha():
            return (1, c.lower(), c)
        elif '\u0980' <= c <= '\u09FF':  # Bangla Unicode range
            return (2, c)
        else:
            return (3, c)
    
    sorted_chars = sorted(chars, key=char_sort_key)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for c in sorted_chars:
            f.write(c + "\n")
    
    print(f"Built character list with {len(sorted_chars)} unique characters")
    print(f"Saved to {output_file}")
    
    # Show character groups
    bangla_chars = [c for c in sorted_chars if '\u0980' <= c <= '\u09FF']
    english_chars = [c for c in sorted_chars if c.isascii() and c.isalpha()]
    digits = [c for c in sorted_chars if c.isdigit()]
    symbols = [c for c in sorted_chars if not c.isalnum() and c != ' ']
    
    print(f"\nCharacter breakdown:")
    print(f"  Bangla: {len(bangla_chars)} - {''.join(bangla_chars[:20])}...")
    print(f"  English: {len(english_chars)} - {''.join(english_chars)}")
    print(f"  Digits: {len(digits)} - {''.join(digits)}")
    print(f"  Symbols: {len(symbols)} - {''.join(symbols)}")


def split_dataset(label_file: Path, output_dir: Path, 
                  train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Split labels into train/val/test sets.
    """
    with open(label_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and "\t" in l]
    
    random.shuffle(lines)
    
    n = len(lines)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, data in [("train", train_lines), ("val", val_lines), ("test", test_lines)]:
        out_file = output_dir / f"{name}_labels.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(data))
        print(f"  {name}: {len(data)} samples -> {out_file}")
    
    print(f"\nTotal: {n} samples split into train/val/test")


def main():
    parser = argparse.ArgumentParser(description="Prepare NID OCR training data")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Build character list
    char_parser = subparsers.add_parser("chars", help="Build character list from labels")
    char_parser.add_argument("--labels", type=Path, 
                            default=Path("training/data/annotations/labels.txt"))
    char_parser.add_argument("--output", type=Path,
                            default=Path("training/configs/nid_chars.txt"))
    
    # Split dataset
    split_parser = subparsers.add_parser("split", help="Split dataset into train/val/test")
    split_parser.add_argument("--labels", type=Path,
                             default=Path("training/data/annotations/labels.txt"))
    split_parser.add_argument("--output-dir", type=Path,
                             default=Path("training/data/annotations"))
    split_parser.add_argument("--train-ratio", type=float, default=0.8)
    split_parser.add_argument("--val-ratio", type=float, default=0.1)
    
    # Create LMDB
    lmdb_parser = subparsers.add_parser("lmdb", help="Create LMDB dataset")
    lmdb_parser.add_argument("--labels", type=Path, required=True,
                            help="Label file (train_labels.txt or val_labels.txt)")
    lmdb_parser.add_argument("--images", type=Path,
                            default=Path("training/data/raw"))
    lmdb_parser.add_argument("--output", type=Path, required=True,
                            help="Output LMDB directory")
    
    # All-in-one preparation
    prep_parser = subparsers.add_parser("prepare", help="Run all preparation steps")
    prep_parser.add_argument("--labels", type=Path,
                            default=Path("training/data/annotations/labels.txt"))
    prep_parser.add_argument("--images", type=Path,
                            default=Path("training/data/raw"))
    
    args = parser.parse_args()
    
    if args.command == "chars":
        build_character_list(args.labels, args.output)
        
    elif args.command == "split":
        split_dataset(args.labels, args.output_dir, args.train_ratio, args.val_ratio)
        
    elif args.command == "lmdb":
        create_lmdb_dataset(args.images, args.labels, args.output)
        
    elif args.command == "prepare":
        print("="*60)
        print("Step 1: Building character list...")
        print("="*60)
        build_character_list(args.labels, Path("training/configs/nid_chars.txt"))
        
        print("\n" + "="*60)
        print("Step 2: Splitting dataset...")
        print("="*60)
        split_dataset(args.labels, Path("training/data/annotations"))
        
        print("\n" + "="*60)
        print("Step 3: Creating LMDB datasets...")
        print("="*60)
        create_lmdb_dataset(
            args.images, 
            Path("training/data/annotations/train_labels.txt"),
            Path("training/data/lmdb/train")
        )
        create_lmdb_dataset(
            args.images,
            Path("training/data/annotations/val_labels.txt"),
            Path("training/data/lmdb/val")
        )
        
        print("\n" + "="*60)
        print("DONE! Ready for training.")
        print("="*60)
        print("\nNext step: Run training with:")
        print("  python training/scripts/train_easyocr.py")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
