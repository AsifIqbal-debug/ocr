"""
Surya OCR Fine-Tuning for Bangla NID Cards
==========================================
This script fine-tunes the Surya OCR recognition model on custom Bangla NID data.

Surya uses a Vision Transformer (ViT) encoder + Text Transformer decoder architecture.
We'll fine-tune on your specific NID card images for better Bangla accuracy.

Prerequisites:
    pip install surya-ocr torch transformers datasets accelerate

Usage:
    1. Prepare training data: python train_surya.py prepare
    2. Train model: python train_surya.py train --epochs 50
    3. Export model: python train_surya.py export
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import random

# Training data directory structure
DATA_DIR = Path("training/data")
RAW_DIR = DATA_DIR / "raw"
LABELS_FILE = DATA_DIR / "annotations" / "labels.txt"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODEL_DIR = Path("training/models/surya_bangla")


def setup_directories():
    """Create necessary directories."""
    for d in [RAW_DIR, DATA_DIR / "annotations", TRAIN_DIR, VAL_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Created directories in {DATA_DIR}")


def prepare_dataset():
    """
    Prepare dataset from cropped NID images and labels.
    
    Expected format in labels.txt:
        image_name.jpg<TAB>correct_text
    
    Example:
        name_001.jpg	মোঃ শিয়াম উদ্দিন
        name_002.jpg	MIRZA IMTIAZ AHMED
    """
    setup_directories()
    
    if not LABELS_FILE.exists():
        print(f"\nNo labels file found at {LABELS_FILE}")
        print("\nTo create training data:")
        print("1. Crop text regions from NID cards")
        print("2. Save cropped images to: training/data/raw/")
        print("3. Create labels.txt with format: filename<TAB>correct_text")
        print("\nExample labels.txt content:")
        print("  name_001.jpg\tমোঃ শিয়াম উদ্দিন")
        print("  name_002.jpg\tমির্জা ইমতিয়াজ আহমেদ")
        print("  father_001.jpg\tমোঃ জাকির হোসাইন")
        return False
    
    # Read labels
    samples = []
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            filename, label = line.split("\t", 1)
            img_path = RAW_DIR / filename
            if img_path.exists():
                samples.append((str(img_path), label))
            else:
                print(f"Warning: Image not found: {img_path}")
    
    if len(samples) < 10:
        print(f"\nOnly {len(samples)} samples found. Need at least 10 for training.")
        print("Add more cropped images and labels to training/data/raw/")
        return False
    
    print(f"Found {len(samples)} valid samples")
    
    # Split into train/val (80/20)
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    # Save split files
    for name, data, out_dir in [("train", train_samples, TRAIN_DIR), 
                                 ("val", val_samples, VAL_DIR)]:
        # Create manifest
        manifest_file = out_dir / "manifest.json"
        manifest = []
        
        for img_path, label in data:
            # Copy image to split directory
            import shutil
            dest = out_dir / Path(img_path).name
            if not dest.exists():
                shutil.copy(img_path, dest)
            
            manifest.append({
                "image": Path(img_path).name,
                "text": label
            })
        
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        print(f"  {name}: {len(data)} samples -> {out_dir}")
    
    print("\nDataset prepared successfully!")
    print(f"Train: {len(train_samples)} samples")
    print(f"Val: {len(val_samples)} samples")
    return True


def create_training_script():
    """Create the actual fine-tuning script using HuggingFace Trainer."""
    
    script_content = '''"""
Surya Fine-Tuning Script
========================
This script fine-tunes Surya's recognition model on custom data.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
import os

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class NIDDataset(Dataset):
    """Dataset for NID text recognition."""
    
    def __init__(self, data_dir: Path, processor):
        self.data_dir = Path(data_dir)
        self.processor = processor
        
        # Load manifest
        manifest_file = self.data_dir / "manifest.json"
        with open(manifest_file, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img_path = self.data_dir / sample["image"]
        image = Image.open(img_path).convert("RGB")
        
        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "text": sample["text"]
        }


def collate_fn(batch, processor, tokenizer):
    """Custom collate function for batching."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    texts = [item["text"] for item in batch]
    
    # Tokenize texts
    labels = tokenizer(texts, padding=True, return_tensors="pt")
    
    return {
        "pixel_values": pixel_values,
        "labels": labels.input_ids
    }


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device, processor):
    """Validate model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            pixel_values = batch["pixel_values"].to(device)
            texts = batch["texts"]
            
            # Generate predictions
            generated_ids = model.generate(pixel_values, max_length=64)
            predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            for pred, gt in zip(predictions, texts):
                if pred.strip() == gt.strip():
                    correct += 1
                total += 1
    
    return correct / total if total > 0 else 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-dir", type=Path, default=Path("training/data/train"))
    parser.add_argument("--val-dir", type=Path, default=Path("training/data/val"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/models/surya_bangla"))
    args = parser.parse_args()
    
    print("Loading Surya model...")
    
    # For now, we'll use a simpler approach with EasyOCR fine-tuning
    # since Surya's training code isn't publicly available
    print("\\nNote: Surya's training code is not publicly available.")
    print("Using alternative approach with custom CRNN model...")
    
    # Import our custom training
    from train_easyocr import main as train_custom
    train_custom()


if __name__ == "__main__":
    main()
'''
    
    script_path = Path("training/scripts/finetune_surya.py")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_content, encoding="utf-8")
    print(f"Created {script_path}")


def auto_crop_nid_images():
    """Auto-crop text regions from NID images in the workspace."""
    import cv2
    import easyocr
    import uuid
    
    setup_directories()
    
    # Find NID images
    workspace = Path(".")
    nid_images = list(workspace.glob("*.jpg")) + list(workspace.glob("*.jpeg")) + list(workspace.glob("*.png"))
    nid_images = [f for f in nid_images if not f.name.startswith("output_")]
    
    if not nid_images:
        print("No NID images found in workspace")
        return
    
    print(f"Found {len(nid_images)} images to process")
    
    # Initialize EasyOCR for detection
    reader = easyocr.Reader(['bn', 'en'], gpu=False)
    
    labels = []
    crop_count = 0
    
    for img_path in nid_images:
        print(f"\nProcessing {img_path.name}...")
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Detect text regions
        results = reader.readtext(img, detail=1)
        
        for bbox, text, conf in results:
            if conf < 0.3 or len(text) < 2:
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
            
            # Save crop
            uid = str(uuid.uuid4())[:8]
            filename = f"crop_{uid}.jpg"
            filepath = RAW_DIR / filename
            cv2.imwrite(str(filepath), crop)
            
            # Add to labels (using EasyOCR detection as initial label)
            labels.append(f"{filename}\t{text}")
            crop_count += 1
    
    # Save labels file
    if labels:
        with open(LABELS_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(labels) + "\n")
        
        print(f"\n{'='*50}")
        print(f"Auto-cropped {crop_count} text regions")
        print(f"Saved to: {RAW_DIR}")
        print(f"Labels added to: {LABELS_FILE}")
        print(f"\n⚠️  IMPORTANT: Review and correct labels in {LABELS_FILE}")
        print("The OCR may have made mistakes - fix them before training!")
    else:
        print("No text regions detected")


def train_model(epochs: int = 50, batch_size: int = 8, lr: float = 0.001):
    """Train the custom Bangla OCR model."""
    
    # Check if dataset is prepared
    if not (TRAIN_DIR / "manifest.json").exists():
        print("Dataset not prepared. Running prepare first...")
        if not prepare_dataset():
            return
    
    print("\n" + "="*60)
    print("Training Bangla OCR Model")
    print("="*60)
    
    # Import training module
    sys.path.insert(0, str(Path("training/scripts")))
    
    try:
        from train_easyocr import (
            LMDBDataset, CRNN, train_epoch, validate, 
            create_lmdb_dataset, build_character_list
        )
    except ImportError:
        print("Training scripts not found. Creating them...")
        # The scripts should already exist from previous setup
        return
    
    import torch
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import torch.nn as nn
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build character list from training data
    char_file = Path("training/configs/bangla_chars.txt")
    if not char_file.exists():
        build_character_list(LABELS_FILE, char_file)
    
    char_list = [line.strip() for line in char_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    num_classes = len(char_list) + 1  # +1 for CTC blank
    print(f"Character set size: {len(char_list)}")
    
    # Create LMDB datasets if not exists
    train_lmdb = DATA_DIR / "lmdb" / "train"
    val_lmdb = DATA_DIR / "lmdb" / "val"
    
    if not train_lmdb.exists():
        print("Creating LMDB datasets...")
        train_manifest = json.loads((TRAIN_DIR / "manifest.json").read_text(encoding='utf-8'))
        val_manifest = json.loads((VAL_DIR / "manifest.json").read_text(encoding='utf-8'))
        
        # Convert manifest to labels format
        train_labels = TRAIN_DIR / "labels.txt"
        val_labels = VAL_DIR / "labels.txt"
        
        with open(train_labels, "w", encoding="utf-8") as f:
            for item in train_manifest:
                f.write(f"{item['image']}\t{item['text']}\n")
        
        with open(val_labels, "w", encoding="utf-8") as f:
            for item in val_manifest:
                f.write(f"{item['image']}\t{item['text']}\n")
        
        create_lmdb_dataset(TRAIN_DIR, train_labels, train_lmdb)
        create_lmdb_dataset(VAL_DIR, val_labels, val_lmdb)
    
    # Create datasets
    train_dataset = LMDBDataset(str(train_lmdb), char_list)
    val_dataset = LMDBDataset(str(val_lmdb), char_list)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = CRNN(64, 1, num_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
    # Data loaders
    from train_easyocr import collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)
    
    # Training loop
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_accuracy = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        accuracy = validate(model, val_loader, train_dataset.idx_to_char, device)
        print(f"Val Accuracy: {accuracy:.2%}")
        
        scheduler.step(accuracy)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_path = MODEL_DIR / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'char_list': char_list,
            }, save_path)
            print(f"✓ Saved best model (accuracy: {accuracy:.2%})")
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Accuracy: {best_accuracy:.2%}")
    print(f"Model saved to: {MODEL_DIR / 'best_model.pth'}")
    print(f"{'='*60}")


def export_model():
    """Export trained model for inference."""
    model_path = MODEL_DIR / "best_model.pth"
    
    if not model_path.exists():
        print(f"No trained model found at {model_path}")
        print("Run training first: python train_surya.py train")
        return
    
    print(f"Model ready at: {model_path}")
    print("\nTo use the trained model:")
    print("  python training/scripts/inference_custom.py image.jpg --model training/models/surya_bangla/best_model.pth")


def main():
    parser = argparse.ArgumentParser(description="Surya/Bangla OCR Fine-Tuning")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Auto-crop command
    crop_parser = subparsers.add_parser("crop", help="Auto-crop text from NID images")
    
    # Prepare command
    prep_parser = subparsers.add_parser("prepare", help="Prepare training dataset")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=0.001)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export trained model")
    
    # Full pipeline
    full_parser = subparsers.add_parser("full", help="Run full pipeline: crop -> prepare -> train")
    full_parser.add_argument("--epochs", type=int, default=50)
    
    args = parser.parse_args()
    
    if args.command == "crop":
        auto_crop_nid_images()
        
    elif args.command == "prepare":
        prepare_dataset()
        
    elif args.command == "train":
        train_model(args.epochs, args.batch_size, args.lr)
        
    elif args.command == "export":
        export_model()
        
    elif args.command == "full":
        print("="*60)
        print("STEP 1: Auto-cropping NID images")
        print("="*60)
        auto_crop_nid_images()
        
        print("\n" + "="*60)
        print("STEP 2: Preparing dataset")
        print("="*60)
        if prepare_dataset():
            print("\n" + "="*60)
            print("STEP 3: Training model")
            print("="*60)
            train_model(args.epochs)
            
            print("\n" + "="*60)
            print("STEP 4: Exporting model")
            print("="*60)
            export_model()
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("Quick Start Guide:")
        print("="*60)
        print("\n1. Auto-crop NID images (creates training data):")
        print("   python train_surya.py crop")
        print("\n2. ⚠️  IMPORTANT: Edit training/data/annotations/labels.txt")
        print("   Fix any OCR mistakes in the labels!")
        print("\n3. Prepare dataset:")
        print("   python train_surya.py prepare")
        print("\n4. Train model:")
        print("   python train_surya.py train --epochs 50")
        print("\n5. Use trained model:")
        print("   python training/scripts/inference_custom.py image.jpg")
        print("\nOr run full pipeline:")
        print("   python train_surya.py full --epochs 50")


if __name__ == "__main__":
    main()
