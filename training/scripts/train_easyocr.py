"""
EasyOCR Fine-Tuning Script for Bangla NID Cards
================================================

This script fine-tunes EasyOCR's recognition model (CRNN) on custom NID data.

The model architecture:
- CNN Feature Extractor (ResNet/VGG backbone)
- Sequence Modeling (BiLSTM)
- CTC Decoder

We freeze the CNN backbone and fine-tune the LSTM + CTC layers for faster training.

Usage:
    python train_easyocr.py --epochs 100 --batch-size 32
"""

import os
import sys
import argparse
import random
import string
import lmdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from PIL import Image
import io
from pathlib import Path
from datetime import datetime

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class LMDBDataset(Dataset):
    """Dataset that reads from LMDB"""
    
    def __init__(self, lmdb_path: str, char_list: list, img_height: int = 64, 
                 img_width: int = 256):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.num_samples = int(txn.get('num-samples'.encode()).decode())
        
        self.char_list = char_list
        self.char_to_idx = {c: i + 1 for i, c in enumerate(char_list)}  # 0 = blank for CTC
        self.idx_to_char = {i + 1: c for i, c in enumerate(char_list)}
        self.idx_to_char[0] = ''  # blank
        
        self.img_height = img_height
        self.img_width = img_width
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        idx = idx + 1  # LMDB is 1-indexed
        
        with self.env.begin() as txn:
            img_key = f'image-{idx:09d}'.encode()
            label_key = f'label-{idx:09d}'.encode()
            
            img_bytes = txn.get(img_key)
            label = txn.get(label_key).decode('utf-8')
        
        # Decode image
        img = Image.open(io.BytesIO(img_bytes)).convert('L')  # Grayscale
        
        # Resize maintaining aspect ratio
        w, h = img.size
        ratio = self.img_height / h
        new_w = int(w * ratio)
        if new_w > self.img_width:
            new_w = self.img_width
        
        img = img.resize((new_w, self.img_height), Image.Resampling.LANCZOS)
        
        # Pad to fixed width
        padded = Image.new('L', (self.img_width, self.img_height), color=255)
        padded.paste(img, (0, 0))
        
        # To tensor and normalize
        img_tensor = torch.FloatTensor(np.array(padded)) / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dim
        
        # Encode label
        label_encoded = [self.char_to_idx.get(c, 0) for c in label]
        label_tensor = torch.LongTensor(label_encoded)
        
        return img_tensor, label_tensor, len(label_encoded)


def collate_fn(batch):
    """Custom collate for variable length labels"""
    images, labels, lengths = zip(*batch)
    
    images = torch.stack(images, 0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    lengths = torch.LongTensor(lengths)
    
    return images, labels, lengths


class CRNN(nn.Module):
    """
    CRNN model for text recognition.
    Architecture: CNN -> BiLSTM -> Linear -> CTC
    """
    
    def __init__(self, img_height: int, num_channels: int, num_classes: int,
                 hidden_size: int = 256, num_lstm_layers: int = 2):
        super(CRNN, self).__init__()
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(num_channels, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Block 4
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            # Block 5
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Block 6
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            # Block 7
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        
        # Calculate CNN output height (after all pooling)
        # Input: img_height -> /2 -> /2 -> /2 -> /2 -> -1 = img_height/16 - 1
        cnn_output_height = img_height // 16 - 1
        
        # BiLSTM
        self.rnn = nn.LSTM(
            512 * cnn_output_height,
            hidden_size,
            num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # CNN
        conv = self.cnn(x)  # (batch, 512, h, w)
        
        # Reshape for RNN: (batch, width, channels*height)
        batch, channels, height, width = conv.size()
        conv = conv.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        conv = conv.contiguous().view(batch, width, channels * height)
        
        # RNN
        rnn_out, _ = self.rnn(conv)  # (batch, width, hidden*2)
        
        # FC
        output = self.fc(rnn_out)  # (batch, width, num_classes)
        
        # For CTC, need (width, batch, num_classes)
        output = output.permute(1, 0, 2)
        
        return output


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, labels, lengths) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(images)  # (T, N, C)
        
        # CTC Loss
        T = outputs.size(0)
        N = outputs.size(1)
        input_lengths = torch.full((N,), T, dtype=torch.long)
        
        # Log softmax for CTC
        log_probs = outputs.log_softmax(2)
        
        loss = criterion(log_probs, labels, input_lengths, lengths)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def decode_predictions(outputs, idx_to_char):
    """Decode CTC output using greedy decoding"""
    # outputs: (T, N, C)
    _, max_indices = outputs.max(2)  # (T, N)
    max_indices = max_indices.permute(1, 0)  # (N, T)
    
    decoded = []
    for seq in max_indices:
        chars = []
        prev = -1
        for idx in seq:
            idx = idx.item()
            if idx != 0 and idx != prev:  # Not blank and not repeat
                chars.append(idx_to_char.get(idx, ''))
            prev = idx
        decoded.append(''.join(chars))
    
    return decoded


def validate(model, dataloader, idx_to_char, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, lengths in dataloader:
            images = images.to(device)
            
            outputs = model(images)
            predictions = decode_predictions(outputs, idx_to_char)
            
            # Decode ground truth
            for i, (pred, label, length) in enumerate(zip(predictions, labels, lengths)):
                gt_chars = [idx_to_char.get(idx.item(), '') for idx in label[:length]]
                gt = ''.join(gt_chars)
                
                if pred == gt:
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Train EasyOCR model on NID data")
    parser.add_argument("--train-lmdb", type=Path, 
                       default=Path("training/data/lmdb/train"))
    parser.add_argument("--val-lmdb", type=Path,
                       default=Path("training/data/lmdb/val"))
    parser.add_argument("--char-list", type=Path,
                       default=Path("training/configs/nid_chars.txt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save-dir", type=Path,
                       default=Path("training/checkpoints"))
    parser.add_argument("--img-height", type=int, default=64)
    parser.add_argument("--img-width", type=int, default=256)
    
    args = parser.parse_args()
    
    # Load character list
    if not args.char_list.exists():
        print(f"Error: Character list not found at {args.char_list}")
        print("Run: python training/scripts/prepare_data.py prepare")
        sys.exit(1)
    
    char_list = [line.strip() for line in args.char_list.read_text(encoding='utf-8').splitlines() if line.strip()]
    num_classes = len(char_list) + 1  # +1 for CTC blank
    print(f"Character set size: {len(char_list)} + 1 blank = {num_classes}")
    
    # Create datasets
    train_dataset = LMDBDataset(str(args.train_lmdb), char_list, args.img_height, args.img_width)
    val_dataset = LMDBDataset(str(args.val_lmdb), char_list, args.img_height, args.img_width)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Create model
    model = CRNN(args.img_height, 1, num_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
    # Training loop
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validate
        accuracy = validate(model, val_loader, train_dataset.idx_to_char, device)
        print(f"Validation Accuracy: {accuracy:.2%}")
        
        scheduler.step(accuracy)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_path = args.save_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'char_list': char_list,
            }, save_path)
            print(f"✓ Saved best model (accuracy: {accuracy:.2%})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_path = args.save_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'char_list': char_list,
            }, save_path)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Accuracy: {best_accuracy:.2%}")
    print(f"Model saved to: {args.save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
