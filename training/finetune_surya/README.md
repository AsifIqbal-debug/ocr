# Fine-tuning Surya OCR for Bangla NID Cards

## Overview
This directory contains scripts to fine-tune Surya OCR model specifically for 
Bangla (Bengali) text recognition on Bangladesh National ID cards.

## The Problem
Surya OCR sometimes misrecognizes Bangla script as Hindi (Devanagari) because
both scripts look similar. Fine-tuning on Bangla-specific data solves this.

## Requirements
- GPU with at least 8GB VRAM (recommended: 16GB+)
- Python 3.10+
- PyTorch 2.0+

## Dataset Preparation

### Step 1: Collect NID Images
Place your NID card images in `data/images/`

### Step 2: Create Ground Truth Labels
For each image, create a text file with the correct Bangla text.

Example: `data/labels/image001.txt`
```
নামঃ মোঃ শিয়াম উদ্দিন
Name: MD SHIYAM UDDIN
পিতাঃ মোঃ আব্দুল করিম
মাতাঃ রাবেয়া বেগম
Date of Birth: 15 Jan 1990
NID No: 1234567890
```

### Step 3: Run Training
```bash
python train.py --data-dir data --epochs 50 --batch-size 4
```

## Files
- `prepare_dataset.py` - Prepares training data from images + labels
- `train.py` - Fine-tuning script
- `evaluate.py` - Evaluate model accuracy
