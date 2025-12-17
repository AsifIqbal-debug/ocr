# NID OCR Fine-Tuning Quickstart Guide
# =====================================

## Overview
This guide walks you through fine-tuning an OCR model on Bangladesh National ID cards
to achieve high accuracy on Bangla text recognition.

## Step 1: Collect Training Data

You need 100-500+ cropped text field images with ground truth labels.
Focus on the problematic fields:
- নাম (Bangla name)
- Name (English name)  
- পিতা (Father's name)
- মাতা (Mother's name)
- Date of Birth
- ID Number

### Option A: Crop from existing NID images

```powershell
# Interactive cropping tool - draws rectangles on NID cards
python training/scripts/crop_fields.py mijja.jpg

# Process multiple cards
python training/scripts/crop_fields.py nur.jpeg
python training/scripts/crop_fields.py test.jpg
```

For each crop, you'll enter the CORRECT ground truth text:
- মির্জা ইমতিয়াজ আহমেদ  (not মির্জা ইমতিয়াজ্জগ আহমেদ)
- মোসাঃ বোকেয়া বেগম  (not মোঢাঃ বোকেয়া বেগম)

### Option B: Manual preparation

1. Create cropped images in `training/data/raw/`:
   - name_001.jpg  (crop of "মির্জা ইমতিয়াজ আহমেদ")
   - name_002.jpg  (crop of "MIRZA IMTIAZ AHMED")
   - father_001.jpg (crop of "মির্জা মোঃ জাকিব হোসাইন")
   - etc.

2. Create labels file `training/data/annotations/labels.txt`:
   ```
   name_001.jpg	মির্জা ইমতিয়াজ আহমেদ
   name_002.jpg	MIRZA IMTIAZ AHMED
   father_001.jpg	মির্জা মোঃ জাকিব হোসাইন
   mother_001.jpg	মোসাঃ বোকেয়া বেগম
   dob_001.jpg	18 May 1998
   id_001.jpg	9593515159
   ```

   Format: `filename<TAB>ground_truth` (TAB-separated)

## Step 2: Prepare Dataset

```powershell
# This will:
# 1. Build character dictionary from your labels
# 2. Split into train/val/test sets (80/10/10)
# 3. Create LMDB datasets for efficient training

python training/scripts/prepare_data.py prepare --labels training/data/annotations/labels.txt
```

## Step 3: Train the Model

```powershell
# Basic training (CPU - slow but works)
python training/scripts/train_easyocr.py --epochs 100 --batch-size 16

# With GPU (much faster)
python training/scripts/train_easyocr.py --epochs 100 --batch-size 64

# Customize parameters
python training/scripts/train_easyocr.py `
    --train-lmdb training/data/lmdb/train `
    --val-lmdb training/data/lmdb/val `
    --epochs 200 `
    --batch-size 32 `
    --lr 0.0005 `
    --save-dir training/checkpoints
```

Training typically takes:
- CPU: 1-4 hours for 100 epochs (depending on dataset size)
- GPU: 10-30 minutes for 100 epochs

## Step 4: Test the Model

```powershell
# Test on a cropped text field
python training/scripts/inference_custom.py test_crop.jpg --model training/checkpoints/best_model.pth
```

## Step 5: Integrate with Main OCR

Once you have a trained model, update `ocr.py` to use it.

### Quick Integration:

```python
# In ocr.py, add a new function:

def run_custom_ocr(image: np.ndarray, model_path: str) -> List[str]:
    """Run custom-trained NID OCR model"""
    from training.scripts.inference_custom import load_model, preprocess_image, predict
    
    model, idx_to_char = load_model(model_path)
    
    # For each detected text region, run custom model
    # ... implementation depends on your detection approach
```

## Tips for Best Results

### Data Collection
- **More data = better accuracy**: Aim for 300+ samples minimum
- **Include variations**: Different lighting, angles, image quality
- **Balance classes**: Similar number of samples for each field type
- **Use correct Unicode**: Ensure Bangla text uses proper NFC normalization

### Common Bangla OCR Mistakes to Train Against
| Wrong       | Correct     | Issue                    |
|-------------|-------------|--------------------------|
| মোঢাঃ       | মোসাঃ       | ঢ vs স confusion        |
| থেসাইন      | হোসাইন      | থ vs হ confusion        |
| ইমতিয়াজ্জগ | ইমতিয়াজ    | Extra conjuncts          |
| বোকেয়া     | রোকেয়া     | ব vs র confusion        |

Include multiple examples of these confusing patterns in your training data.

### Data Augmentation (built into training)
The training script automatically applies:
- Random brightness/contrast changes
- Slight rotation (-5° to +5°)
- Blur and noise
- Scale variations

## Troubleshooting

### "Not enough training samples"
You need at least 50 samples. Collect more NID images and crop more fields.

### "Validation accuracy stuck at 0%"
- Check your labels file format (must be TAB-separated)
- Ensure image files exist and are readable
- Try reducing learning rate: `--lr 0.0001`

### "Out of memory"
- Reduce batch size: `--batch-size 8`
- Reduce image width: `--img-width 200`

### Model overfitting (training acc high, validation acc low)
- Collect more diverse training data
- Add data augmentation
- Reduce model complexity or add dropout

## File Structure After Setup

```
training/
├── data/
│   ├── raw/                    # Cropped text field images
│   │   ├── name_001.jpg
│   │   ├── name_002.jpg
│   │   └── ...
│   ├── annotations/
│   │   ├── labels.txt          # All labels
│   │   ├── train_labels.txt    # Training split
│   │   ├── val_labels.txt      # Validation split
│   │   └── test_labels.txt     # Test split
│   └── lmdb/
│       ├── train/              # LMDB training dataset
│       └── val/                # LMDB validation dataset
├── configs/
│   └── nid_chars.txt           # Character dictionary
├── checkpoints/
│   ├── best_model.pth          # Best validation accuracy
│   └── checkpoint_epoch_*.pth  # Periodic checkpoints
└── scripts/
    ├── crop_fields.py          # Interactive cropping tool
    ├── prepare_data.py         # Dataset preparation
    ├── train_easyocr.py        # Training script
    └── inference_custom.py     # Inference with trained model
```
