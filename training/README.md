# Surya OCR Fine-tuning for Bangla NID Cards
## Complete Training Pipeline

This directory contains everything needed to fine-tune Surya OCR for accurate Bangla text recognition on Bangladesh National ID cards.

## Quick Start

### Step 1: Extract Training Data from NID Images
```powershell
# Run from project root
cd G:\BRAC\OCR_Bangla

# Process one NID image
python training/scripts/auto_crop.py mijja.jpg

# Process multiple images
python training/scripts/auto_crop.py mijja.jpg nur.jpeg baje.jpg madam.jpg
```

### Step 2: Create Ground Truth Labels
```powershell
# Interactive labeling (opens each crop and asks for correct text)
python training/scripts/create_labels.py

# Or import from a prepared file
python training/scripts/create_labels.py --import-file my_labels.txt
```

### Step 3: Train the Model
```powershell
# Prepare dataset and start training
python training/train_surya_bangla.py --epochs 50

# Or just prepare dataset (no training)
python training/train_surya_bangla.py --prepare-only
```

## Directory Structure

```
training/
├── data/
│   ├── raw/                  # Cropped text regions from NID images
│   ├── annotations/          # Ground truth labels
│   │   └── labels.txt        # Tab-separated: filename<TAB>text
│   └── processed/            # Processed training data
│
├── models/
│   └── surya_bangla/         # Fine-tuned model checkpoints
│       ├── dataset/          # HuggingFace format dataset
│       └── checkpoints/      # Model weights
│
├── scripts/
│   ├── auto_crop.py          # Extract text regions from NID images
│   └── create_labels.py      # Interactive labeling tool
│
└── train_surya_bangla.py     # Main training script
```

## Label File Format

The `labels.txt` file uses tab-separated values:
```
crop_001.jpg	নামঃ মোঃ আরিফুল ইসলাম
crop_002.jpg	পিতাঃ মোঃ করিম উদ্দিন
crop_003.jpg	মাতাঃ মোসাঃ ফাতেমা বেগম
crop_004.jpg	জন্ম তারিখঃ ০১ জানু ১৯৯০
crop_005.jpg	ID No: 1234567890123
```

## Common Bangla NID Fields

| Field | Bangla | Transliteration |
|-------|--------|-----------------|
| Name | নামঃ | Naam |
| Father | পিতাঃ | Pita |
| Mother | মাতাঃ | Mata |
| Husband | স্বামীঃ | Shami |
| Date of Birth | জন্ম তারিখঃ | Jonmo Tarikh |
| Address | ঠিকানাঃ | Thikana |
| National ID | জাতীয় পরিচয় পত্র | Jatiyo Porichoy Potro |

## Bangla Numbers

| English | Bangla |
|---------|--------|
| 0 | ০ |
| 1 | ১ |
| 2 | ২ |
| 3 | ৩ |
| 4 | ৪ |
| 5 | ৫ |
| 6 | ৬ |
| 7 | ৭ |
| 8 | ৮ |
| 9 | ৯ |

## Tips for Better Training

1. **Collect Diverse Data**: Include NID images from different:
   - Lighting conditions
   - Camera angles
   - Card conditions (new, worn, damaged)
   - Print variations

2. **Label Accurately**: Double-check Bangla text for:
   - Correct conjuncts (যুক্তাক্ষর)
   - Proper vowel marks (কার/ফলা)
   - Correct visarga (ঃ) vs colon (:)

3. **Balance Dataset**: Ensure good coverage of:
   - All common Bangla characters
   - Numbers (both Bangla and English)
   - Field labels (নামঃ, পিতাঃ, etc.)
   - Personal names (wide variety)

4. **Augment Data**: The training script can augment data with:
   - Rotation (±5°)
   - Brightness variations
   - Contrast adjustments
   - Slight blur

## Hardware Requirements

| GPU | Batch Size | Training Time (50 epochs, 1000 samples) |
|-----|------------|----------------------------------------|
| RTX 4090 | 16 | ~2 hours |
| RTX 3080 | 8 | ~4 hours |
| RTX 3060 | 4 | ~8 hours |
| CPU only | 1 | Not recommended |

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size:
```powershell
python training/train_surya_bangla.py --batch-size 4
```

### Slow Training
Enable gradient checkpointing:
```powershell
python training/train_surya_bangla.py --gradient-checkpointing
```

### Poor Results
- Add more diverse training data
- Increase epochs
- Check label accuracy
- Try lower learning rate

## Using Fine-tuned Model

After training, use the fine-tuned model:

```python
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor
from PIL import Image

# Load fine-tuned model
foundation = FoundationPredictor()
rec_predictor = RecognitionPredictor(
    foundation,
    model_path="training/models/surya_bangla/checkpoints/final"
)

# Run OCR
image = Image.open("nid_image.jpg")
results = rec_predictor([image], ["bn", "en"])

for line in results[0].text_lines:
    print(line.text)
```

## Official Surya Fine-tuning

For advanced fine-tuning, use the official Surya script:

```powershell
# Upload dataset to HuggingFace first
huggingface-cli login
huggingface-cli upload your-username/bangla-nid-ocr training/models/surya_bangla/dataset

# Run official fine-tuning
python -m surya.scripts.finetune_ocr \
    --output_dir training/models/surya_bangla/checkpoints \
    --dataset_name your-username/bangla-nid-ocr \
    --per_device_train_batch_size 16 \
    --gradient_checkpointing true \
    --max_sequence_length 512 \
    --num_train_epochs 50
```

## Contact

For questions about Surya fine-tuning internals, contact: hi@datalab.to
