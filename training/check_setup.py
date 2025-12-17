"""
Test the Surya fine-tuning setup
================================
"""
import sys
from pathlib import Path

def check_surya_finetuning():
    """Check if Surya fine-tuning requirements are met."""
    
    print("="*60)
    print("Surya OCR Fine-tuning Check")
    print("="*60)
    
    # 1. Check Surya installation
    print("\n1. Checking Surya installation...")
    try:
        import surya
        print(f"   ✓ Surya installed")
        
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        print(f"   ✓ Surya predictors available")
    except ImportError as e:
        print(f"   ✗ Surya not found: {e}")
        return False
    
    # 2. Check transformers
    print("\n2. Checking transformers...")
    try:
        from transformers import TrainingArguments, Trainer
        import transformers
        print(f"   ✓ Transformers {transformers.__version__}")
    except ImportError:
        print(f"   ✗ transformers not installed")
        print(f"   Run: pip install transformers")
        return False
    
    # 3. Check datasets
    print("\n3. Checking datasets library...")
    try:
        from datasets import Dataset, DatasetDict
        import datasets
        print(f"   ✓ Datasets {datasets.__version__}")
    except ImportError:
        print(f"   ✗ datasets not installed")
        print(f"   Run: pip install datasets")
        return False
    
    # 4. Check GPU
    print("\n4. Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print(f"   ⚠ No GPU found - training will be very slow!")
            print(f"     Consider using Google Colab or cloud GPU")
    except:
        print(f"   ⚠ Could not check GPU status")
    
    # 5. Check training data
    print("\n5. Checking training data...")
    data_dir = Path("training/data/raw")
    labels_file = Path("training/data/annotations/labels.txt")
    
    if data_dir.exists():
        images = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
        print(f"   ✓ Found {len(images)} training images in {data_dir}")
    else:
        print(f"   ✗ Training data directory not found: {data_dir}")
        print(f"   Run: python training/scripts/auto_crop.py <nid_image.jpg>")
        return False
    
    if labels_file.exists():
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = [l.strip() for l in f if '\t' in l]
        print(f"   ✓ Found {len(labels)} labels in {labels_file}")
    else:
        print(f"   ✗ Labels file not found: {labels_file}")
        print(f"   Run: python training/scripts/create_labels.py")
        return False
    
    # 6. Check Surya fine-tune script
    print("\n6. Checking Surya fine-tune script...")
    try:
        import surya
        surya_path = Path(surya.__file__).parent
        finetune_script = surya_path / "scripts" / "finetune_ocr.py"
        
        if finetune_script.exists():
            print(f"   ✓ Official script found: {finetune_script}")
        else:
            # Check alternate locations
            possible = [
                surya_path.parent / "scripts" / "finetune_ocr.py",
                Path(sys.prefix) / "Scripts" / "finetune_ocr.py",
            ]
            found = False
            for p in possible:
                if p.exists():
                    print(f"   ✓ Script found at: {p}")
                    found = True
                    break
            if not found:
                print(f"   ⚠ Official script not found")
                print(f"     Will use custom training loop")
    except Exception as e:
        print(f"   ⚠ Could not locate script: {e}")
    
    print("\n" + "="*60)
    print("Setup check complete!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Review/correct labels:")
    print("   python training/scripts/create_labels.py")
    print("")
    print("2. Prepare dataset and train:")
    print("   python training/train_surya_bangla.py --epochs 50")
    print("")
    print("For GPU training (recommended):")
    print("   - Use Google Colab with T4/V100 GPU")
    print("   - Upload training/data folder to Colab")
    print("   - Run training script there")
    
    return True


if __name__ == "__main__":
    check_surya_finetuning()
