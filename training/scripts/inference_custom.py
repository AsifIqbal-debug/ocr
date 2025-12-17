"""
Use custom-trained NID OCR model for inference
==============================================

After training, use this script to run OCR with your fine-tuned model.

Usage:
    python inference_custom.py image.jpg --model training/checkpoints/best_model.pth
"""

import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# Import the CRNN model from training script
sys.path.insert(0, str(Path(__file__).parent))
from train_easyocr import CRNN, decode_predictions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_image(img_path: str, img_height: int = 64, img_width: int = 256):
    """Preprocess image for the model"""
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize maintaining aspect ratio
    h, w = gray.shape
    ratio = img_height / h
    new_w = int(w * ratio)
    if new_w > img_width:
        new_w = img_width
    
    gray = cv2.resize(gray, (new_w, img_height), interpolation=cv2.INTER_CUBIC)
    
    # Pad to fixed width
    padded = np.ones((img_height, img_width), dtype=np.uint8) * 255
    padded[:, :new_w] = gray
    
    # To tensor
    tensor = torch.FloatTensor(padded) / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    return tensor


def load_model(model_path: str):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    char_list = checkpoint['char_list']
    num_classes = len(char_list) + 1
    
    model = CRNN(64, 1, num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    idx_to_char = {i + 1: c for i, c in enumerate(char_list)}
    idx_to_char[0] = ''
    
    return model, idx_to_char


def predict(model, image_tensor, idx_to_char):
    """Run prediction on image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        predictions = decode_predictions(output, idx_to_char)
    
    return predictions[0]


def main():
    parser = argparse.ArgumentParser(description="Run custom NID OCR model")
    parser.add_argument("image", help="Image file to process")
    parser.add_argument("--model", type=Path, 
                       default=Path("training/checkpoints/best_model.pth"),
                       help="Path to trained model")
    
    args = parser.parse_args()
    
    if not args.model.exists():
        print(f"Error: Model not found at {args.model}")
        print("Train a model first with: python training/scripts/train_easyocr.py")
        sys.exit(1)
    
    print(f"Loading model from {args.model}...")
    model, idx_to_char = load_model(str(args.model))
    
    print(f"Processing {args.image}...")
    img_tensor = preprocess_image(args.image)
    
    result = predict(model, img_tensor, idx_to_char)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
