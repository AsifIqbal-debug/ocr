"""
OCR Model Accuracy Comparison Tool
===================================
Compare accuracy of ocr.py (EasyOCR), ocr_surya.py (Surya), and nid_ocr.py (NID Pipeline)
against ground truth data for Bangladesh NID cards.

Usage:
    python compare_ocr_accuracy.py image.jpg --ground-truth ground_truth.json
    python compare_ocr_accuracy.py --batch images/ --ground-truth-dir gt/
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import difflib


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for a single field."""
    exact_match: bool
    character_accuracy: float  # Character-level accuracy (0-1)
    word_accuracy: float  # Word-level accuracy (0-1)
    levenshtein_distance: int
    predicted: str
    ground_truth: str


@dataclass
class ModelResult:
    """Results for a single model."""
    model_name: str
    total_time: float
    fields: Dict[str, AccuracyMetrics]
    overall_char_accuracy: float
    overall_word_accuracy: float
    overall_exact_match_rate: float


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def character_accuracy(predicted: str, ground_truth: str) -> float:
    """Calculate character-level accuracy."""
    if not ground_truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    
    # Use sequence matcher for character alignment
    matcher = difflib.SequenceMatcher(None, predicted, ground_truth)
    return matcher.ratio()


def word_accuracy(predicted: str, ground_truth: str) -> float:
    """Calculate word-level accuracy."""
    pred_words = set(predicted.lower().split())
    gt_words = set(ground_truth.lower().split())
    
    if not gt_words:
        return 1.0 if not pred_words else 0.0
    
    correct = len(pred_words & gt_words)
    total = len(gt_words)
    
    return correct / total if total > 0 else 0.0


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    # Remove extra spaces, convert to lowercase for comparison
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def calculate_metrics(predicted: str, ground_truth: str) -> AccuracyMetrics:
    """Calculate accuracy metrics for a field."""
    pred = normalize_text(predicted or "")
    gt = normalize_text(ground_truth or "")
    
    return AccuracyMetrics(
        exact_match=(pred.lower() == gt.lower()),
        character_accuracy=character_accuracy(pred, gt),
        word_accuracy=word_accuracy(pred, gt),
        levenshtein_distance=levenshtein_distance(pred, gt),
        predicted=pred,
        ground_truth=gt
    )


def run_easyocr(image_path: str) -> Dict[str, str]:
    """Run EasyOCR (ocr.py) and extract NID fields from raw text."""
    import cv2
    import easyocr
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    
    # Run OCR
    reader = easyocr.Reader(['bn', 'en'], gpu=False, verbose=False)
    results = reader.readtext(processed, detail=0)
    
    # Parse results to extract NID fields
    fields = {
        "nid_number": None,
        "name_bangla": None,
        "name_english": None,
        "father_name": None,
        "mother_name": None,
        "date_of_birth": None,
    }
    
    full_text = " ".join(results)
    
    # Extract NID number
    nid_match = re.search(r'(?:ID\s*(?:NO|No)?[:\s]*)?(\d[\d\s]{9,17})', full_text)
    if nid_match:
        fields["nid_number"] = re.sub(r'\s+', '', nid_match.group(1))
    
    # Extract date of birth
    dob_match = re.search(r'(?:Date\s*of\s*Birth|DOB)[:\s]*(\d{1,2}\s*\w+\s*\d{4})', full_text, re.IGNORECASE)
    if dob_match:
        fields["date_of_birth"] = dob_match.group(1)
    
    # Extract names from results
    for i, line in enumerate(results):
        line_lower = line.lower()
        
        # English name (usually uppercase)
        if line.isupper() and len(line) > 5 and re.match(r'^[A-Z\s]+$', line):
            if not fields["name_english"]:
                fields["name_english"] = line
        
        # Bangla name after নাম:
        if 'নাম' in line and i + 1 < len(results):
            next_line = results[i + 1]
            if re.search(r'[\u0980-\u09FF]', next_line):
                fields["name_bangla"] = next_line
        
        # Father name after পিতা:
        if 'পিতা' in line:
            # Extract text after পিতা:
            match = re.search(r'পিতা[:\s]*(.+)', line)
            if match:
                fields["father_name"] = match.group(1).strip()
            elif i + 1 < len(results):
                fields["father_name"] = results[i + 1]
        
        # Mother name after মাতা:
        if 'মাতা' in line:
            match = re.search(r'মাতা[:\s]*(.+)', line)
            if match:
                fields["mother_name"] = match.group(1).strip()
            elif i + 1 < len(results):
                fields["mother_name"] = results[i + 1]
    
    return fields


def run_surya(image_path: str) -> Dict[str, str]:
    """Run Surya OCR (ocr_surya.py) and extract NID fields."""
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from PIL import Image
    import cv2
    
    # Load and preprocess
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    if w < 1500:
        scale = 2 if w >= 800 else 3
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_rgb)
    
    # Run Surya
    foundation = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation)
    
    results = rec_predictor(
        [image],
        det_predictor=det_predictor,
        sort_lines=True,
        math_mode=False,
    )
    
    # Extract text lines
    lines = []
    if results and len(results) > 0:
        for line in results[0].text_lines:
            text = line.text.strip()
            text = re.sub(r'<[^>]+>', '', text)
            if text:
                lines.append(text)
    
    # Parse to NID fields
    fields = {
        "nid_number": None,
        "name_bangla": None,
        "name_english": None,
        "father_name": None,
        "mother_name": None,
        "date_of_birth": None,
    }
    
    full_text = " ".join(lines)
    
    # Extract NID number
    nid_match = re.search(r'ID\s*NO[:\s]*(\d+)', full_text, re.IGNORECASE)
    if nid_match:
        fields["nid_number"] = nid_match.group(1)
    else:
        nid_match = re.search(r'(\d{10,17})', full_text)
        if nid_match:
            fields["nid_number"] = nid_match.group(1)
    
    # Extract date of birth
    dob_match = re.search(r'Date\s*of\s*Birth[:\s]*(\d{1,2}\s*\w+\s*\d{4})', full_text, re.IGNORECASE)
    if dob_match:
        fields["date_of_birth"] = dob_match.group(1)
    
    # Extract from lines
    for i, line in enumerate(lines):
        # English name
        if line.isupper() and len(line) > 5 and re.match(r'^[A-Z\s]+$', line):
            if not fields["name_english"]:
                fields["name_english"] = line
        
        # Name patterns
        if re.search(r'^Name[:\s]', line, re.IGNORECASE):
            match = re.search(r'Name[:\s]*(.+)', line, re.IGNORECASE)
            if match and match.group(1).strip():
                fields["name_english"] = match.group(1).strip().upper()
        
        # Bangla fields
        if 'নাম' in line and ':' in line:
            match = re.search(r'নাম[:\s]*(.+)', line)
            if match:
                fields["name_bangla"] = match.group(1).strip()
        
        if 'পিতা' in line:
            match = re.search(r'পিতা[:\s]*(.+)', line)
            if match:
                fields["father_name"] = match.group(1).strip()
        
        if 'মাতা' in line:
            match = re.search(r'মাতা[:\s]*(.+)', line)
            if match:
                fields["mother_name"] = match.group(1).strip()
    
    return fields


def run_nid_ocr(image_path: str) -> Dict[str, str]:
    """Run NID OCR pipeline (nid_ocr.py) with full image mode."""
    # Import the NID pipeline
    sys.path.insert(0, str(Path(image_path).parent))
    from nid_ocr import NIDOCRPipeline
    
    pipeline = NIDOCRPipeline()
    # Use full image OCR mode (better for most images)
    result = pipeline.process_full_image_ocr(image_path)
    
    # Extract fields from structured result
    fields = {
        "nid_number": result.get("nid_number", {}).get("value"),
        "name_bangla": result.get("name_bangla", {}).get("value"),
        "name_english": result.get("name_english", {}).get("value"),
        "father_name": result.get("father_name", {}).get("value"),
        "mother_name": result.get("mother_name", {}).get("value"),
        "date_of_birth": result.get("date_of_birth", {}).get("value"),
    }
    
    # Convert date format if needed (YYYY-MM-DD to DD Mon YYYY for comparison)
    if fields["date_of_birth"] and re.match(r'\d{4}-\d{2}-\d{2}', fields["date_of_birth"]):
        # Keep as is for fair comparison, or convert
        pass
    
    return fields


def evaluate_model(model_name: str, model_func, image_path: str, 
                   ground_truth: Dict[str, str]) -> ModelResult:
    """Evaluate a single model against ground truth."""
    print(f"\n{'='*60}")
    print(f"Running {model_name}...")
    print('='*60)
    
    start_time = time.time()
    try:
        predictions = model_func(image_path)
    except Exception as e:
        print(f"Error running {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return ModelResult(
            model_name=model_name,
            total_time=0,
            fields={},
            overall_char_accuracy=0,
            overall_word_accuracy=0,
            overall_exact_match_rate=0
        )
    
    elapsed = time.time() - start_time
    
    # Calculate metrics for each field
    field_metrics = {}
    total_char_acc = 0
    total_word_acc = 0
    exact_matches = 0
    valid_fields = 0
    
    for field_name, gt_value in ground_truth.items():
        if field_name in ["address", "blood_group", "birth_place"]:
            continue  # Skip complex fields for now
        
        pred_value = predictions.get(field_name, "")
        
        if gt_value:  # Only evaluate if ground truth exists
            metrics = calculate_metrics(pred_value, gt_value)
            field_metrics[field_name] = metrics
            
            total_char_acc += metrics.character_accuracy
            total_word_acc += metrics.word_accuracy
            if metrics.exact_match:
                exact_matches += 1
            valid_fields += 1
            
            # Print comparison
            status = "✓" if metrics.exact_match else "✗"
            print(f"\n{field_name}:")
            print(f"  Ground Truth: {gt_value}")
            print(f"  Predicted:    {pred_value or '(empty)'}")
            print(f"  Char Acc: {metrics.character_accuracy:.1%} | Word Acc: {metrics.word_accuracy:.1%} | {status}")
    
    # Calculate overall metrics
    overall_char = total_char_acc / valid_fields if valid_fields > 0 else 0
    overall_word = total_word_acc / valid_fields if valid_fields > 0 else 0
    exact_rate = exact_matches / valid_fields if valid_fields > 0 else 0
    
    print(f"\n{model_name} Summary:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Overall Character Accuracy: {overall_char:.1%}")
    print(f"  Overall Word Accuracy: {overall_word:.1%}")
    print(f"  Exact Match Rate: {exact_rate:.1%} ({exact_matches}/{valid_fields})")
    
    return ModelResult(
        model_name=model_name,
        total_time=elapsed,
        fields={k: asdict(v) for k, v in field_metrics.items()},
        overall_char_accuracy=overall_char,
        overall_word_accuracy=overall_word,
        overall_exact_match_rate=exact_rate
    )


def compare_models(image_path: str, ground_truth: Dict[str, str], 
                   output_path: Path = None) -> Dict[str, Any]:
    """Compare all three OCR models."""
    
    print("\n" + "="*70)
    print("    OCR MODEL ACCURACY COMPARISON")
    print("    Bangladesh NID Card Recognition")
    print("="*70)
    print(f"\nImage: {image_path}")
    print(f"Ground Truth Fields: {list(ground_truth.keys())}")
    
    results = {}
    
    # Test EasyOCR
    try:
        results["easyocr"] = evaluate_model("EasyOCR (ocr.py)", run_easyocr, image_path, ground_truth)
    except Exception as e:
        print(f"EasyOCR failed: {e}")
    
    # Test Surya
    try:
        results["surya"] = evaluate_model("Surya (ocr_surya.py)", run_surya, image_path, ground_truth)
    except Exception as e:
        print(f"Surya failed: {e}")
    
    # Test NID Pipeline
    try:
        results["nid_ocr"] = evaluate_model("NID Pipeline (nid_ocr.py)", run_nid_ocr, image_path, ground_truth)
    except Exception as e:
        print(f"NID Pipeline failed: {e}")
    
    # Print comparison table
    print("\n" + "="*70)
    print("    FINAL COMPARISON")
    print("="*70)
    print(f"\n{'Model':<25} {'Char Acc':<12} {'Word Acc':<12} {'Exact Match':<12} {'Time':<10}")
    print("-"*70)
    
    for name, result in results.items():
        if isinstance(result, ModelResult):
            print(f"{result.model_name:<25} {result.overall_char_accuracy:>10.1%} {result.overall_word_accuracy:>10.1%} {result.overall_exact_match_rate:>10.1%} {result.total_time:>8.2f}s")
    
    # Determine winner
    best_model = max(results.items(), key=lambda x: x[1].overall_char_accuracy if isinstance(x[1], ModelResult) else 0)
    print(f"\n🏆 Best Model: {best_model[0]} (Character Accuracy: {best_model[1].overall_char_accuracy:.1%})")
    
    # Save results
    output = {
        "image": str(image_path),
        "ground_truth": ground_truth,
        "results": {k: asdict(v) if isinstance(v, ModelResult) else v for k, v in results.items()},
        "winner": best_model[0]
    }
    
    if output_path:
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResults saved to: {output_path}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Compare OCR model accuracy")
    parser.add_argument("image", help="NID card image to process")
    parser.add_argument("--ground-truth", "-gt", type=Path, required=True,
                        help="Ground truth JSON file")
    parser.add_argument("--output", "-o", type=Path, 
                        help="Output JSON file for results")
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    if not args.ground_truth.exists():
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        print("\nCreate a ground truth JSON file with format:")
        print(json.dumps({
            "nid_number": "1234567890",
            "name_bangla": "নাম বাংলায়",
            "name_english": "NAME IN ENGLISH",
            "father_name": "পিতার নাম",
            "mother_name": "মাতার নাম",
            "date_of_birth": "01 Jan 1990"
        }, ensure_ascii=False, indent=2))
        sys.exit(1)
    
    # Load ground truth
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    
    # Run comparison
    compare_models(args.image, ground_truth, args.output)


if __name__ == "__main__":
    main()
