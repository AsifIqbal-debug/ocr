"""
Bangladesh NID Card OCR Pipeline
=================================
Complete OCR system for Bangladesh National ID cards.

Features:
- NID card detection and alignment
- Region-based cropping for each field
- Surya OCR with Bangla + English
- NID-specific post-processing and validation
- Structured JSON output with confidence scores

Usage:
    python nid_ocr.py image.jpg
    python nid_ocr.py image.jpg --output result.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from PIL import Image
import cv2
import numpy as np


@dataclass
class FieldResult:
    """Result for a single NID field."""
    value: Optional[str]
    confidence: float
    raw_text: Optional[str] = None


@dataclass 
class AddressResult:
    """Structured address result."""
    full: Optional[str]
    village_mohalla: Optional[str] = None
    post_office: Optional[str] = None
    upazila_thana: Optional[str] = None
    district: Optional[str] = None
    confidence: float = 0.0


@dataclass
class NIDResult:
    """Complete NID OCR result."""
    nid_number: FieldResult
    name_bangla: FieldResult
    name_english: FieldResult
    father_name: FieldResult
    mother_name: FieldResult
    date_of_birth: FieldResult
    address: AddressResult
    

# NID field regions (relative coordinates as percentages)
# These are approximate for standard Bangladesh NID cards
NID_REGIONS = {
    "nid_number": {"x": 0.05, "y": 0.85, "w": 0.45, "h": 0.12},
    "name_bangla": {"x": 0.25, "y": 0.28, "w": 0.70, "h": 0.08},
    "name_english": {"x": 0.25, "y": 0.36, "w": 0.70, "h": 0.07},
    "father_name": {"x": 0.25, "y": 0.44, "w": 0.70, "h": 0.07},
    "mother_name": {"x": 0.25, "y": 0.52, "w": 0.70, "h": 0.07},
    "date_of_birth": {"x": 0.25, "y": 0.60, "w": 0.50, "h": 0.07},
}

# Back side regions
NID_BACK_REGIONS = {
    "address": {"x": 0.15, "y": 0.20, "w": 0.80, "h": 0.25},
}


class NIDProcessor:
    """Post-processor for NID OCR results."""
    
    # Character corrections for OCR errors
    DIGIT_CORRECTIONS = {
        'O': '0', 'o': '0', 'Q': '0',
        'I': '1', 'l': '1', '|': '1', 'i': '1',
        'Z': '2', 'z': '2',
        'E': '3',
        'A': '4', 'h': '4',
        'S': '5', 's': '5',
        'G': '6', 'b': '6',
        'T': '7',
        'B': '8',
        'g': '9', 'q': '9',
    }
    
    # Month mappings
    MONTHS_EN = {
        'jan': '01', 'january': '01',
        'feb': '02', 'february': '02',
        'mar': '03', 'march': '03',
        'apr': '04', 'april': '04',
        'may': '05',
        'jun': '06', 'june': '06',
        'jul': '07', 'july': '07',
        'aug': '08', 'august': '08',
        'sep': '09', 'sept': '09', 'september': '09',
        'oct': '10', 'october': '10',
        'nov': '11', 'november': '11',
        'dec': '12', 'december': '12',
    }
    
    MONTHS_BN = {
        'জানু': '01', 'জানুয়ারি': '01', 'জানুয়ারী': '01',
        'ফেব্রু': '02', 'ফেব্রুয়ারি': '02', 'ফেব্রুয়ারী': '02',
        'মার্চ': '03',
        'এপ্রি': '04', 'এপ্রিল': '04',
        'মে': '05',
        'জুন': '06',
        'জুলা': '07', 'জুলাই': '07',
        'আগ': '08', 'আগস্ট': '08',
        'সেপ্টে': '09', 'সেপ্টেম্বর': '09',
        'অক্টো': '10', 'অক্টোবর': '10',
        'নভে': '11', 'নভেম্বর': '11',
        'ডিসে': '12', 'ডিসেম্বর': '12',
    }
    
    # Bangla to English digit mapping
    BN_DIGITS = {'০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
                 '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'}
    
    # Hindi to Bangla character corrections
    HINDI_TO_BANGLA = {
        'न': 'ন', 'ा': 'া', 'म': 'ম', 'ज': 'জ', 'ु': 'ু', 'ल': 'ল',
        'व': 'ব', 'त': 'ত', 'ि': 'ি', 'प': 'প', 'क': 'ক', 'र': 'র',
        'ह': 'হ', 'स': 'স', 'ी': 'ী', 'े': 'ে', 'ो': 'ো', 'द': 'দ',
        'ं': 'ং', 'ः': 'ঃ', 'आ': 'আ', 'इ': 'ই', 'उ': 'উ', 'ए': 'এ',
        'ओ': 'ও', 'अ': 'অ', 'ख': 'খ', 'ग': 'গ', 'घ': 'ঘ', 'च': 'চ',
        'छ': 'ছ', 'झ': 'ঝ', 'ट': 'ট', 'ठ': 'ঠ', 'ड': 'ড', 'ढ': 'ঢ',
        'ण': 'ণ', 'थ': 'থ', 'ध': 'ধ', 'फ': 'ফ', 'ब': 'ব', 'भ': 'ভ',
        'य': 'য', 'श': 'শ', 'ष': 'ষ', 'ै': 'ৈ', 'ौ': 'ৌ', 'ृ': 'ৃ',
        '्': '্', 'ँ': 'ঁ',
    }
    
    def clean_nid_number(self, text: str) -> Tuple[Optional[str], float]:
        """Clean and validate NID number."""
        if not text:
            return None, 0.0
        
        # Remove all non-digit characters except potential OCR errors
        cleaned = text.upper()
        
        # Convert Bangla digits to English
        for bn, en in self.BN_DIGITS.items():
            cleaned = cleaned.replace(bn, en)
        
        # Apply OCR corrections
        result = ""
        for char in cleaned:
            if char.isdigit():
                result += char
            elif char in self.DIGIT_CORRECTIONS:
                result += self.DIGIT_CORRECTIONS[char]
            elif char in ' -.:':
                continue  # Skip separators
        
        # Validate length (10, 13, or 17 digits for Bangladesh NID)
        if len(result) == 10 or len(result) == 13 or len(result) == 17:
            return result, 0.95
        elif 9 <= len(result) <= 18:
            return result, 0.7
        elif len(result) > 0:
            return result, 0.4
        
        return None, 0.0
    
    def clean_date(self, text: str) -> Tuple[Optional[str], float]:
        """Parse and clean date of birth to YYYY-MM-DD format."""
        if not text:
            return None, 0.0
        
        text = text.strip()
        
        # Convert Bangla digits to English
        for bn, en in self.BN_DIGITS.items():
            text = text.replace(bn, en)
        
        # Try various date patterns
        
        # Pattern 1: DD Mon YYYY (e.g., "06 Oct 2004")
        match = re.search(r'(\d{1,2})\s*([A-Za-z]+)\s*(\d{4})', text)
        if match:
            day, month_str, year = match.groups()
            month = self.MONTHS_EN.get(month_str.lower()[:3])
            if month:
                return f"{year}-{month}-{day.zfill(2)}", 0.95
        
        # Pattern 2: DD-MM-YYYY or DD/MM/YYYY
        match = re.search(r'(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})', text)
        if match:
            day, month, year = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}", 0.95
        
        # Pattern 3: YYYY-MM-DD (already correct)
        match = re.search(r'(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})', text)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}", 0.95
        
        # Pattern 4: Bangla date (e.g., "২৭-১১-২০০৪")
        match = re.search(r'(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})', text)
        if match:
            day, month, year = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}", 0.90
        
        # Pattern 5: Just year extraction as fallback
        match = re.search(r'(\d{4})', text)
        if match:
            year = match.group(1)
            if 1900 <= int(year) <= 2025:
                return f"{year}-01-01", 0.5
        
        return None, 0.0
    
    def clean_name_bangla(self, text: str) -> Tuple[Optional[str], float]:
        """Clean Bangla name."""
        if not text:
            return None, 0.0
        
        # Remove field labels
        text = re.sub(r'^(নাম|নামঃ|নাম:)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(পিতা|পিতাঃ|পিতা:)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(মাতা|মাতাঃ|মাতা:)\s*', '', text, flags=re.IGNORECASE)
        
        # Convert Hindi characters to Bangla
        for hindi, bangla in self.HINDI_TO_BANGLA.items():
            text = text.replace(hindi, bangla)
        
        # Remove non-Bangla characters (keep spaces and some punctuation)
        cleaned = ""
        for char in text:
            if '\u0980' <= char <= '\u09FF' or char in ' .:।':
                cleaned += char
        
        cleaned = cleaned.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize spaces
        
        if cleaned:
            # Check if mostly Bangla
            bangla_count = len(re.findall(r'[\u0980-\u09FF]', cleaned))
            if bangla_count > len(cleaned) * 0.5:
                return cleaned, 0.9
            return cleaned, 0.6
        
        return None, 0.0
    
    def clean_name_english(self, text: str) -> Tuple[Optional[str], float]:
        """Clean English name."""
        if not text:
            return None, 0.0
        
        # Remove field labels
        text = re.sub(r'^(Name|NAME|name)[:.]?\s*', '', text, flags=re.IGNORECASE)
        
        # Keep only ASCII letters and spaces
        cleaned = ""
        for char in text:
            if char.isascii() and (char.isalpha() or char == ' '):
                cleaned += char
        
        # Normalize to uppercase
        cleaned = cleaned.upper().strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize spaces
        
        if cleaned and len(cleaned) > 2:
            return cleaned, 0.95
        
        return None, 0.0
    
    def clean_address(self, text: str) -> AddressResult:
        """Parse and structure address."""
        if not text:
            return AddressResult(full=None, confidence=0.0)
        
        # Convert Hindi to Bangla
        for hindi, bangla in self.HINDI_TO_BANGLA.items():
            text = text.replace(hindi, bangla)
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        result = AddressResult(full=text, confidence=0.7)
        
        # Extract district (জেলা)
        match = re.search(r'জেলা[:\s]*([^\s,।]+)', text)
        if match:
            result.district = match.group(1)
            result.confidence = 0.85
        
        # Extract upazila/thana (উপজেলা/থানা)
        match = re.search(r'(উপজেলা|থানা)[:\s]*([^\s,।]+)', text)
        if match:
            result.upazila_thana = match.group(2)
            result.confidence = 0.85
        
        # Extract post office (ডাকঘর)
        match = re.search(r'(ডাকঘর|পোস্ট)[:\s]*([^\s,।]+)', text)
        if match:
            result.post_office = match.group(2)
        
        # Extract village/mohalla (গ্রাম/মহল্লা)
        match = re.search(r'(গ্রাম|মহল্লা|বাসা|হোল্ডিং)[:\s]*([^\s,।]+)', text)
        if match:
            result.village_mohalla = match.group(2)
        
        return result


class NIDOCRPipeline:
    """Complete NID OCR pipeline."""
    
    def __init__(self):
        self.processor = NIDProcessor()
        self._surya_loaded = False
        self.foundation = None
        self.det_predictor = None
        self.rec_predictor = None
    
    def _load_surya(self):
        """Lazy load Surya models."""
        if self._surya_loaded:
            return
        
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        
        self.foundation = FoundationPredictor()
        self.det_predictor = DetectionPredictor()
        self.rec_predictor = RecognitionPredictor(self.foundation)
        self._surya_loaded = True
    
    def detect_and_align_nid(self, image: np.ndarray) -> np.ndarray:
        """Detect NID card and align it."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest rectangle-like contour
            largest = max(contours, key=cv2.contourArea)
            
            # Approximate to polygon
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            
            if len(approx) == 4 and cv2.contourArea(approx) > image.shape[0] * image.shape[1] * 0.1:
                # Order points: top-left, top-right, bottom-right, bottom-left
                pts = approx.reshape(4, 2)
                rect = self._order_points(pts)
                
                # Compute destination dimensions
                width = max(
                    np.linalg.norm(rect[0] - rect[1]),
                    np.linalg.norm(rect[2] - rect[3])
                )
                height = max(
                    np.linalg.norm(rect[0] - rect[3]),
                    np.linalg.norm(rect[1] - rect[2])
                )
                
                # Standard NID aspect ratio is approximately 1.585:1
                if width > height:
                    dst = np.array([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ], dtype="float32")
                else:
                    # Card is vertical, need to rotate
                    dst = np.array([
                        [0, 0],
                        [height - 1, 0],
                        [height - 1, width - 1],
                        [0, width - 1]
                    ], dtype="float32")
                    width, height = height, width
                
                # Perspective transform
                M = cv2.getPerspectiveTransform(rect, dst)
                aligned = cv2.warpPerspective(image, M, (int(width), int(height)))
                return aligned
        
        # Return original if detection fails
        return image
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        return rect
    
    def crop_region(self, image: np.ndarray, region: dict) -> np.ndarray:
        """Crop a region from the image."""
        h, w = image.shape[:2]
        
        x1 = int(region["x"] * w)
        y1 = int(region["y"] * h)
        x2 = int((region["x"] + region["w"]) * w)
        y2 = int((region["y"] + region["h"]) * h)
        
        # Add padding
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        return image[y1:y2, x1:x2]
    
    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess cropped region for better OCR."""
        # Upscale if small
        h, w = crop.shape[:2]
        if w < 300:
            scale = 3
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        
        # Apply CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Convert back to BGR for Surya
        result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def ocr_crop(self, crop: np.ndarray) -> Tuple[str, float]:
        """Run OCR on a cropped region."""
        self._load_surya()
        
        # Preprocess
        processed = self.preprocess_crop(crop)
        
        # Convert to PIL
        image = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        
        # Run OCR
        results = self.rec_predictor(
            [image],
            det_predictor=self.det_predictor,
            sort_lines=True,
            math_mode=False,
        )
        
        if results and len(results) > 0:
            lines = []
            confidences = []
            for line in results[0].text_lines:
                text = line.text.strip()
                # Clean HTML tags
                text = re.sub(r'<[^>]+>', '', text)
                if text:
                    lines.append(text)
                    if hasattr(line, 'confidence'):
                        confidences.append(line.confidence)
            
            combined = " ".join(lines)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
            return combined, avg_conf
        
        return "", 0.0
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process a single NID image and return structured results."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Cannot read image", "confidence": 0.0}
        
        # Detect and align NID card
        aligned = self.detect_and_align_nid(image)
        
        # Process each field
        results = {}
        
        # Process front side fields
        for field_name, region in NID_REGIONS.items():
            crop = self.crop_region(aligned, region)
            raw_text, ocr_conf = self.ocr_crop(crop)
            
            if field_name == "nid_number":
                value, conf = self.processor.clean_nid_number(raw_text)
                results[field_name] = {
                    "value": value,
                    "confidence": round(min(conf, ocr_conf), 2),
                    "raw": raw_text
                }
            
            elif field_name == "date_of_birth":
                value, conf = self.processor.clean_date(raw_text)
                results[field_name] = {
                    "value": value,
                    "confidence": round(min(conf, ocr_conf), 2),
                    "raw": raw_text
                }
            
            elif field_name == "name_bangla":
                value, conf = self.processor.clean_name_bangla(raw_text)
                results[field_name] = {
                    "value": value,
                    "confidence": round(min(conf, ocr_conf), 2),
                    "raw": raw_text
                }
            
            elif field_name == "name_english":
                value, conf = self.processor.clean_name_english(raw_text)
                results[field_name] = {
                    "value": value,
                    "confidence": round(min(conf, ocr_conf), 2),
                    "raw": raw_text
                }
            
            elif field_name == "father_name":
                value, conf = self.processor.clean_name_bangla(raw_text)
                results[field_name] = {
                    "value": value,
                    "confidence": round(min(conf, ocr_conf), 2),
                    "raw": raw_text
                }
            
            elif field_name == "mother_name":
                value, conf = self.processor.clean_name_bangla(raw_text)
                results[field_name] = {
                    "value": value,
                    "confidence": round(min(conf, ocr_conf), 2),
                    "raw": raw_text
                }
        
        # Address (typically on back side, but try full image OCR)
        full_text, full_conf = self.ocr_crop(aligned)
        addr_result = self.processor.clean_address(full_text)
        results["address"] = {
            "full": addr_result.full,
            "village_mohalla": addr_result.village_mohalla,
            "post_office": addr_result.post_office,
            "upazila_thana": addr_result.upazila_thana,
            "district": addr_result.district,
            "confidence": round(addr_result.confidence * full_conf, 2)
        }
        
        return results
    
    def process_full_image_ocr(self, image_path: str) -> Dict[str, Any]:
        """
        Alternative: Run OCR on full image and extract fields from text.
        Better for images where region cropping doesn't work well.
        """
        self._load_surya()
        
        # Load and preprocess
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Cannot read image", "confidence": 0.0}
        
        # Upscale if needed
        h, w = image.shape[:2]
        if w < 1500:
            scale = 2 if w >= 800 else 3
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Run OCR
        results = self.rec_predictor(
            [pil_image],
            det_predictor=self.det_predictor,
            sort_lines=True,
            math_mode=False,
        )
        
        if not results or len(results) == 0:
            return {"error": "OCR failed", "confidence": 0.0}
        
        # Collect all text lines with confidence
        lines = []
        for line in results[0].text_lines:
            text = line.text.strip()
            text = re.sub(r'<[^>]+>', '', text)
            conf = line.confidence if hasattr(line, 'confidence') else 0.5
            if text:
                lines.append({"text": text, "conf": conf})
        
        # Extract fields from text
        return self._extract_fields_from_lines(lines)
    
    def _extract_fields_from_lines(self, lines: list) -> Dict[str, Any]:
        """Extract NID fields from OCR text lines."""
        results = {
            "nid_number": {"value": None, "confidence": 0.0, "raw": None},
            "name_bangla": {"value": None, "confidence": 0.0, "raw": None},
            "name_english": {"value": None, "confidence": 0.0, "raw": None},
            "father_name": {"value": None, "confidence": 0.0, "raw": None},
            "mother_name": {"value": None, "confidence": 0.0, "raw": None},
            "date_of_birth": {"value": None, "confidence": 0.0, "raw": None},
            "address": {"full": None, "confidence": 0.0},
        }
        
        # Collect all Bangla names that appear after নাম/পিতা/মাতা labels
        bangla_names_queue = []
        
        for i, line in enumerate(lines):
            text = line["text"]
            conf = line["conf"]
            text_lower = text.lower()
            
            # NID Number - look for digit sequences
            if re.search(r'(id\s*no|nid\s*no|আইডি)', text_lower):
                nid_match = re.search(r'[\d\s]{10,}', text)
                if nid_match:
                    value, nid_conf = self.processor.clean_nid_number(nid_match.group())
                    if value:
                        results["nid_number"] = {
                            "value": value,
                            "confidence": round(min(nid_conf, conf), 2),
                            "raw": text
                        }
            # Also check for standalone number sequences (NID on separate line)
            elif re.match(r'^[\d\s\-\.]+$', text):
                cleaned = re.sub(r'[\s\-\.]', '', text)
                if 10 <= len(cleaned) <= 17:
                    value, nid_conf = self.processor.clean_nid_number(text)
                    if value and not results["nid_number"]["value"]:
                        results["nid_number"] = {
                            "value": value,
                            "confidence": round(min(nid_conf, conf), 2),
                            "raw": text
                        }
            
            # Date of Birth
            if re.search(r'(date\s*of\s*birth|birth|জন্ম)', text_lower):
                value, dob_conf = self.processor.clean_date(text)
                if value:
                    results["date_of_birth"] = {
                        "value": value,
                        "confidence": round(min(dob_conf, conf), 2),
                        "raw": text
                    }
            # Also check for date patterns without label
            elif re.search(r'\d{1,2}\s+[A-Za-z]+\s+\d{4}', text):
                value, dob_conf = self.processor.clean_date(text)
                if value and not results["date_of_birth"]["value"]:
                    results["date_of_birth"] = {
                        "value": value,
                        "confidence": round(min(dob_conf * 0.8, conf), 2),
                        "raw": text
                    }
            
            # Name (Bangla) - look for নাম or text after নাম label
            if re.search(r'নাম', text):
                # Try to extract name from same line
                name_part = re.sub(r'^নাম[:\s]*', '', text)
                # Convert Hindi to Bangla
                for hindi, bangla in self.processor.HINDI_TO_BANGLA.items():
                    name_part = name_part.replace(hindi, bangla)
                
                if name_part and len(name_part) > 2:
                    value, name_conf = self.processor.clean_name_bangla(name_part)
                    if value and not results["name_bangla"]["value"]:
                        results["name_bangla"] = {
                            "value": value,
                            "confidence": round(min(name_conf, conf), 2),
                            "raw": text
                        }
            
            # Check for standalone Bangla text that could be a name (after নাম label)
            bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
            if bangla_chars > 5 and not re.search(r'(সরকার|পরিচয়|জাতীয়|বাংলাদেশ|কার্ড|Card|National)', text):
                bangla_names_queue.append({"text": text, "conf": conf, "idx": i})
            
            # Name (English) - look for Name: or all caps text
            if re.match(r'^name', text_lower):
                value, name_conf = self.processor.clean_name_english(text)
                if value:
                    results["name_english"] = {
                        "value": value,
                        "confidence": round(min(name_conf, conf), 2),
                        "raw": text
                    }
            elif text.isupper() and len(text) > 5 and text.replace(" ", "").isalpha():
                # Check it's not a header
                if not re.search(r'(NATIONAL|CARD|GOVERNMENT|BANGLADESH|REPUBLIC)', text):
                    value, name_conf = self.processor.clean_name_english(text)
                    if value and not results["name_english"]["value"]:
                        results["name_english"] = {
                            "value": value,
                            "confidence": round(min(name_conf, conf), 2),
                            "raw": text
                        }
            
            # Father name - handle OCR errors like 'পিত', 'পতা', 'পত' instead of 'পিতা'
            if re.search(r'(পিতা|পিতার|পিত[^া]|^পিত$|পতা|^পতা?$)', text):
                name_part = re.sub(r'^(পিতার?\s*নাম|পিতা?|পতা?)[:\s]*', '', text)
                for hindi, bangla in self.processor.HINDI_TO_BANGLA.items():
                    name_part = name_part.replace(hindi, bangla)
                if name_part and len(name_part) > 2:
                    value, name_conf = self.processor.clean_name_bangla(name_part)
                    if value:
                        results["father_name"] = {
                            "value": value,
                            "confidence": round(min(name_conf, conf), 2),
                            "raw": text
                        }
                # If label only (পিতা/পতা on its own line), check next line for the name
                elif i + 1 < len(lines) and not results["father_name"]["value"]:
                    next_text = lines[i + 1]["text"]
                    next_conf = lines[i + 1]["conf"]
                    for hindi, bangla in self.processor.HINDI_TO_BANGLA.items():
                        next_text = next_text.replace(hindi, bangla)
                    value, name_conf = self.processor.clean_name_bangla(next_text)
                    if value:
                        results["father_name"] = {
                            "value": value,
                            "confidence": round(min(name_conf, next_conf) * 0.9, 2),
                            "raw": f"{text} {next_text}"
                        }
            
            # Mother name - handle OCR errors like 'য়াতা', 'সাতা', 'মাত' instead of 'মাতা'
            if re.search(r'(মাতা|মাতার|মাত[^া]|^মাত$|য়াতা|সাতা)', text):
                name_part = re.sub(r'^(মাতার?\s*নাম|মাতা?|য়াতা|সাতা)[:\s]*', '', text)
                for hindi, bangla in self.processor.HINDI_TO_BANGLA.items():
                    name_part = name_part.replace(hindi, bangla)
                if name_part and len(name_part) > 2:
                    value, name_conf = self.processor.clean_name_bangla(name_part)
                    if value:
                        results["mother_name"] = {
                            "value": value,
                            "confidence": round(min(name_conf, conf), 2),
                            "raw": text
                        }
                # If label only, check next line for the name
                elif i + 1 < len(lines) and not results["mother_name"]["value"]:
                    next_text = lines[i + 1]["text"]
                    next_conf = lines[i + 1]["conf"]
                    for hindi, bangla in self.processor.HINDI_TO_BANGLA.items():
                        next_text = next_text.replace(hindi, bangla)
                    value, name_conf = self.processor.clean_name_bangla(next_text)
                    if value:
                        results["mother_name"] = {
                            "value": value,
                            "confidence": round(min(name_conf, next_conf) * 0.9, 2),
                            "raw": f"{text} {next_text}"
                        }
                # Also check PREVIOUS line if label is alone (OCR sometimes puts label after name)
                if not results["mother_name"]["value"] and i > 0:
                    prev_text = lines[i - 1]["text"]
                    prev_conf = lines[i - 1]["conf"]
                    # Only use if it looks like a Bangla name (not a label or English text)
                    if re.search(r'[\u0980-\u09FF]{3,}', prev_text) and not re.search(r'(পিতা|নাম|মাতা|সরকার|বাংলাদেশ|পরিচয়)', prev_text):
                        for hindi, bangla in self.processor.HINDI_TO_BANGLA.items():
                            prev_text = prev_text.replace(hindi, bangla)
                        value, name_conf = self.processor.clean_name_bangla(prev_text)
                        if value:
                            results["mother_name"] = {
                                "value": value,
                                "confidence": round(min(name_conf, prev_conf) * 0.85, 2),
                                "raw": f"{prev_text} {text}"
                            }
        
        # If we didn't find name_bangla but have Bangla text candidates, use the first one
        # that appears after header text
        if not results["name_bangla"]["value"] and bangla_names_queue:
            for item in bangla_names_queue:
                text = item["text"]
                # Skip if it looks like father/mother name
                if re.search(r'(পিতা|মাতা|মোঃ\s+)', text) and results["father_name"]["value"]:
                    continue
                value, name_conf = self.processor.clean_name_bangla(text)
                if value:
                    results["name_bangla"] = {
                        "value": value,
                        "confidence": round(min(name_conf * 0.8, item["conf"]), 2),
                        "raw": text
                    }
                    break
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Bangladesh NID Card OCR")
    parser.add_argument("image", help="NID card image file")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file")
    parser.add_argument("--mode", choices=["region", "full"], default="full",
                        help="OCR mode: 'region' for cropped regions, 'full' for full image")
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(json.dumps({"error": f"Image not found: {args.image}"}))
        sys.exit(1)
    
    pipeline = NIDOCRPipeline()
    
    if args.mode == "region":
        result = pipeline.process_image(args.image)
    else:
        result = pipeline.process_full_image_ocr(args.image)
    
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    
    if args.output:
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
