"""
Bangla OCR Post-Processing Corrections
======================================
This module provides a correction system for common OCR errors in Bangla text,
especially for NID cards.
"""

import re
import json
from pathlib import Path

# Common Bangla OCR errors and their corrections
# Format: wrong_text -> correct_text
BANGLA_CORRECTIONS = {
    # Similar character confusions
    'জাকিব': 'জাকির',
    'হেসাইন': 'হোসাইন',
    'থেসাইন': 'হোসাইন',
    'মোঢাঃ': 'মোসাঃ',
    'মোডাঃ': 'মোসাঃ',
    'ইমতিয়াজ্জগ': 'ইমতিয়াজ',
    'মির্জ্জা': 'মির্জা',
    'শিত:': 'পিতা:',
    'সাতা:': 'মাতা:',
    'রাব্বানী': 'রাব্বানী',  # Keep correct
    'মোন্তফা': 'মোস্তফা',
    'মোন্তাফা': 'মোস্তফা',
    'মোছাঃ': 'মোছাঃ',  # Keep correct (or মোসাঃ)
    'বোকেয়া': 'রোকেয়া',
    
    # Common prefix corrections
    'মোঃ': 'মোঃ',  # Keep correct
    'ম:': 'মোঃ',
    'মাে:': 'মোঃ',
    
    # Hindi to Bangla character corrections (when Hindi detected)
    'नाम': 'নাম',
    'पिता': 'পিতা',
    'माता': 'মাতা',
}

# English corrections
ENGLISH_CORRECTIONS = {
    'Govemmen': 'Government',
    'Govemment': 'Government',
    'Goverment': 'Government',
    'IDGOLAM': 'MD GOLAM',
    'RABBANIE': 'RABBANI',
    'BABL': 'BABU',
    'IID No': 'NID No',
    'IID NO': 'NID NO',
    'NOORALAI': 'NOOR ALAM',
    'ROMANARAHHAN': 'ROMANA RAHMAN',
    'Birtn': 'Birth',
    'Fen': 'Feb',
}

# Hindi word to Bangla word mappings - more reliable than character mapping
HINDI_WORD_TO_BANGLA = {
    'नाम': 'নাম',
    'नाम:': 'নাম:',
    'पिता': 'পিতা',
    'पिता:': 'পিতা:',
    'माता': 'মাতা', 
    'माता:': 'মাতা:',
    'जन्म': 'জন্ম',
    'तारीख': 'তারিখ',
}

# Character-level corrections (Hindi to Bangla) - only numbers, punctuation safe
HINDI_TO_BANGLA = {
    # Only map numbers (safe, no combining issues)
    '०': '০', '१': '১', '२': '২', '३': '৩', '४': '৪',
    '५': '৫', '६': '৬', '७': '৭', '८': '৮', '९': '৯',
}


class BanglaOCRCorrector:
    """Post-processing corrector for Bangla OCR output."""
    
    def __init__(self, custom_corrections_file=None):
        self.bangla_corrections = BANGLA_CORRECTIONS.copy()
        self.english_corrections = ENGLISH_CORRECTIONS.copy()
        
        # Load custom corrections if provided
        if custom_corrections_file and Path(custom_corrections_file).exists():
            with open(custom_corrections_file, 'r', encoding='utf-8') as f:
                custom = json.load(f)
                self.bangla_corrections.update(custom.get('bangla', {}))
                self.english_corrections.update(custom.get('english', {}))
    
    def convert_hindi_to_bangla(self, text):
        """Convert Hindi script to Bangla equivalents."""
        # First apply word-level Hindi->Bangla replacements (most reliable)
        for hindi, bangla in HINDI_WORD_TO_BANGLA.items():
            text = text.replace(hindi, bangla)
        
        # Then Hindi->Bangla from BANGLA_CORRECTIONS
        for wrong, correct in self.bangla_corrections.items():
            if any('\u0900' <= c <= '\u097F' for c in wrong):  # Hindi range
                text = text.replace(wrong, correct)
        
        # Then number conversion (safe)
        for hindi, bangla in HINDI_TO_BANGLA.items():
            text = text.replace(hindi, bangla)
        return text
    
    def apply_word_corrections(self, text):
        """Apply word-level corrections."""
        # Apply Bangla corrections (skip Hindi->Bangla word corrections already done)
        for wrong, correct in self.bangla_corrections.items():
            if not any('\u0900' <= c <= '\u097F' for c in wrong):  # Skip Hindi
                text = text.replace(wrong, correct)
        
        # Apply English corrections  
        for wrong, correct in self.english_corrections.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def correct(self, text):
        """Apply all corrections to OCR text."""
        if not text:
            return text
        
        # Step 1: Convert any Hindi characters/words to Bangla
        text = self.convert_hindi_to_bangla(text)
        
        # Step 2: Apply word-level corrections
        text = self.apply_word_corrections(text)
        
        # Step 3: Fix common patterns
        # Fix duplicate characters in conjuncts
        text = re.sub(r'র্জ্জ', 'র্জ', text)  # মির্জ্জা -> মির্জা
        text = re.sub(r'্্', '্', text)  # Remove double hasanta
        
        return text
    
    def correct_ocr_result(self, ocr_result):
        """
        Correct an OCR result dictionary.
        
        Args:
            ocr_result: dict with 'text' field or list of dicts
            
        Returns:
            Corrected result
        """
        if isinstance(ocr_result, list):
            return [self.correct_ocr_result(item) for item in ocr_result]
        
        if isinstance(ocr_result, dict) and 'text' in ocr_result:
            ocr_result['text'] = self.correct(ocr_result['text'])
            ocr_result['corrected'] = True
        
        return ocr_result
    
    def add_correction(self, wrong, correct, lang='bangla'):
        """Add a new correction rule."""
        if lang == 'bangla':
            self.bangla_corrections[wrong] = correct
        else:
            self.english_corrections[wrong] = correct
    
    def save_corrections(self, filepath):
        """Save current corrections to a JSON file."""
        data = {
            'bangla': self.bangla_corrections,
            'english': self.english_corrections
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def test_corrections():
    """Test the correction system."""
    corrector = BanglaOCRCorrector()
    
    test_cases = [
        ("মির্জ্জা ইমতিয়াজ্জগ আহমেদ", "মির্জা ইমতিয়াজ আহমেদ"),
        ("মির্জা মোঃ জাকিব থেসাইন", "মির্জা মোঃ জাকির হোসাইন"),
        ("মোঢাঃ বোকেয়া বেগম", "মোসাঃ রোকেয়া বেগম"),
        ("Govemmen of the People's Republic", "Government of the People's Republic"),
        ("IID No", "NID No"),
        ("नाम: মির্জা", "নাম: মির্জা"),  # Hindi to Bangla
    ]
    
    print("=" * 60)
    print("Bangla OCR Correction System - Test Results")
    print("=" * 60)
    
    all_passed = True
    for original, expected in test_cases:
        result = corrector.correct(original)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"\n{status} Original:  {original}")
        print(f"  Expected:  {expected}")
        print(f"  Got:       {result}")
    
    print("\n" + "=" * 60)
    print(f"All tests passed: {all_passed}")
    return all_passed


if __name__ == '__main__':
    test_corrections()
