"""OCR error correction functions for clinical notes."""

import re
import json
import pandas as pd


def load_ocr_config(config_path):
    """Load OCR config from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['ocr_substitutions'], set(config['valid_mixed_terms'])


def has_letter_and_digit(word):
    """Check if word contains both letters and digits."""
    has_letter = any(c.isalpha() for c in word)
    has_digit = any(c.isdigit() for c in word)
    return has_letter and has_digit


def fix_ocr_word(word, ocr_subs, valid_terms):
    """Fix OCR errors in a single word."""
    if not has_letter_and_digit(word):
        return word
    if word.lower() in valid_terms:
        return word
    fixed = word.lower()
    for digit, letter in ocr_subs.items():
        fixed = fixed.replace(digit, letter)
    return fixed


def fix_ocr_text(text, ocr_subs, valid_terms):
    """Fix OCR errors in entire text."""
    if pd.isna(text):
        return text
    text = str(text)
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text)
    for word in words:
        if has_letter_and_digit(word) and word.lower() not in valid_terms:
            fixed = fix_ocr_word(word, ocr_subs, valid_terms)
            if fixed != word.lower():
                if word[0].isupper():
                    fixed = fixed.capitalize()
                text = text.replace(word, fixed)
    return text


def has_ocr_errors(text, valid_terms):
    """Check if text has potential OCR errors."""
    if pd.isna(text):
        return False
    words = re.findall(r'\b[a-zA-Z0-9]+\b', str(text))
    for word in words:
        if has_letter_and_digit(word) and word.lower() not in valid_terms:
            return True
    return False


def strip_html(text):
    """Remove HTML tags from text."""
    if pd.isna(text):
        return text
    return re.sub(r'<[^>]+>', '', str(text))


def has_html_tags(text):
    """Check if text contains HTML tags."""
    if pd.isna(text):
        return False
    return bool(re.search(r'<[^>]+>', str(text)))
