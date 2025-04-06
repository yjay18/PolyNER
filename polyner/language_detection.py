"""
Module for language detection functionality.
"""

import re
from typing import Dict, List, Optional

from langdetect import LangDetectException, detect

# Cache for language detection results to improve performance
_language_cache = {}


def detect_language(text: str, min_length: int = 3) -> Optional[str]:
    """
    Detect the language of a given text.

    Args:
        text: Input text
        min_length: Minimum text length to attempt language detection

    Returns:
        ISO language code or None if detection failed
    """
    # Clean the text
    text = text.strip()

    # Skip short texts
    if len(text) < min_length:
        return None

    # Check cache first
    if text in _language_cache:
        return _language_cache[text]

    try:
        # Detect language
        lang = detect(text)

        # Cache the result
        _language_cache[text] = lang

        return lang
    except LangDetectException:
        return None


def detect_language_with_confidence(text: str) -> Dict[str, float]:
    """
    Detect language with confidence scores.

    Args:
        text: Input text

    Returns:
        Dictionary mapping language codes to confidence scores
    """
    from langdetect import DetectorFactory

    # Set seed for reproducibility
    DetectorFactory.seed = 0

    try:
        detector = DetectorFactory.create()
        detector.append(text)

        # Get probabilities for each language
        return {lang.lang: lang.prob for lang in detector.get_probabilities()}
    except LangDetectException:
        return {}


def is_multilingual(text: str, threshold: float = 0.3) -> bool:
    """
    Determine if a text is multilingual.

    Args:
        text: Input text
        threshold: Confidence threshold for secondary language

    Returns:
        True if text appears to be multilingual
    """
    lang_probs = detect_language_with_confidence(text)

    # Sort by probability
    sorted_langs = sorted(lang_probs.items(), key=lambda x: x[1], reverse=True)

    # Check if we have at least two languages with significant probability
    return len(sorted_langs) >= 2 and sorted_langs[1][1] >= threshold
