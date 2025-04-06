"""
Module for text tokenization and normalization.
"""

import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional

import spacy
from spacy.tokens import Doc

from .emoji_handling import extract_emojis, is_emoji

# Global spaCy model cache
_nlp_models = {}


def get_spacy_model(model_name: str = "en_core_web_sm") -> Any:
    """
    Get or load a spaCy model.

    Args:
        model_name: Name of the spaCy model

    Returns:
        Loaded spaCy model
    """
    if model_name not in _nlp_models:
        try:
            _nlp_models[model_name] = spacy.load(model_name)
        except OSError:
            # If model not found, download it
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", model_name])
            _nlp_models[model_name] = spacy.load(model_name)

    return _nlp_models[model_name]


def tokenize_text(text: str, preserve_emojis: bool = True) -> List[str]:
    """
    Tokenize text using spaCy while preserving emojis.

    Args:
        text: Input text
        preserve_emojis: Whether to preserve emojis as separate tokens

    Returns:
        List of tokens
    """
    if not text:
        return []

    # Get spaCy model
    nlp = get_spacy_model()

    # Extract emojis first
    emojis = extract_emojis(text) if preserve_emojis else []

    # Replace emojis with placeholders if we want to preserve them
    if preserve_emojis and emojis:
        # Create unique placeholders for each emoji
        placeholders = {}
        text_with_placeholders = text

        for i, emoji_char in enumerate(emojis):
            placeholder = f"EMOJI_PLACEHOLDER_{i}"
            placeholders[placeholder] = emoji_char
            text_with_placeholders = text_with_placeholders.replace(
                emoji_char, f" {placeholder} "
            )

        # Process with spaCy
        doc = nlp(text_with_placeholders)
        tokens = [token.text for token in doc]

        # Replace placeholders back with emojis
        for i, token in enumerate(tokens):
            if token in placeholders:
                tokens[i] = placeholders[token]
    else:
        # If we don't need to preserve emojis, just tokenize normally
        doc = nlp(text)
        tokens = [token.text for token in doc]

    # Filter out empty tokens
    return [token for token in tokens if token.strip() or is_emoji(token)]


def normalize_token(
    token: str, lowercase: bool = True, remove_accents: bool = True
) -> str:
    """
    Normalize a token by lowercasing and removing accents.

    Args:
        token: Input token
        lowercase: Whether to convert to lowercase
        remove_accents: Whether to remove accents

    Returns:
        Normalized token
    """
    # Skip normalization for emojis
    if is_emoji(token):
        return token

    # Lowercase if requested
    if lowercase:
        token = token.lower()

    # Remove accents if requested
    if remove_accents:
        token = "".join(
            c
            for c in unicodedata.normalize("NFD", token)
            if unicodedata.category(c) != "Mn"
        )

    return token


def get_token_features(token: str, nlp_model: Optional[Any] = None) -> Dict[str, Any]:
    """
    Get linguistic features for a token using spaCy.

    Args:
        token: Input token
        nlp_model: Optional spaCy model

    Returns:
        Dictionary of token features
    """
    # Skip for emojis
    if is_emoji(token):
        return {
            "token": token,
            "pos": None,
            "lemma": token,
            "is_stop": False,
            "is_punct": False,
            "is_emoji": True,
        }

    # Load spaCy model if not provided
    if nlp_model is None:
        nlp_model = get_spacy_model()

    # Process the token
    doc = nlp_model(token)
    spacy_token = doc[0] if len(doc) > 0 else None

    if spacy_token:
        return {
            "token": token,
            "pos": spacy_token.pos_,
            "lemma": spacy_token.lemma_,
            "is_stop": spacy_token.is_stop,
            "is_punct": spacy_token.is_punct,
            "is_emoji": False,
        }
    else:
        return {
            "token": token,
            "pos": None,
            "lemma": token,
            "is_stop": False,
            "is_punct": False,
            "is_emoji": False,
        }


def split_by_language(text: str) -> Dict[str, List[str]]:
    """
    Split text into segments by detected language.

    Args:
        text: Input text

    Returns:
        Dictionary mapping language codes to lists of text segments
    """
    from .language_detection import detect_language

    # Tokenize the text
    tokens = tokenize_text(text)

    # Group tokens by language
    language_segments = {}
    current_lang = None
    current_segment = []

    for token in tokens:
        # Skip emojis for language detection
        if is_emoji(token):
            if current_segment:
                current_segment.append(token)
            continue

        # Detect language of the token
        token_lang = detect_language(token)

        # If language changes or couldn't be detected, start a new segment
        if token_lang != current_lang:
            # Save the current segment if it exists
            if current_segment and current_lang:
                if current_lang not in language_segments:
                    language_segments[current_lang] = []
                language_segments[current_lang].append(" ".join(current_segment))
                current_segment = []

            current_lang = token_lang

        # Add token to current segment
        current_segment.append(token)

    # Add the last segment
    if current_segment and current_lang:
        if current_lang not in language_segments:
            language_segments[current_lang] = []
        language_segments[current_lang].append(" ".join(current_segment))

    return language_segments
