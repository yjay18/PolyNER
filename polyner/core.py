"""
Core module containing the main PolyNER class.
"""
from typing import Any, List, Optional

import pandas as pd
import spacy

from .emoji_handling import is_emoji
from .language_detection import detect_language
from .tokenization import normalize_token, tokenize_text


class PolyNER:
    """
    Main class for processing multilingual text with emojis and performing NER.
    """

    def __init__(
        self,
        ner_model: Optional[Any] = None,
        languages: Optional[List[str]] = None,
        normalize: bool = True,
    ):
        """
        Initialize the PolyNER processor.

        Args:
            ner_model: Optional custom NER model (spaCy model)
            languages: Optional list of languages to consider
            normalize: Whether to normalize tokens
        """
        self.normalize = normalize
        self.languages = languages

        # Load default spaCy model if none provided
        if ner_model is None:
            try:
                self.ner_model = spacy.load("en_core_web_sm")
            except OSError:
                # If model not found, download it
                import subprocess

                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.ner_model = spacy.load("en_core_web_sm")
        else:
            self.ner_model = ner_model

    def process(self, text: str) -> pd.DataFrame:
        """
        Process text and return structured data.

        Args:
            text: Input text that may contain multiple languages and emojis

        Returns:
            DataFrame with token-level information
        """
        # Extract emojis first
        # emojis = extract_emojis(text)

        # Tokenize the text
        tokens = tokenize_text(text)

        # Process each token
        results = []
        for token in tokens:
            # Check if token is an emoji
            emoji_flag = is_emoji(token)

            # Detect language (None for emojis)
            lang = None if emoji_flag else detect_language(token)

            # Normalize token if needed
            norm_token = token
            if self.normalize and not emoji_flag:
                norm_token = normalize_token(token)

            # Get entity label if applicable
            entity_label = None
            if not emoji_flag:
                # Use spaCy for entity recognition
                doc = self.ner_model(token)
                if doc.ents:
                    entity_label = doc.ents[0].label_

            # Add to results
            results.append(
                {
                    "token": token,
                    "language": lang,
                    "is_emoji": emoji_flag,
                    "norm_token": norm_token,
                    "entity_label": entity_label,
                }
            )

        # Convert to DataFrame
        return pd.DataFrame(results)

    def process_batch(self, texts: List[str]) -> List[pd.DataFrame]:
        """
        Process a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of DataFrames with token-level information
        """
        return [self.process(text) for text in texts]

    def load_custom_model(self, model_path: str) -> None:
        """
        Load a custom NER model.

        Args:
            model_path: Path to the custom model
        """
        self.ner_model = spacy.load(model_path)
