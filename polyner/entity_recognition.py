"""
Module for named entity recognition.
"""

import re
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import spacy


def recognize_entities(text: str, model: Any = None) -> List[Dict[str, Any]]:
    """
    Recognize named entities in text using spaCy.

    Args:
        text: Input text
        model: Optional spaCy model

    Returns:
        List of dictionaries with entity information
    """
    # Load default model if none provided
    if model is None:
        try:
            model = spacy.load("en_core_web_sm")
        except OSError:
            # If model not found, download it
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            model = spacy.load("en_core_web_sm")

    # Process the text
    doc = model(text)

    # Extract entities
    entities = []
    for ent in doc.ents:
        entities.append(
            {
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
                "description": spacy.explain(ent.label_),
            }
        )

    return entities


def recognize_entities_multilingual(
    text: str,
    model_name: str = "Babelscape/wikineural-multilingual-ner",
    models: Dict[str, Any] = None,
    confidence_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Recognize entities in multilingual text using context-aware entity recognition.

    Args:
        text: Input text
        model_name: Name of the model to use (default: "Babelscape/wikineural-multilingual-ner")
        models: Dictionary mapping language codes to spaCy models (for fallback)
        confidence_threshold: Minimum confidence score for entities (0.0 to 1.0)

    Returns:
        List of dictionaries with entity information
    """
    import string

    # Skip empty text
    if not text:
        return []

    # Step 1: Try to use Hugging Face transformer model
    try:
        # Import inside try block to avoid dependency issues
        from transformers import pipeline

        # Initialize the NER pipeline with the specified model
        ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="simple")

        # Get entity predictions with context
        predictions = ner_pipeline(text)

        # Process and filter predictions
        entities = []
        for pred in predictions:
            # Filter out low confidence predictions and punctuation
            if (
                pred.get("score", 1.0) > confidence_threshold
                and not pred["word"].strip() in string.punctuation
            ):
                entities.append(
                    {
                        "text": pred["word"],
                        "start": pred["start"],
                        "end": pred["end"],
                        "label": pred["entity_group"],
                        "score": pred.get("score", 1.0),
                        "source": "huggingface",
                    }
                )

        # If we found entities, return them
        if entities:
            return entities

    except Exception as e:
        print(f"Error using Hugging Face model: {str(e)}")

    # Step 2: If model_name looks like a spaCy model, try to use it directly
    if (
        model_name.endswith(".spacy")
        or model_name.find("_core_") > 0
        or os.path.exists(model_name)
    ):
        try:
            import spacy

            nlp = spacy.load(model_name)
            doc = nlp(text)

            entities = []
            for ent in doc.ents:
                if not ent.text.strip() in string.punctuation:
                    entities.append(
                        {
                            "text": ent.text,
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "label": ent.label_,
                            "source": "spacy",
                        }
                    )

            # If we found entities, return them
            if entities:
                return entities

        except Exception as e:
            print(f"Error using spaCy model {model_name}: {str(e)}")

    # Step 3: Try language-specific processing with spaCy models
    if models is not None:
        try:
            # Use language detection and split text by language
            from .language_detection import detect_language
            import re

            # Split into sentences for better language detection
            sentences = re.split(r"(?<=[.!?])\s+", text)

            # Process each sentence with language detection
            all_entities = []
            current_pos = 0

            for sentence in sentences:
                if not sentence.strip():  # Skip empty sentences
                    current_pos += len(sentence) + 1
                    continue

                # Detect language of the sentence
                lang = detect_language(sentence)

                # Select the appropriate model for this language
                if lang in models:
                    model = models[lang]
                else:
                    # Fall back to English or the first available model
                    model = models.get("en", next(iter(models.values())))

                # Process the sentence
                doc = model(sentence)

                # Extract entities with correct position in original text
                for ent in doc.ents:
                    if not ent.text.strip() in string.punctuation:
                        all_entities.append(
                            {
                                "text": ent.text,
                                "start": current_pos + ent.start_char,
                                "end": current_pos + ent.end_char,
                                "label": ent.label_,
                                "language": lang,
                                "source": "spacy_multilingual",
                            }
                        )

                # Update position for next sentence
                current_pos += len(sentence) + 1  # +1 for space

            # Sort entities by position
            all_entities.sort(key=lambda e: e["start"])

            # If we found entities, return them
            if all_entities:
                return all_entities

        except Exception as e:
            print(f"Error in multilingual spaCy processing: {str(e)}")

    # Step 4: Last resort - try with English spaCy model
    try:
        import spacy

        model = spacy.load("en_core_web_sm")
        doc = model(text)

        entities = []
        for ent in doc.ents:
            if not ent.text.strip() in string.punctuation:
                entities.append(
                    {
                        "text": ent.text,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "label": ent.label_,
                        "source": "spacy_fallback",
                    }
                )

        return entities

    except Exception as e:
        print(f"Error in final fallback: {str(e)}")
        return []  # Return empty list if all attempts fail


class DictionaryEntityRecognizer:
    """
    Entity recognizer based on dictionaries of terms.
    """

    def __init__(self):
        self.entity_dictionaries = {}

    def add_entity_dictionary(
        self, entity_type: str, terms: List[str], case_sensitive: bool = False
    ) -> None:
        """
        Add a dictionary of terms for a specific entity type.

        Args:
            entity_type: Type of entity (e.g., "PERSON", "ORG")
            terms: List of terms to recognize
            case_sensitive: Whether matching should be case-sensitive
        """
        # Sort terms by length (longest first) to ensure we match the longest possible term
        sorted_terms = sorted(terms, key=len, reverse=True)

        self.entity_dictionaries[entity_type] = {
            "terms": sorted_terms,
            "case_sensitive": case_sensitive,
        }

    def recognize_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Recognize entities in text using the dictionaries.

        Args:
            text: Input text

        Returns:
            List of dictionaries with entity information
        """
        entities = []

        # Process each entity type
        for entity_type, dictionary in self.entity_dictionaries.items():
            terms = dictionary["terms"]
            case_sensitive = dictionary["case_sensitive"]

            # Prepare text for matching
            match_text = text if case_sensitive else text.lower()

            # Find all occurrences of each term
            for term in terms:
                search_term = term if case_sensitive else term.lower()

                # Find all occurrences
                start = 0
                while start < len(match_text):
                    pos = match_text.find(search_term, start)
                    if pos == -1:
                        break

                    # Get the actual text from the original
                    original_term = text[pos : pos + len(term)]

                    # Add to entities
                    entities.append(
                        {
                            "text": original_term,
                            "start": pos,
                            "end": pos + len(term),
                            "label": entity_type,
                            "description": f"Custom {entity_type}",
                        }
                    )

                    # Move past this occurrence
                    start = pos + 1

        # Sort entities by their position in the text
        entities.sort(key=lambda e: e["start"])

        # Remove overlapping entities (keep the longest one)
        non_overlapping = []
        for entity in entities:
            # Check if this entity overlaps with any in the non_overlapping list
            overlaps = False
            for i, existing in enumerate(non_overlapping):
                if (
                    entity["start"] < existing["end"]
                    and entity["end"] > existing["start"]
                ):
                    # if they overlap - keep the longer one
                    if (entity["end"] - entity["start"]) > (
                        existing["end"] - existing["start"]
                    ):
                        non_overlapping[i] = entity
                    overlaps = True
                    break

            if not overlaps:
                non_overlapping.append(entity)

        return non_overlapping
