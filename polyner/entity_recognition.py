"""
Module for named entity recognition.
"""

from typing import List, Dict, Any, Optional, Tuple
import spacy
import re

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
        entities.append({
            "text": ent.text,
            "start": ent.start_char,
            "end": ent.end_char,
            "label": ent.label_,
            "description": spacy.explain(ent.label_)
        })
    
    return entities

def recognize_entities_multilingual(text: str, models: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Recognize entities in multilingual text using language-specific models.
    
    Args:
        text: Input text
        models: Dictionary mapping language codes to spaCy models
        
    Returns:
        List of dictionaries with entity information
    """
    from .language_detection import detect_language
    from .tokenization import split_by_language
    
    # Split text by language
    language_segments = split_by_language(text)
    
    # Initialize default models if none provided
    if models is None:
        models = {}
        # Load English model by default
        try:
            models["en"] = spacy.load("en_core_web_sm")
        except OSError:
            # If model not found, download it
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            models["en"] = spacy.load("en_core_web_sm")
    
    # Process each language segment with the appropriate model
    all_entities = []
    offset = 0
    
    for lang, segments in language_segments.items():
        # Use the appropriate model for the language, or fall back to English
        model = models.get(lang, models.get("en"))
        
        for segment in segments:
            # Find the segment in the original text to get the correct offset
            segment_start = text.find(segment, offset)
            if segment_start == -1:
                continue
            
            # Update offset for next search
            offset = segment_start + len(segment)
            
            # Recognize entities in this segment
            doc = model(segment)
            
            for ent in doc.ents:
                all_entities.append({
                    "text": ent.text,
                    "start": segment_start + ent.start_char,
                    "end": segment_start + ent.end_char,
                    "label": ent.label_,
                    "language": lang,
                    "description": spacy.explain(ent.label_)
                })
    
    # Sort entities by their position in the text
    all_entities.sort(key=lambda e: e["start"])
    
    return all_entities

class DictionaryEntityRecognizer:
    """
    Entity recognizer based on dictionaries of terms.
    """
    
    def __init__(self):
        self.entity_dictionaries = {}
    
    def add_entity_dictionary(self, entity_type: str, terms: List[str], case_sensitive: bool = False) -> None:
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
            "case_sensitive": case_sensitive
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
                    original_term = text[pos:pos + len(term)]
                    
                    # Add to entities
                    entities.append({
                        "text": original_term,
                        "start": pos,
                        "end": pos + len(term),
                        "label": entity_type,
                        "description": f"Custom {entity_type}"
                    })
                    
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
                if (entity["start"] < existing["end"] and entity["end"] > existing["start"]):
                    # if they overlap - keep the longer one
                    if (entity["end"] - entity["start"]) > (existing["end"] - existing["start"]):
                        non_overlapping[i] = entity
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(entity)
        
        return non_overlapping
