"""
Core module containing the main PolyNER class.
"""
from typing import Any, Dict, List, Optional

import pandas as pd
import spacy

from .emoji_handling import is_emoji
from .language_detection import detect_language
from .tokenization import normalize_token, tokenize_text
from .entity_recognition import recognize_entities, recognize_entities_multilingual


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
            
        # Initialize language models dictionary for multilingual processing
        self.language_models = {}
        if self.languages:
            for lang in self.languages:
                try:
                    model_name = f"{lang}_core_web_sm"
                    self.language_models[lang] = spacy.load(model_name)
                except OSError:
                    # Skip if not available
                    pass
        
        # Always ensure English is available
        if "en" not in self.language_models:
            self.language_models["en"] = self.ner_model

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
    
    def process_multi(
        self, 
        text: str, 
        model_name: str = "Babelscape/wikineural-multilingual-ner"
    ) -> pd.DataFrame:
        """
        Process multilingual text using a specialized model for NER.
        
        Args:
            text: Input text that may contain multiple languages and emojis
            model_name: Name or path of the model to use (default: "Babelscape/wikineural-multilingual-ner")
                      Can be a Hugging Face model name like "xlm-roberta-base-finetuned-panx-all"
                      or a spaCy model path or name like "en_core_web_sm"
                      
        Returns:
            DataFrame with token-level information
        """
        if not text:
            return pd.DataFrame()
            
        # Tokenize the text
        tokens = tokenize_text(text)
        
        # Get entity predictions using specified model
        entities = recognize_entities_multilingual(
            text, 
            model_name=model_name, 
            models=self.language_models
        )
        
        # Process each token and check if it's part of an entity
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
                
            # Get token position in the text (approximate)
            token_pos = text.find(token)
            
            # Get entity label if the token is part of an entity
            entity_label = None
            entity_score = None
            
            # Only check entities if this isn't an emoji
            if not emoji_flag and token_pos >= 0:
                for entity in entities:
                    entity_start = entity["start"]
                    entity_end = entity["end"]
                    
                    # Check if token position falls within this entity
                    if token_pos >= entity_start and token_pos + len(token) <= entity_end:
                        entity_label = entity["label"]
                        # Include confidence score if available
                        if "score" in entity:
                            entity_score = entity["score"]
                        break
            
            # Add to results with optional score
            result_dict = {
                "token": token,
                "language": lang,
                "is_emoji": emoji_flag,
                "norm_token": norm_token,
                "entity_label": entity_label,
            }
            
            # Add score if available
            if entity_score is not None:
                result_dict["entity_score"] = entity_score
                
            results.append(result_dict)
            
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
        
    def process_batch_multi(
        self, 
        texts: List[str], 
        model_name: str = "Babelscape/wikineural-multilingual-ner"
    ) -> List[pd.DataFrame]:
        """
        Process a batch of texts using multilingual model.

        Args:
            texts: List of input texts
            model_name: Name or path of the model to use

        Returns:
            List of DataFrames with token-level information
        """
        return [self.process_multi(text, model_name) for text in texts]

    def load_custom_model(self, model_path: str) -> None:
        """
        Load a custom spaCy NER model.

        Args:
            model_path: Path to the custom model
        """
        self.ner_model = spacy.load(model_path)
        
    def add_language_model(self, language: str, model_name: Optional[str] = None) -> bool:
        """
        Add a language model for multilingual processing.

        Args:
            language: Language code (e.g., 'fr', 'es', 'de')
            model_name: Optional specific model name, otherwise uses {language}_core_web_sm

        Returns:
            True if successfully added, False otherwise
        """
        if language in self.language_models:
            return True  # Already loaded
            
        if model_name is None:
            model_name = f"{language}_core_web_sm"
            
        try:
            # Try to load the model
            self.language_models[language] = spacy.load(model_name)
            return True
        except OSError:
            # Try to download the model
            try:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", model_name])
                self.language_models[language] = spacy.load(model_name)
                return True
            except:
                # Failed to download or load
                return False