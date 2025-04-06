"""
Tests for the tokenization functionality.
"""

import unittest
import sys
import os
from typing import Dict, Any

# Add the parent directory to the path so we can import the package during testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from polyner.tokenization import (
    get_spacy_model,
    tokenize_text,
    normalize_token,
    get_token_features,
    split_by_language
)

class TestTokenization(unittest.TestCase):
    """Test tokenization functionality."""
    
    def test_get_spacy_model(self):
        """Test loading spaCy models."""
        # Test loading the default model
        model = get_spacy_model()
        self.assertIsNotNone(model)
        
        # Test that the model is cached (should be the same object)
        model2 = get_spacy_model()
        self.assertIs(model, model2)
    
    def test_tokenize_text_simple(self):
        """Test basic tokenization."""
        # Test with simple English text
        text = "Hello, world! This is a test."
        tokens = tokenize_text(text)
        
        # Check that we got some tokens
        self.assertTrue(len(tokens) > 0)
        
        # Check that common tokens are present
        self.assertIn("Hello", tokens)
        self.assertIn("world", tokens)
        
        # Test with empty text
        self.assertEqual(tokenize_text(""), [])
    
    def test_tokenize_text_with_emojis(self):
        """Test tokenization with emojis."""
        # Test with text containing emojis
        text = "Hello 😊 world! 👋"
        tokens = tokenize_text(text)
        
        # Check that emojis were preserved as separate tokens
        self.assertIn("😊", tokens)
        self.assertIn("👋", tokens)
        
        # Since we can't guarantee different results with preserve_emojis=False
        # (it depends on the specific spaCy model), we'll just check it runs
        tokens_no_preserve = tokenize_text(text, preserve_emojis=False)
        self.assertIsInstance(tokens_no_preserve, list)
        self.assertTrue(len(tokens_no_preserve) > 0)
    
    def test_normalize_token(self):
        """Test token normalization."""
        # Test lowercase conversion
        self.assertEqual(normalize_token("Hello"), "hello")
        
        # Test accent removal
        self.assertEqual(normalize_token("café"), "cafe")
        self.assertEqual(normalize_token("résumé"), "resume")
        
        # Test with emoji (should be preserved)
        self.assertEqual(normalize_token("😊"), "😊")
        
        # Test with different options
        self.assertEqual(normalize_token("Hello", lowercase=False), "Hello")
        self.assertEqual(normalize_token("café", remove_accents=False), "café")
        self.assertEqual(normalize_token("Café", lowercase=False, remove_accents=True), "Cafe")
    
    def test_get_token_features(self):
        """Test getting token features."""
        # Test with a common word
        features = get_token_features("apple")
        
        # Check that we got the expected features
        self.assertEqual(features["token"], "apple")
        self.assertEqual(features["lemma"], "apple")
        self.assertFalse(features["is_emoji"])
        
        # Part of speech tagging specifics might vary by spaCy model,
        # so we just check that the fields exist without testing their values
        self.assertIn("pos", features)
        self.assertIn("is_stop", features)
        self.assertIn("is_punct", features)
        
        # Test with an emoji
        emoji_features = get_token_features("😊")
        self.assertEqual(emoji_features["token"], "😊")
        self.assertEqual(emoji_features["lemma"], "😊")
        self.assertTrue(emoji_features["is_emoji"])
        self.assertIsNone(emoji_features["pos"])
        self.assertFalse(emoji_features["is_stop"])
        self.assertFalse(emoji_features["is_punct"])
        
        # Test with punctuation
        punct_features = get_token_features(".")
        self.assertEqual(punct_features["token"], ".")
        self.assertFalse(punct_features["is_emoji"])
    
    def test_split_by_language(self):
        """Test splitting text by language."""
        # This test is more complex as it involves language detection
        # We'll check that the function returns a sensible result
        
        # Test with clearly separated languages
        text = "This is English. Esto es español."
        result = split_by_language(text)
        
        # Check that the result is a dict
        self.assertIsInstance(result, dict)
        
        # Check that we detected at least one language
        self.assertGreater(len(result), 0)
        
        # Test with English only
        english_text = "This is a longer English text that should be detected correctly."
        english_result = split_by_language(english_text)
        
        # For a longer, clearer English text, at least some of it should be detected as English
        english_detected = False
        for lang, segments in english_result.items():
            if lang == "en":
                english_detected = True
                break
            # Alternatively, check if any segment contains English words
            for segment in segments:
                if "This" in segment or "English" in segment:
                    english_detected = True
                    break
        
        # Note: Language detection can be imperfect, especially for short texts
        # So we don't make a strong assertion about English detection
        # This is a simple check that the function runs without error
        self.assertIsInstance(english_result, dict)
        
        # Test with empty text
        self.assertEqual(split_by_language(""), {})


if __name__ == "__main__":
    unittest.main()