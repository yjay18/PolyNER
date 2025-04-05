"""
Tests for the core PolyNER functionality.
"""

import unittest
import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import the package during testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from polyner import PolyNER

class TestCore(unittest.TestCase):
    """Test core functionality of the PolyNER class."""
    
    def setUp(self):
        """Set up test environment."""
        self.processor = PolyNER()
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        # Default initialization
        processor = PolyNER()
        self.assertTrue(processor.normalize)
        self.assertIsNone(processor.languages)
        
        # Custom initialization
        processor = PolyNER(normalize=False, languages=["en", "fr"])
        self.assertFalse(processor.normalize)
        self.assertEqual(processor.languages, ["en", "fr"])
    
    def test_process_empty_text(self):
        """Test processing empty text."""
        result = self.processor.process("")
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertTrue(result.empty)
    
    def test_process_simple_text(self):
        """Test processing simple text."""
        text = "Hello world."
        result = self.processor.process(text)
        
        # Check that the DataFrame has the expected structure
        self.assertIn("token", result.columns)
        self.assertIn("language", result.columns)
        self.assertIn("is_emoji", result.columns)
        self.assertIn("norm_token", result.columns)
        self.assertIn("entity_label", result.columns)
        
        # Check that tokens were properly extracted
        tokens = result["token"].tolist()
        self.assertIn("Hello", tokens)
        self.assertIn("world", tokens)
        
        # Check normalization
        norm_tokens = result["norm_token"].tolist()
        self.assertIn("hello", norm_tokens)
        self.assertIn("world", norm_tokens)
    
    def test_process_text_with_entities(self):
        """Test processing text with named entities."""
        text = "Apple Inc. is headquartered in Cupertino, California."
        result = self.processor.process(text)
        
        # Check that some entities were detected
        self.assertTrue(result["entity_label"].notna().any())
        
        # Common entities that should be detected
        entities = result[result["entity_label"].notna()]
        entity_texts = entities["token"].tolist()
        
        # At least one of these should be detected as an entity
        expected_entities = ["Apple", "Cupertino", "California"]
        self.assertTrue(any(entity in entity_texts for entity in expected_entities))
    
    def test_process_with_emojis(self):
        """Test processing text with emojis."""
        text = "Hello 😊 world! 👋"
        result = self.processor.process(text)
        
        # Check that emojis were detected
        emojis = result[result["is_emoji"] == True]
        self.assertFalse(emojis.empty)
        
        emoji_tokens = emojis["token"].tolist()
        self.assertIn("😊", emoji_tokens)
        self.assertIn("👋", emoji_tokens)
    
    def test_process_batch(self):
        """Test batch processing of multiple texts."""
        texts = [
            "Hello world.",
            "Apple Inc. is in California.",
            "Hello 😊 emoji."
        ]
        
        results = self.processor.process_batch(texts)
        
        # Check results
        self.assertEqual(len(results), 3)
        self.assertTrue(all(isinstance(result, pd.DataFrame) for result in results))
        
        # First text should have no entities or emojis
        self.assertFalse(results[0]["entity_label"].notna().any())
        self.assertFalse(results[0]["is_emoji"].any())
        
        # Second text should have entities but no emojis
        self.assertTrue(results[1]["entity_label"].notna().any())
        self.assertFalse(results[1]["is_emoji"].any())
        
        # Third text should have emojis
        self.assertTrue(results[2]["is_emoji"].any())
        
    def test_load_custom_model(self):
        """Test loading a custom model."""
        # This is a more integration-type test and might be difficult to test in isolation
        # A more comprehensive test would involve creating a mock model
        # For now, we'll just check that the method exists and takes the right parameters
        processor = PolyNER()
        self.assertTrue(callable(getattr(processor, "load_custom_model", None)))


if __name__ == "__main__":
    unittest.main()