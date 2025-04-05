"""
Tests for the entity recognition functionality.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import the package during testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from polyner.entity_recognition import (
    recognize_entities,
    recognize_entities_multilingual,
    DictionaryEntityRecognizer
)

class TestEntityRecognition(unittest.TestCase):
    """Test entity recognition functionality."""
    
    def test_recognize_entities(self):
        """Test basic entity recognition."""
        # Test with a simple text containing known entities
        text = "Apple Inc. is located in Cupertino, California. Tim Cook is the CEO."
        entities = recognize_entities(text)
        
        # Check that we found some entities
        self.assertGreater(len(entities), 0)
        
        # Check entity structure
        for entity in entities:
            self.assertIn("text", entity)
            self.assertIn("start", entity)
            self.assertIn("end", entity)
            self.assertIn("label", entity)
            self.assertIn("description", entity)
        
        # Check that common entities were detected
        entity_texts = [entity["text"] for entity in entities]
        common_entities = ["Apple", "Cupertino", "California", "Tim Cook"]
        
        # At least one of these should be detected
        self.assertTrue(any(entity in entity_texts for entity in common_entities))
    
    def test_recognize_entities_empty_text(self):
        """Test entity recognition with empty text."""
        entities = recognize_entities("")
        self.assertEqual(len(entities), 0)
    
    def test_dictionary_entity_recognizer(self):
        """Test the dictionary-based entity recognizer."""
        # Create a recognizer and add dictionaries
        recognizer = DictionaryEntityRecognizer()
        
        # Add product dictionary
        recognizer.add_entity_dictionary(
            "PRODUCT",
            ["iPhone", "MacBook Pro", "iPad"],
            case_sensitive=True
        )
        
        # Add company dictionary
        recognizer.add_entity_dictionary(
            "COMPANY",
            ["Apple", "Google", "Microsoft"],
            case_sensitive=True
        )
        
        # Test recognition
        text = "Apple released a new iPhone model. Microsoft announced a new Surface laptop."
        entities = recognizer.recognize_entities(text)
        
        # Check that we found entities
        self.assertGreater(len(entities), 0)
        
        # Check that specific entities were detected
        entity_data = {(entity["text"], entity["label"]) for entity in entities}
        
        self.assertIn(("Apple", "COMPANY"), entity_data)
        self.assertIn(("iPhone", "PRODUCT"), entity_data)
        self.assertIn(("Microsoft", "COMPANY"), entity_data)
        
        # Test case sensitivity
        recognizer = DictionaryEntityRecognizer()
        recognizer.add_entity_dictionary(
            "COMPANY",
            ["Apple", "Google", "Microsoft"],
            case_sensitive=True
        )
        
        # With case_sensitive=True, "apple" should not match "Apple"
        text = "apple is not the same as Apple."
        entities = recognizer.recognize_entities(text)
        
        entity_texts = [entity["text"] for entity in entities]
        self.assertIn("Apple", entity_texts)
        self.assertNotIn("apple", entity_texts)
        
        # With case_sensitive=False, "apple" should match "Apple"
        recognizer = DictionaryEntityRecognizer()
        recognizer.add_entity_dictionary(
            "COMPANY",
            ["Apple", "Google", "Microsoft"],
            case_sensitive=False
        )
        
        entities = recognizer.recognize_entities(text)
        entity_texts = [entity["text"] for entity in entities]
        self.assertIn("Apple", entity_texts)
        self.assertIn("apple", entity_texts)
    
    def test_overlapping_entities(self):
        """Test handling of overlapping entities."""
        recognizer = DictionaryEntityRecognizer()
        
        # Add dictionaries with potentially overlapping terms
        recognizer.add_entity_dictionary(
            "LOCATION",
            ["New York", "York", "New York City"],
            case_sensitive=True
        )
        
        # "New York City" should be preferred over "New York" due to length
        text = "I love New York City."
        entities = recognizer.recognize_entities(text)
        
        # Check that we only got one entity
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["text"], "New York City")
        self.assertEqual(entities[0]["label"], "LOCATION")
    
    def test_recognize_entities_multilingual(self):
        """Test multilingual entity recognition."""
        # This test is more complex and might require mocking language models
        # For now, we'll check that the function exists and takes the right parameters
        self.assertTrue(callable(recognize_entities_multilingual))
        
        # For a simple test, we can use English text
        # This mainly tests that the function runs without error
        text = "Apple Inc. is headquartered in Cupertino."
        try:
            entities = recognize_entities_multilingual(text)
            # Check that we got some entities
            self.assertIsInstance(entities, list)
        except Exception as e:
            self.fail(f"recognize_entities_multilingual raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()