"""
Basic tests for the PolyNER library.
"""

import sys
import os
import pandas as pd

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from polyner import PolyNER
from polyner.utils import filter_emojis, filter_by_language, get_language_distribution

def test_basic_functionality():
    """Test basic functionality of the PolyNER library."""
    # Initialize the processor
    processor = PolyNER()
    
    # Process a simple English text
    text = "Apple Inc. is headquartered in Cupertino, California."
    result = processor.process(text)
    
    print("Basic English text processing:")
    print(result)
    print("\n")
    
    # Check that we have the expected columns
    assert all(col in result.columns for col in ["token", "language", "is_emoji", "norm_token", "entity_label"])
    
    # Check that we detected some entities
    assert result["entity_label"].notna().any(), "No entities detected"
    
    # Check that the language was detected correctly
    assert "en" in result["language"].values, "English language not detected"

def test_multilingual_with_emojis():
    """Test processing multilingual text with emojis."""
    # Initialize the processor
    processor = PolyNER()
    
    # Process a text with mixed languages and emojis
    text = "Hello world! 你好世界! 😊 Bonjour le monde! 👋"
    result = processor.process(text)
    
    print("Multilingual text with emojis processing:")
    print(result)
    print("\n")
    
    # Check that we detected emojis
    emojis = filter_emojis(result)
    assert not emojis.empty, "No emojis detected"
    assert "😊" in emojis["token"].values, "Specific emoji not detected"
    assert "👋" in emojis["token"].values, "Specific emoji not detected"
    
    # Check that we detected multiple languages
    languages = get_language_distribution(result)
    print("Detected languages:", languages)
    assert len(languages) >= 2, "Not enough languages detected"

def test_entity_recognition():
    """Test entity recognition functionality."""
    # Initialize the processor
    processor = PolyNER()
    
    # Process a text with named entities
    text = "Microsoft was founded by Bill Gates. The company is headquartered in Redmond, Washington."
    result = processor.process(text)
    
    print("Entity recognition:")
    print(result[result["entity_label"].notna()])
    print("\n")
    
    # Check that we detected the expected entities
    entities = result[result["entity_label"].notna()]
    entity_texts = entities["token"].tolist()
    
    assert any("Microsoft" in text for text in entity_texts), "Microsoft not detected as entity"
    assert any("Bill" in text for text in entity_texts), "Bill Gates not detected as entity"
    assert any("Redmond" in text for text in entity_texts), "Redmond not detected as entity"
    assert any("Washington" in text for text in entity_texts), "Washington not detected as entity"

if __name__ == "__main__":
    print("Running basic tests for PolyNER...\n")
    
    test_basic_functionality()
    test_multilingual_with_emojis()
    test_entity_recognition()
    
    print("All tests completed successfully!")
