"""
Advanced tests for the PolyNER library.
"""

import sys
import os
import pandas as pd

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from polyner import PolyNER
from polyner.entity_recognition import DictionaryEntityRecognizer
from polyner.utils import filter_by_entity, get_entity_distribution

def test_custom_dictionary():
    """Test using a custom dictionary for entity recognition."""
    # Create a dictionary recognizer
    dict_recognizer = DictionaryEntityRecognizer()
    
    # Add some custom entity dictionaries
    dict_recognizer.add_entity_dictionary(
        "PRODUCT",
        ["iPhone", "MacBook Pro", "iPad", "Apple Watch", "AirPods"],
        case_sensitive=True
    )
    
    dict_recognizer.add_entity_dictionary(
        "PROGRAMMING_LANGUAGE",
        ["Python", "JavaScript", "Java", "C++", "Ruby", "Go"],
        case_sensitive=True
    )
    
    # Process a text with the dictionary recognizer
    text = "I use Python for data science and JavaScript for web development. My MacBook Pro runs both perfectly."
    entities = dict_recognizer.recognize_entities(text)
    
    print("Custom dictionary entity recognition:")
    for entity in entities:
        print(f"{entity['text']} - {entity['label']}")
    print("\n")
    
    # Check that we detected the expected entities
    entity_texts = [entity["text"] for entity in entities]
    assert "Python" in entity_texts, "Python not detected as entity"
    assert "JavaScript" in entity_texts, "JavaScript not detected as entity"
    assert "MacBook Pro" in entity_texts, "MacBook Pro not detected as entity"

def test_batch_processing():
    """Test batch processing functionality."""
    # Initialize the processor
    processor = PolyNER()
    
    # Create a batch of texts
    texts = [
        "Apple Inc. is headquartered in Cupertino, California.",
        "Google's main office is in Mountain View.",
        "Amazon was founded by Jeff Bezos in Seattle.",
        "Microsoft is based in Redmond, Washington."
    ]
    
    # Process the batch
    results = processor.process_batch(texts)
    
    print("Batch processing results:")
    for i, result in enumerate(results):
        print(f"Text {i+1}:")
        print(result[result["entity_label"].notna()])
        print("\n")
    
    # Check that we have the expected number of results
    assert len(results) == len(texts), "Number of results doesn't match number of input texts"
    
    # Check that each result is a DataFrame
    assert all(isinstance(result, pd.DataFrame) for result in results), "Not all results are DataFrames"

def test_language_specific_processing():
    """Test processing text in different languages."""
    # Initialize the processor
    processor = PolyNER()
    
    # Process texts in different languages
    texts = {
        "en": "The United Nations is headquartered in New York City.",
        "es": "Madrid es la capital de España.",
        "fr": "Paris est la capitale de la France.",
        "de": "Berlin ist die Hauptstadt von Deutschland.",
        "zh": "北京是中国的首都。",
        "ja": "東京は日本の首都です。",
        "ru": "Москва - столица России."
    }
    
    print("Language-specific processing:")
    for lang_code, text in texts.items():
        result = processor.process(text)
        detected_langs = result["language"].unique()
        
        print(f"Text in {lang_code}:")
        print(f"Detected languages: {detected_langs}")
        print(result)
        print("\n")
        
        # Check that we detected the language correctly (might not be exact due to short text)
        # This is a loose check since language detection isn't perfect
        if lang_code not in ["zh", "ja"]:  # These might be detected differently
            assert any(lang.startswith(lang_code) for lang in detected_langs if lang), f"Language {lang_code} not detected"

if __name__ == "__main__":
    print("Running advanced tests for PolyNER...\n")
    
    test_custom_dictionary()
    test_batch_processing()
    test_language_specific_processing()
    
    print("All advanced tests completed successfully!")
