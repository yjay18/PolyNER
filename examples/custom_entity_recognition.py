"""
Example script demonstrating how to use PolyNER with custom dictionaries and models.
"""

import sys
import os
import pandas as pd

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from polyner import PolyNER
from polyner.entity_recognition import DictionaryEntityRecognizer
from polyner.utils import filter_by_entity, get_entity_distribution

def custom_dictionary_example():
    """
    Example of using custom dictionaries for specialized entity recognition.
    """
    print("=" * 80)
    print("CUSTOM DICTIONARY EXAMPLE")
    print("=" * 80)
    
    # Create a dictionary recognizer
    dict_recognizer = DictionaryEntityRecognizer()
    
    # Add custom entity dictionaries for tech products
    dict_recognizer.add_entity_dictionary(
        "SMARTPHONE",
        ["iPhone 13", "iPhone 12", "Galaxy S21", "Pixel 6", "OnePlus 9"],
        case_sensitive=True
    )
    
    dict_recognizer.add_entity_dictionary(
        "LAPTOP",
        ["MacBook Pro", "MacBook Air", "Dell XPS", "ThinkPad X1", "Surface Laptop"],
        case_sensitive=True
    )
    
    dict_recognizer.add_entity_dictionary(
        "TECH_COMPANY",
        ["Apple", "Google", "Microsoft", "Amazon", "Meta", "Samsung", "Tesla"],
        case_sensitive=True
    )
    
    # Sample text about tech products
    tech_text = """
    I recently upgraded from my old iPhone 12 to the new iPhone 13 Pro Max, and the camera is amazing!
    My friend uses a Galaxy S21 and loves it too. For work, I use a MacBook Pro, but I'm considering
    switching to a Dell XPS or ThinkPad X1 for better compatibility with our office software.
    
    Apple and Google are releasing new AI features, while Microsoft is focusing on cloud services.
    Meta (formerly Facebook) is investing heavily in VR technology.
    """
    
    # Recognize entities using the custom dictionaries
    entities = dict_recognizer.recognize_entities(tech_text)
    
    # Display the results
    print("\nDetected tech entities:")
    for entity in entities:
        print(f"  {entity['text']} - {entity['label']}")
    
    # Count entities by type
    entity_counts = {}
    for entity in entities:
        label = entity["label"]
        if label not in entity_counts:
            entity_counts[label] = 0
        entity_counts[label] += 1
    
    print("\nEntity counts by type:")
    for label, count in entity_counts.items():
        print(f"  {label}: {count}")
    
    print("=" * 80)

def combined_approach_example():
    """
    Example of combining standard NER with custom dictionaries.
    """
    print("=" * 80)
    print("COMBINED APPROACH EXAMPLE")
    print("=" * 80)
    
    # Initialize the standard PolyNER processor
    processor = PolyNER()
    
    # Create a custom dictionary recognizer for medical terms
    medical_recognizer = DictionaryEntityRecognizer()
    
    # Add medical dictionaries
    medical_recognizer.add_entity_dictionary(
        "MEDICATION",
        ["Aspirin", "Ibuprofen", "Paracetamol", "Amoxicillin", "Lipitor", "Prozac"],
        case_sensitive=True
    )
    
    medical_recognizer.add_entity_dictionary(
        "CONDITION",
        ["Hypertension", "Diabetes", "Asthma", "Arthritis", "Depression"],
        case_sensitive=True
    )
    
    # Sample medical text
    medical_text = """
    Patient John Smith (42) was admitted to Massachusetts General Hospital in Boston on June 15, 2023.
    He has a history of Hypertension and Type 2 Diabetes. Current medications include Lipitor (20mg daily)
    and Metformin (500mg twice daily). The patient reported taking Ibuprofen for occasional headaches.
    Dr. Sarah Johnson recommended continuing current treatment and scheduled a follow-up appointment
    in 3 months at the clinic in New York.
    """
    
    # Process with standard NER
    standard_result = processor.process(medical_text)
    
    # Process with custom medical dictionary
    medical_entities = medical_recognizer.recognize_entities(medical_text)
    
    # Display standard NER results
    print("\nStandard NER results:")
    entities = standard_result[standard_result["entity_label"].notna()]
    for _, row in entities.iterrows():
        print(f"  {row['token']} - {row['entity_label']}")
    
    # Display custom medical entities
    print("\nCustom medical entities:")
    for entity in medical_entities:
        print(f"  {entity['text']} - {entity['label']}")
    
    # Combine the results (in a real application, you would merge these more carefully)
    print("\nCombined entity recognition provides both standard entities (PERSON, ORG, DATE) and domain-specific entities (MEDICATION, CONDITION)")
    
    print("=" * 80)

def main():
    """Main function to demonstrate custom entity recognition."""
    custom_dictionary_example()
    combined_approach_example()

if __name__ == "__main__":
    main()
