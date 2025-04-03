"""
Example script demonstrating how to use PolyNER for multilingual text analysis.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from polyner import PolyNER
from polyner.utils import (
    filter_by_language, 
    filter_emojis, 
    get_language_distribution, 
    get_entity_distribution,
    get_emoji_distribution
)

def analyze_multilingual_text(text):
    """
    Analyze multilingual text with PolyNER and display results.
    
    Args:
        text: Input text to analyze
    """
    print("=" * 80)
    print("Analyzing text:")
    print(text)
    print("=" * 80)
    
    # Initialize the processor
    processor = PolyNER()
    
    # Process the text
    result = processor.process(text)
    
    # Display the full result
    print("\nFull analysis result:")
    print(result)
    
    # Display language distribution
    lang_dist = get_language_distribution(result)
    print("\nLanguage distribution:")
    for lang, count in lang_dist.items():
        print(f"  {lang}: {count} tokens")
    
    # Display emoji distribution
    emoji_dist = get_emoji_distribution(result)
    if emoji_dist:
        print("\nEmoji distribution:")
        for emoji, count in emoji_dist.items():
            print(f"  {emoji}: {count}")
    
    # Display entity distribution
    entity_dist = get_entity_distribution(result)
    if entity_dist:
        print("\nEntity distribution:")
        for entity_type, count in entity_dist.items():
            print(f"  {entity_type}: {count}")
        
        # Display detected entities
        entities = result[result["entity_label"].notna()]
        if not entities.empty:
            print("\nDetected entities:")
            for _, row in entities.iterrows():
                print(f"  {row['token']} - {row['entity_label']}")
    
    print("=" * 80)
    return result

def main():
    """Main function to demonstrate PolyNER functionality."""
    # Example 1: Social media post with multiple languages and emojis
    social_media_post = """
    Just had an amazing dinner at Le Petit Bistro in Paris! 🍷 The escargot was délicieux! 
    Next stop: Berlin for Oktoberfest! 🍺 Can't wait to try some authentic German bratwurst.
    私は日本食も大好きです。寿司と天ぷらは最高です！🍣
    """
    
    result1 = analyze_multilingual_text(social_media_post)
    
    # Example 2: News article with named entities
    news_article = """
    WASHINGTON (Reuters) - U.S. President Joe Biden met with Chinese President Xi Jinping 
    at the G20 summit in Rome on Tuesday. The leaders discussed trade relations between 
    the United States and China, as well as climate change initiatives ahead of the 
    COP26 conference in Glasgow, Scotland.
    
    Meanwhile, tech giants Apple and Google announced new AI partnerships, 
    while Tesla's stock surged 5% after reporting record quarterly earnings.
    """
    
    result2 = analyze_multilingual_text(news_article)
    
    # Example 3: Customer feedback with emojis and sentiment
    customer_feedback = """
    I absolutely love ❤️ your new app! The interface is so intuitive and user-friendly 👍.
    However, I noticed a few bugs in the checkout process 🐛. Sometimes it crashes when I try to pay 😡.
    Overall though, great job! Can't wait for the next update! 🎉
    """
    
    result3 = analyze_multilingual_text(customer_feedback)

if __name__ == "__main__":
    main()
