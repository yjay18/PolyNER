"""
Module for emoji detection and handling.
"""

import emoji
from typing import List, Set

def is_emoji(text: str) -> bool:
    """
    Check if a string is an emoji.
    
    Args:
        text: Input text
        
    Returns:
        True if the text is an emoji
    """
    return emoji.is_emoji(text)

def extract_emojis(text: str) -> List[str]:
    """
    Extract all emojis from a text.
    
    Args:
        text: Input text
        
    Returns:
        List of emojis found in the text
    """
    return [c for c in text if emoji.is_emoji(c)]

def get_emoji_description(emoji_char: str) -> str:
    """
    Get the description of an emoji.
    
    Args:
        emoji_char: Emoji character
        
    Returns:
        Description of the emoji or empty string if not found
    """
    if not is_emoji(emoji_char):
        return ""
    
    # Get emoji name from the emoji library
    emoji_data = emoji.EMOJI_DATA.get(emoji_char, {})
    return emoji_data.get('en', '')

def categorize_emoji(emoji_char: str) -> str:
    """
    Categorize an emoji into a general category.
    
    Args:
        emoji_char: Emoji character
        
    Returns:
        Category of the emoji or 'unknown' if not found
    """
    if not is_emoji(emoji_char):
        return "not_emoji"
    
    # Get emoji data
    emoji_data = emoji.EMOJI_DATA.get(emoji_char, {})
    
    # Extract category from the emoji data
    # This is a simplified approach
    # TODO: Better more robust implementation
    description = emoji_data.get('en', '').lower()
    
    # Simple categorization based on keywords in description
    if any(word in description for word in ['face', 'smile', 'laugh', 'wink']):
        return "face"
    elif any(word in description for word in ['hand', 'finger', 'arm']):
        return "hand"
    elif any(word in description for word in ['heart', 'love']):
        return "heart"
    elif any(word in description for word in ['flag']):
        return "flag"
    elif any(word in description for word in ['animal', 'cat', 'dog', 'bird']):
        return "animal"
    elif any(word in description for word in ['food', 'fruit', 'drink']):
        return "food"
    else:
        return "other"