"""
Tests for the emoji handling functionality.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import the package during testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from polyner.emoji_handling import (
    is_emoji,
    extract_emojis,
    get_emoji_description,
    categorize_emoji
)

class TestEmojiHandling(unittest.TestCase):
    """Test emoji handling functionality."""
    
    def test_is_emoji(self):
        """Test emoji detection."""
        # Test with common emojis
        self.assertTrue(is_emoji("😊"))
        self.assertTrue(is_emoji("👍"))
        self.assertTrue(is_emoji("🎉"))
        self.assertTrue(is_emoji("❤️"))
        
        # Test with non-emojis
        self.assertFalse(is_emoji("a"))
        self.assertFalse(is_emoji("Hello"))
        self.assertFalse(is_emoji("123"))
        self.assertFalse(is_emoji("."))
        
        # Test with empty string
        self.assertFalse(is_emoji(""))
        
        # Test with multi-character strings
        self.assertFalse(is_emoji("a😊"))
        self.assertFalse(is_emoji("😊a"))
    
    def test_extract_emojis(self):
        """Test emoji extraction from text."""
        # Test with text containing emojis
        text = "Hello 😊 world! 👍 How are you? 🎉"
        emojis = extract_emojis(text)
        
        self.assertEqual(len(emojis), 3)
        self.assertIn("😊", emojis)
        self.assertIn("👍", emojis)
        self.assertIn("🎉", emojis)
        
        # Test with text without emojis
        text = "Hello world! How are you?"
        emojis = extract_emojis(text)
        self.assertEqual(emojis, [])
        
        # Test with only emojis
        text = "😊👍🎉"
        emojis = extract_emojis(text)
        self.assertEqual(len(emojis), 3)
        self.assertEqual(emojis, ["😊", "👍", "🎉"])
        
        # Test with empty text
        emojis = extract_emojis("")
        self.assertEqual(emojis, [])
    
    def test_get_emoji_description(self):
        """Test getting emoji descriptions."""
        # Test with common emojis
        self.assertTrue(len(get_emoji_description("😊")) > 0)
        self.assertTrue(len(get_emoji_description("👍")) > 0)
        
        # Test with non-emoji
        self.assertEqual(get_emoji_description("a"), "")
        self.assertEqual(get_emoji_description(""), "")
    
    def test_categorize_emoji(self):
        """Test emoji categorization."""
        # Test face category
        self.assertEqual(categorize_emoji("😊"), "face")
        self.assertEqual(categorize_emoji("😂"), "face")
        
        # Test heart category
        self.assertEqual(categorize_emoji("❤️"), "heart")
        
        # Test hand category
        self.assertEqual(categorize_emoji("👍"), "hand")
        self.assertEqual(categorize_emoji("👋"), "hand")
        
        # Test non-emoji
        self.assertEqual(categorize_emoji("a"), "not_emoji")
        self.assertEqual(categorize_emoji(""), "not_emoji")
        
        # Additional categories might be harder to test explicitly
        # as they depend on the emoji description from the emoji library
        # But we can at least check that they return a string
        self.assertTrue(isinstance(categorize_emoji("🐱"), str))  # animal
        self.assertTrue(isinstance(categorize_emoji("🍎"), str))  # food
        self.assertTrue(isinstance(categorize_emoji("🏁"), str))  # flag


if __name__ == "__main__":
    unittest.main()