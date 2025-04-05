"""
Tests for the utility functions.
"""

import unittest
import sys
import os
import pandas as pd
import tempfile

# Add the parent directory to the path so we can import the package during testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from polyner.utils import (
    dataframe_to_json,
    dataframe_to_csv,
    merge_dataframes,
    filter_by_language,
    filter_by_entity,
    filter_emojis,
    get_language_distribution,
    get_entity_distribution,
    get_emoji_distribution,
    save_to_file,
    load_from_file
)

class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            "token": ["Hello", "world", "😊", "Bonjour", "le", "monde", "👋"],
            "language": ["en", "en", None, "fr", "fr", "fr", None],
            "is_emoji": [False, False, True, False, False, False, True],
            "norm_token": ["hello", "world", "😊", "bonjour", "le", "monde", "👋"],
            "entity_label": ["GREETING", None, None, "GREETING", None, None, None]
        })
    
    def test_dataframe_to_json(self):
        """Test converting DataFrame to JSON."""
        # Convert to JSON
        json_str = dataframe_to_json(self.df)
        
        # Check that it's a valid JSON string
        self.assertIsInstance(json_str, str)
        
        # Try to parse it back to a DataFrame
        df_from_json = pd.read_json(json_str)
        
        # Check that it has the same shape
        self.assertEqual(df_from_json.shape, self.df.shape)
    
    def test_dataframe_to_csv(self):
        """Test converting DataFrame to CSV."""
        # Convert to CSV
        csv_str = dataframe_to_csv(self.df)
        
        # Check that it's a valid CSV string
        self.assertIsInstance(csv_str, str)
        
        # Try to parse it back to a DataFrame
        df_from_csv = pd.read_csv(pd.StringIO(csv_str))
        
        # Check that it has the same shape
        self.assertEqual(df_from_csv.shape, self.df.shape)
    
    def test_merge_dataframes(self):
        """Test merging multiple DataFrames."""
        # Create additional DataFrames
        df1 = pd.DataFrame({"token": ["Hello"], "language": ["en"], "is_emoji": [False]})
        df2 = pd.DataFrame({"token": ["world"], "language": ["en"], "is_emoji": [False]})
        
        # Merge them
        merged_df = merge_dataframes([df1, df2])
        
        # Check the result
        self.assertEqual(len(merged_df), 2)
        self.assertIn("Hello", merged_df["token"].values)
        self.assertIn("world", merged_df["token"].values)
        
        # Test with empty list
        empty_merged = merge_dataframes([])
        self.assertTrue(empty_merged.empty)
    
    def test_filter_by_language(self):
        """Test filtering by language."""
        # Filter English tokens
        en_df = filter_by_language(self.df, "en")
        
        # Check that we only got English tokens
        self.assertEqual(len(en_df), 2)
        self.assertTrue(all(lang == "en" for lang in en_df["language"]))
        self.assertIn("Hello", en_df["token"].values)
        self.assertIn("world", en_df["token"].values)
        
        # Filter French tokens
        fr_df = filter_by_language(self.df, "fr")
        
        # Check that we only got French tokens
        self.assertEqual(len(fr_df), 3)
        self.assertTrue(all(lang == "fr" for lang in fr_df["language"]))
        self.assertIn("Bonjour", fr_df["token"].values)
    
    def test_filter_by_entity(self):
        """Test filtering by entity."""
        # Filter all entities
        entities_df = filter_by_entity(self.df)
        
        # Check that we only got tokens with entities
        self.assertEqual(len(entities_df), 2)
        self.assertTrue(all(entity is not None for entity in entities_df["entity_label"]))
        self.assertIn("Hello", entities_df["token"].values)
        self.assertIn("Bonjour", entities_df["token"].values)
        
        # Filter by specific entity type
        greeting_df = filter_by_entity(self.df, "GREETING")
        
        # Check that we only got GREETING entities
        self.assertEqual(len(greeting_df), 2)
        self.assertTrue(all(entity == "GREETING" for entity in greeting_df["entity_label"]))
    
    def test_filter_emojis(self):
        """Test filtering emojis."""
        # Filter emoji tokens
        emojis_df = filter_emojis(self.df)
        
        # Check that we only got emoji tokens
        self.assertEqual(len(emojis_df), 2)
        self.assertTrue(all(is_emoji for is_emoji in emojis_df["is_emoji"]))
        self.assertIn("😊", emojis_df["token"].values)
        self.assertIn("👋", emojis_df["token"].values)
    
    def test_get_language_distribution(self):
        """Test getting language distribution."""
        # Get language distribution
        lang_dist = get_language_distribution(self.df)
        
        # Check that we got the expected counts
        self.assertEqual(lang_dist["en"], 2)
        self.assertEqual(lang_dist["fr"], 3)
        
        # Check that we didn't count None or emojis
        self.assertEqual(len(lang_dist), 2)
    
    def test_get_entity_distribution(self):
        """Test getting entity distribution."""
        # Get entity distribution
        entity_dist = get_entity_distribution(self.df)
        
        # Check that we got the expected counts
        self.assertEqual(entity_dist["GREETING"], 2)
        
        # Check that we didn't count None
        self.assertEqual(len(entity_dist), 1)
    
    def test_get_emoji_distribution(self):
        """Test getting emoji distribution."""
        # Get emoji distribution
        emoji_dist = get_emoji_distribution(self.df)
        
        # Check that we got the expected counts
        self.assertEqual(emoji_dist["😊"], 1)
        self.assertEqual(emoji_dist["👋"], 1)
        
        # Check that we didn't count non-emojis
        self.assertEqual(len(emoji_dist), 2)
    
    def test_save_and_load_file(self):
        """Test saving and loading to/from files."""
        # We'll use temporary files for testing
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as csv_file, \
             tempfile.NamedTemporaryFile(suffix='.json', delete=False) as json_file:
            
            try:
                # Save to CSV
                csv_path = csv_file.name
                save_to_file(self.df, csv_path, format="csv")
                
                # Load from CSV
                loaded_csv = load_from_file(csv_path)
                
                # Check that it loaded correctly
                self.assertEqual(loaded_csv.shape, self.df.shape)
                
                # Save to JSON
                json_path = json_file.name
                save_to_file(self.df, json_path, format="json")
                
                # Load from JSON
                loaded_json = load_from_file(json_path)
                
                # Check that it loaded correctly
                self.assertEqual(loaded_json.shape, self.df.shape)
                
            finally:
                # Clean up
                os.unlink(csv_path)
                os.unlink(json_path)
    
    def test_unsupported_format(self):
        """Test handling of unsupported formats."""
        # Try to save with an unsupported format
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file, \
             self.assertRaises(ValueError):
            save_to_file(self.df, temp_file.name, format="txt")
        
        # Try to load from an unsupported format
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file, \
             self.assertRaises(ValueError):
            load_from_file(temp_file.name)


if __name__ == "__main__":
    unittest.main()