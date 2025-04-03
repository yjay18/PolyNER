"""
Utility functions for the PolyNER library.
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Union
import json
import csv
import io

def dataframe_to_json(df: pd.DataFrame, orient: str = "records") -> str:
    """
    Convert a DataFrame to JSON.
    
    Args:
        df: Input DataFrame
        orient: Orientation of the JSON output
        
    Returns:
        JSON string
    """
    return df.to_json(orient=orient)

def dataframe_to_csv(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to CSV.
    
    Args:
        df: Input DataFrame
        
    Returns:
        CSV string
    """
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()

def merge_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple DataFrames into one.
    
    Args:
        dfs: List of DataFrames
        
    Returns:
        Merged DataFrame
    """
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)

def filter_by_language(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """
    Filter DataFrame to only include tokens of a specific language.
    
    Args:
        df: Input DataFrame
        language: Language code
        
    Returns:
        Filtered DataFrame
    """
    return df[df["language"] == language]

def filter_by_entity(df: pd.DataFrame, entity_type: Optional[str] = None) -> pd.DataFrame:
    """
    Filter DataFrame to only include tokens with entity labels.
    
    Args:
        df: Input DataFrame
        entity_type: Optional specific entity type to filter for
        
    Returns:
        Filtered DataFrame
    """
    if entity_type:
        return df[df["entity_label"] == entity_type]
    else:
        return df[df["entity_label"].notna()]

def filter_emojis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to only include emoji tokens.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Filtered DataFrame
    """
    return df[df["is_emoji"] == True]

def get_language_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get the distribution of languages in the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping language codes to counts
    """
    # Filter out emojis and None values
    lang_df = df[(df["language"].notna()) & (df["is_emoji"] == False)]
    
    # Count occurrences of each language
    return lang_df["language"].value_counts().to_dict()

def get_entity_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get the distribution of entity types in the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping entity types to counts
    """
    # Filter out None values
    entity_df = df[df["entity_label"].notna()]
    
    # Count occurrences of each entity type
    return entity_df["entity_label"].value_counts().to_dict()

def get_emoji_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get the distribution of emojis in the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping emojis to counts
    """
    # Filter to only include emojis
    emoji_df = df[df["is_emoji"] == True]
    
    # Count occurrences of each emoji
    return emoji_df["token"].value_counts().to_dict()

def save_to_file(df: pd.DataFrame, file_path: str, format: str = "csv") -> None:
    """
    Save DataFrame to a file.
    
    Args:
        df: Input DataFrame
        file_path: Path to save the file
        format: File format (csv, json, excel)
    """
    format = format.lower()
    
    if format == "csv":
        df.to_csv(file_path, index=False)
    elif format == "json":
        df.to_json(file_path, orient="records", indent=2)
    elif format == "excel":
        df.to_excel(file_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_from_file(file_path: str) -> pd.DataFrame:
    """
    Load DataFrame from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Loaded DataFrame
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        return pd.read_json(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
