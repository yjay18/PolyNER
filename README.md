# PolyNER

PolyNER is a Python library for multilingual Named Entity Recognition to simplify NER related workflows.

## Features (v.0.1.0)

- **Language Detection**: Automatically detect the language of each text snippet.
- **Tokenization and Normalization**: Split text into tokens and normalize them (lowercase, remove punctuation, etc.).
- **Emoji Handling**: Detect and properly separate emojis as their own tokens.
- **Data Organization**: Output structured data with columns for each category (tokens, language, emojis, recognized entities).
- **Extensibility**: Load custom NER models or dictionaries.

## Installation

```bash
pip install polyner
```

## Quick Start

```python
from polyner import PolyNER

# Initialize the processor
processor = PolyNER()

# Process a text with mixed languages and emojis
text = "Hello world! 你好世界! 😊 Bonjour le monde!"
result = processor.process(text)

# Display the results
print(result)
```

## Output Format

PolyNER returns a pandas DataFrame with the following columns:

- `token`: The individual token
- `language`: Detected language of the token
- `is_emoji`: Boolean indicating if the token is an emoji
- `norm_token`: Normalized version of the token
- `entity_label`: Entity type if recognized (PERSON, LOC, ORG, etc.)

## Using Custom NER Models

```python
import spacy

# Load your custom model
custom_model = spacy.load("your_custom_model")

# Initialize with custom model
processor = PolyNER(ner_model=custom_model)

# Process text
result = processor.process("Your text here with custom entities")
```

## License

MIT
