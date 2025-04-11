# PolyNER

PolyNER is a Python library for multilingual Named Entity Recognition to simplify NER related workflows.

## Features (v.0.2.0)

- **Language Detection**: Automatically detect the language of each text snippet.
- **Tokenization and Normalization**: Split text into tokens and normalize them (lowercase, remove punctuation, etc.).
- **Emoji Handling**: Detect and properly separate emojis as their own tokens.
- **Data Organization**: Output structured data with columns for each category (tokens, language, emojis, recognized entities).
- **Extensibility**: Load custom NER models or dictionaries.
- **Multilingual Support**: Process text in multiple languages with enhanced entity recognition.

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

### Multilingual Processing

```python
from polyner import PolyNER

# Initialize the processor
processor = PolyNER()

# Process a multilingual text
text = """
Apple Inc. is headquartered in Cupertino, California.
Google tiene su sede en Mountain View.
Amazon wurde von Jeff Bezos gegründet.
"""

# Process with multilingual support
result = processor.process_multi(text)

# Show detected entities
entities = result[result["entity_label"].notna()]
print(entities[["token", "language", "entity_label"]])

# You can also specify a different model
result = processor.process_multi(text, model_name="xlm-roberta-base-finetuned-panx-all")
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
from polyner import PolyNER

# Load your custom model
custom_model = spacy.load("your_custom_model")

# Initialize with custom model
processor = PolyNER(ner_model=custom_model)

# Process text
result = processor.process("Your text here with custom entities")
```

### Using Hugging Face Transformer Models

```python
from polyner import PolyNER

# Initialize the processor
processor = PolyNER()

# Process with a specific Hugging Face model
text = "Microsoft was founded by Bill Gates in Redmond, Washington."
result = processor.process_multi(text, model_name="Babelscape/wikineural-multilingual-ner")
```

### Batch Processing

```python
from polyner import PolyNER

processor = PolyNER()

# List of texts
texts = [
    "Apple Inc. is based in Cupertino.",
    "Google tiene su sede en Mountain View.",
    "Amazon wurde von Jeff Bezos gegründet."
]

# Process batch
results = processor.process_batch_multi(texts)

# Access results
for i, df in enumerate(results):
    print(f"Text {i+1} entities:")
    entities = df[df["entity_label"].notna()]
    print(entities[["token", "language", "entity_label"]])
```

## License

MIT
