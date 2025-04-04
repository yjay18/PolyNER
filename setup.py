from setuptools import setup, find_packages

setup(
    name="polyner",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "langdetect",
        "emoji",
        "spacy",
    ],
    python_requires=">=3.7",
    author="PolyNER Team",
    author_email="example@example.com",
    description="A multilingual NER library that handles text, emojis, and multiple languages",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/polyner",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)