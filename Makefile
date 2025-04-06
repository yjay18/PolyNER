# Makefile for PolyNER

.PHONY: dev clean test lint format dist install uninstall

# Python interpreter to use
PYTHON = python
PIP = pip

# Project settings
PROJECT_NAME = polyner
TEST_DIR = tests
SOURCE_DIR = $(PROJECT_NAME)

# Default target
.DEFAULT_GOAL := help

# Help target
help:
	@echo "Available commands:"
	@echo "  make dev         - Install development dependencies"
	@echo "  make test        - Run all unit tests"
	@echo "  make clean       - Clean build artifacts and cache files"
	@echo "  make lint        - Run linters (flake8, mypy)"
	@echo "  make format      - Format code with black and isort"
	@echo "  make dist        - Create distribution packages"
	@echo "  make install     - Install the package locally"
	@echo "  make uninstall   - Uninstall the package"

# Install development dependencies
dev:
	$(PIP) install -e ".[dev]" --use-pep517
	$(PYTHON) -m spacy download en_core_web_sm

# Alternative if noe dev extras
dev-alt:
	$(PIP) install -e .
	$(PIP) install pytest pytest-cov flake8 mypy black isort
	$(PYTHON) -m spacy download en_core_web_sm

# Run tests
test:
	$(PYTHON) -m pytest $(TEST_DIR) -v

# Run tests with coverage
test-cov:
	$(PYTHON) -m pytest $(TEST_DIR) --cov=$(SOURCE_DIR) --cov-report=term --cov-report=html

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run linters
lint:
	$(PYTHON) -m flake8 --ignore=E501 $(SOURCE_DIR) $(TEST_DIR)

# Format code
format:
	$(PYTHON) -m black $(SOURCE_DIR) $(TEST_DIR)

# Create distribution packages
dist: clean
	$(PYTHON) -m build

# Install the package
install:
	$(PIP) install .

# Uninstall the package
uninstall:
	$(PIP) uninstall -y $(PROJECT_NAME)