# Variables
PYTHON = python3

.PHONY: test clean

# Run all unit tests inside the test directory
test:
	PYTHONPATH=. $(PYTHON) -m unittest discover -s test -p "test_*.py"

# Clean up temporary Python caching artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete