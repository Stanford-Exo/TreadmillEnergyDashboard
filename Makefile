# Variables
PYTHON = python3

.PHONY: test clean validate export

# Run all unit tests inside the test directory
test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s test -p "test_*.py"

# Run the offline Kalman Filter validation evaluation
validate:
	PYTHONPATH=src $(PYTHON) -m validation.validate_kf

# Run the AddBiomechanics data exporter
export:
	PYTHONPATH=src $(PYTHON) -m addbiomechanics_export.export

# Clean up temporary Python caching artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete