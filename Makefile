# Variables
PYTHON = python3

.PHONY: test clean validate export

# GNU Make pattern to capture trailing targets and translate them into flags.
# This allows running: make validate plot
ifeq (validate,$(firstword $(MAKECMDGOALS)))
  VALIDATE_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  
  # Translate positional target words to script flags (e.g., 'plot' -> '--plot')
  VALIDATE_FLAGS := $(subst plot,--plot,$(VALIDATE_ARGS))
  
  # Turn trailing arguments into empty do-nothing targets to prevent "No rule to make target" errors
  $(eval $(VALIDATE_ARGS):;@:)
endif

# Run all unit tests inside the test directory
test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s test -p "test_*.py"

# Run the offline Kalman Filter validation evaluation
validate:
	PYTHONPATH=src $(PYTHON) -m validation.validate_kf $(VALIDATE_FLAGS)

# Run the AddBiomechanics data exporter
export:
	PYTHONPATH=src $(PYTHON) -m addbiomechanics_export.export

# Clean up temporary Python caching artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete