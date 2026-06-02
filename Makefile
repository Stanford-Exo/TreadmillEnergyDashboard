# Variables
PYTHON = python3

.PHONY: test clean validate validate-strides export analyze-poggensee

# GNU Make pattern to capture trailing targets and translate them into flags.
# This allows running: make validate plot
ifeq (validate,$(firstword $(MAKECMDGOALS)))
  VALIDATE_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))

  # Translate positional target words to script flags (e.g., 'plot' -> '--plot')
  VALIDATE_FLAGS := $(subst plot,--plot,$(VALIDATE_ARGS))

  # Turn trailing arguments into empty do-nothing targets to prevent "No rule to make target" errors
  $(eval $(VALIDATE_ARGS):;@:)
endif

# Capture trailing targets for validate-strides.
# This allows running: make validate-strides plot
ifeq (validate-strides,$(firstword $(MAKECMDGOALS)))
  STRIDES_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))

  # Translate positional target words to script flags (e.g., 'plot' -> '--plot')
  STRIDES_FLAGS := $(subst plot,--plot,$(STRIDES_ARGS))

  # Turn trailing arguments into empty do-nothing targets to prevent "No rule to make target" errors
  $(eval $(STRIDES_ARGS):;@:)
endif

# Run all unit tests inside the test directory
test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s test -p "test_*.py"

# Run the offline Kalman Filter validation evaluation
validate:
	PYTHONPATH=src $(PYTHON) -m validation.validate_kf $(VALIDATE_FLAGS)

# Run stride-based gait segmentation and energy aggregation
validate-strides:
	PYTHONPATH=src $(PYTHON) -m validation.validate_strides $(STRIDES_FLAGS)

# Run the AddBiomechanics data exporter
export:
	PYTHONPATH=src $(PYTHON) -m export.addbiomechanics_export.export

# Run the Katie Poggensee Exoskeleton data exporter
export-poggensee:
	PYTHONPATH=src $(PYTHON) -m export.poggensee_export.export_poggensee

precompute-poggensee:
	PYTHONPATH=src $(PYTHON) -m analysis.precompute_poggensee

# Run the precompute script through the cProfile profiler
profile-poggensee:
		PYTHONPATH=src $(PYTHON) -m cProfile -o profile.stats -m analysis.precompute_poggensee
		snakeviz profile.stats

animate-poggensee:
	PYTHONPATH=src $(PYTHON) -m analysis.animate_adaptation --save

correlate-metabolics:
	PYTHONPATH=src $(PYTHON) -m analysis.correlate_metabolics

# Run the Katie Poggensee stride power analyzer
analyze-poggensee:
	PYTHONPATH=src $(PYTHON) -m analysis.analyze_poggensee

# Run the custom coefficient fitting regression tool
fit-metabolics:
	PYTHONPATH=src $(PYTHON) -m analysis.fit_metabolic_coefficients

# Run the Python REST/Ingestion backend server
server:
	PYTHONPATH=src $(PYTHON) src/server/server.py

# Clean up temporary Python caching artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
