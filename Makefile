# Mental Health Tweet Classifier - Makefile
# Provides convenient commands for setup, training, evaluation, and deployment

# Variables
PYTHON := python
PIP := pip
PROJECT_NAME := mental_health_tweet_classifier
SRC_DIR := src
DATA_DIR := dataset
MODELS_DIR := models
REPORTS_DIR := reports
NOTEBOOKS_DIR := notebooks
TESTS_DIR := tests

# Virtual environment name (optional)
VENV := venv

# Default target
.DEFAULT_GOAL := help

##@ Setup Commands

.PHONY: setup
setup: ## Install dependencies and set up the project
	@echo "Setting up $(PROJECT_NAME)..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Downloading NLTK data..."
	$(PYTHON) -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
	@mkdir -p $(MODELS_DIR) $(REPORTS_DIR) logs experiments
	@echo "Setup complete! ✅"

.PHONY: setup-dev
setup-dev: setup ## Install development dependencies
	@echo "Installing development dependencies..."
	$(PIP) install pytest black flake8 mypy jupyter
	@echo "Development setup complete! ✅"

.PHONY: create-venv
create-venv: ## Create virtual environment
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created! Activate with: source $(VENV)/bin/activate (Linux/Mac) or $(VENV)\Scripts\activate (Windows)"

##@ Data Commands

.PHONY: explore-data
explore-data: ## Run exploratory data analysis notebook
	@echo "Running EDA notebook..."
	$(PYTHON) -m jupyter nbconvert --to notebook --execute $(NOTEBOOKS_DIR)/01_explore.ipynb --output 01_explore_executed.ipynb
	@echo "EDA complete! Check $(NOTEBOOKS_DIR)/01_explore_executed.ipynb"

.PHONY: clean-data
clean-data: ## Clean and preprocess the dataset
	@echo "Cleaning dataset..."
	$(PYTHON) -m $(SRC_DIR).data.preprocess --input $(DATA_DIR) --output $(DATA_DIR)/processed
	@echo "Data cleaning complete! ✅"

##@ Training Commands

.PHONY: train-baseline
train-baseline: ## Train baseline models (TF-IDF + Classical ML)
	@echo "Training baseline models..."
	$(PYTHON) -m $(SRC_DIR).models.baseline --data $(DATA_DIR) --output $(MODELS_DIR)/baseline
	@echo "Baseline training complete! ✅"

.PHONY: train-transformer
train-transformer: ## Train transformer model (BERT/RoBERTa)
	@echo "Training transformer model..."
	$(PYTHON) -m $(SRC_DIR).models.transformer --data $(DATA_DIR) --output $(MODELS_DIR)/transformer
	@echo "Transformer training complete! ✅"

.PHONY: train-all
train-all: train-baseline train-transformer ## Train all models
	@echo "All models trained! ✅"

##@ Evaluation Commands

.PHONY: evaluate-baseline
evaluate-baseline: ## Evaluate baseline models
	@echo "Evaluating baseline models..."
	$(PYTHON) -m $(SRC_DIR).eval.evaluate --model $(MODELS_DIR)/baseline --data $(DATA_DIR) --output $(REPORTS_DIR)/baseline_evaluation
	@echo "Baseline evaluation complete! ✅"

.PHONY: evaluate-transformer
evaluate-transformer: ## Evaluate transformer model
	@echo "Evaluating transformer model..."
	$(PYTHON) -m $(SRC_DIR).eval.evaluate --model $(MODELS_DIR)/transformer --data $(DATA_DIR) --output $(REPORTS_DIR)/transformer_evaluation
	@echo "Transformer evaluation complete! ✅"

.PHONY: evaluate-all
evaluate-all: evaluate-baseline evaluate-transformer ## Evaluate all models
	@echo "All models evaluated! ✅"

##@ Explanation Commands

.PHONY: explain-predictions
explain-predictions: ## Generate model explanations (SHAP, LIME)
	@echo "Generating model explanations..."
	$(PYTHON) -m $(SRC_DIR).eval.explain --model $(MODELS_DIR) --data $(DATA_DIR) --output $(REPORTS_DIR)/explanations
	@echo "Explanations generated! ✅"

##@ Demo Commands

.PHONY: demo
demo: ## Run Streamlit demo application
	@echo "Starting Streamlit demo..."
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.port 8501
	@echo "Demo available at http://localhost:8501"

.PHONY: api
api: ## Start FastAPI server (if implemented)
	@echo "Starting FastAPI server..."
	$(PYTHON) -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
	@echo "API available at http://localhost:8000"

##@ Testing Commands

.PHONY: test
test: ## Run unit tests
	@echo "Running tests..."
	$(PYTHON) -m pytest $(TESTS_DIR) -v
	@echo "Tests complete! ✅"

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	$(PYTHON) -m pytest $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/ ✅"

##@ Code Quality Commands

.PHONY: format
format: ## Format code with Black
	@echo "Formatting code..."
	$(PYTHON) -m black $(SRC_DIR) $(TESTS_DIR) app/
	@echo "Code formatting complete! ✅"

.PHONY: lint
lint: ## Lint code with flake8
	@echo "Linting code..."
	$(PYTHON) -m flake8 $(SRC_DIR) $(TESTS_DIR) app/
	@echo "Linting complete! ✅"

.PHONY: type-check
type-check: ## Type check with mypy
	@echo "Type checking..."
	$(PYTHON) -m mypy $(SRC_DIR)
	@echo "Type checking complete! ✅"

.PHONY: quality
quality: format lint type-check ## Run all code quality checks
	@echo "All quality checks complete! ✅"

##@ Docker Commands

.PHONY: docker-build
docker-build: ## Build Docker image
	@echo "Building Docker image..."
	docker build -t $(PROJECT_NAME) .
	@echo "Docker image built! ✅"

.PHONY: docker-run
docker-run: ## Run Docker container
	@echo "Running Docker container..."
	docker run -p 8501:8501 $(PROJECT_NAME)
	@echo "Container running at http://localhost:8501"

.PHONY: docker-shell
docker-shell: ## Open shell in Docker container
	@echo "Opening shell in container..."
	docker run -it $(PROJECT_NAME) /bin/bash

##@ Cleanup Commands

.PHONY: clean
clean: ## Clean up generated files
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	@echo "Cleanup complete! ✅"

.PHONY: clean-models
clean-models: ## Remove trained models
	@echo "Removing trained models..."
	rm -rf $(MODELS_DIR)/*
	@echo "Models removed! ✅"

.PHONY: clean-reports
clean-reports: ## Remove evaluation reports
	@echo "Removing reports..."
	rm -rf $(REPORTS_DIR)/*
	@echo "Reports removed! ✅"

.PHONY: clean-all
clean-all: clean clean-models clean-reports ## Clean everything
	@echo "Everything cleaned! ✅"

##@ Documentation Commands

.PHONY: docs
docs: ## Generate documentation
	@echo "Generating documentation..."
	$(PYTHON) -c "import pydoc; pydoc.writedoc('$(SRC_DIR)')"
	@echo "Documentation generated! ✅"

.PHONY: notebook
notebook: ## Start Jupyter notebook server
	@echo "Starting Jupyter notebook..."
	$(PYTHON) -m jupyter notebook --notebook-dir=$(NOTEBOOKS_DIR)

##@ Git Commands

.PHONY: git-init
git-init: ## Initialize git repository with initial commit
	@echo "Initializing git repository..."
	git init
	git add .
	git commit -m "Initial commit: Mental Health Tweet Classifier project setup"
	@echo "Git repository initialized! ✅"

##@ Pipeline Commands

.PHONY: full-pipeline
full-pipeline: setup explore-data train-all evaluate-all explain-predictions ## Run complete ML pipeline
	@echo "Full ML pipeline complete! ✅"

.PHONY: quick-start
quick-start: setup train-baseline evaluate-baseline demo ## Quick start for demo
	@echo "Quick start complete! Demo running at http://localhost:8501 ✅"

##@ Help

.PHONY: help
help: ## Display help message
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1mUsage:\033[0m\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: info
info: ## Show project information
	@echo "\n📊 Mental Health Tweet Classifier Project"
	@echo "========================================="
	@echo "Project: $(PROJECT_NAME)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo ""
	@echo "📁 Project Structure:"
	@echo "  $(SRC_DIR)/          - Source code"
	@echo "  $(DATA_DIR)/         - Dataset files"
	@echo "  $(MODELS_DIR)/       - Trained models"
	@echo "  $(REPORTS_DIR)/      - Evaluation reports"
	@echo "  $(NOTEBOOKS_DIR)/    - Jupyter notebooks"
	@echo "  $(TESTS_DIR)/        - Unit tests"
	@echo "  app/             - Demo applications"
	@echo "  docs/            - Documentation"
	@echo ""
	@echo "⚠️  IMPORTANT: This is for research/educational purposes only!"
	@echo "   NOT for clinical diagnosis or medical decisions."
	@echo "   See docs/ethics.md for full guidelines."
	@echo ""

# Special targets that don't create files
.PHONY: all setup setup-dev create-venv explore-data clean-data train-baseline train-transformer train-all evaluate-baseline evaluate-transformer evaluate-all explain-predictions demo api test test-coverage format lint type-check quality docker-build docker-run docker-shell clean clean-models clean-reports clean-all docs notebook git-init full-pipeline quick-start help info