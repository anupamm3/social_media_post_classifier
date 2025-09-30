# Mental Health Tweet Classifier

A comprehensive machine learning project for classifying tweets related to mental health, specifically identifying depression-related content vs. non-depression content. This project implements both traditional ML baselines and state-of-the-art transformer models with proper evaluation, explainability, and ethical considerations.

⚠️ **IMPORTANT DISCLAIMER**: This model is for research and educational purposes only. It is NOT intended for clinical diagnosis, medical advice, or treatment recommendations. If you or someone you know is experiencing mental health issues, please consult qualified mental health professionals.

## 🎯 Project Overview

This project provides:
- **Binary classification**: Depression vs Non-Depression tweets
- **Multiple model types**: Traditional ML (TF-IDF + Logistic Regression/Random Forest) and Transformers (BERT/RoBERTa)
- **Comprehensive evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC with proper cross-validation
- **Model explainability**: SHAP values, feature importance, and attention visualization
- **Interactive demo**: Streamlit web application with safety disclaimers
- **Reproducible pipeline**: End-to-end automation with experiment tracking

## 📊 Dataset

The project uses 4 CSV files containing tweet data:
- `d_tweets.csv` (3,497 samples) - Depression-related tweets
- `non_d_tweets.csv` (4,810 samples) - Non-depression tweets  
- `clean_d_tweets.csv` (3,083 samples) - Preprocessed depression tweets
- `clean_non_d_tweets.csv` (4,688 samples) - Preprocessed non-depression tweets

**Total**: ~16,000 tweets with balanced class distribution

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ installed
- Git (optional, for version control)  
- 4GB+ RAM recommended
- GPU optional (for transformer models)

### 1. Windows Setup (Recommended)

```cmd
# Clone the repository
git clone <repository-url>
cd mental_health_tweet

# Automatic setup using our Windows script
.\setup.bat setup

# Or using PowerShell directly
.\setup.ps1 setup
```

### 1. Manual Setup (All Platforms)

```bash
# Clone the repository
git clone <repository-url>
cd mental_health_tweet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download spaCy model (optional, for advanced preprocessing)
python -m spacy download en_core_web_sm
```

### 2. Windows Quick Commands

**Using our automated scripts:**

```cmd
# Explore the dataset
.\setup.bat explore-data

# Train baseline model
.\setup.bat train-baseline

# Train transformer model (GPU recommended) 
.\setup.bat train-transformer

# Evaluate models
.\setup.bat evaluate-all

# Start demo app
.\setup.bat demo

# Run complete pipeline
.\setup.bat pipeline
```

**Get help and see all available commands:**
```cmd
.\setup.bat help
```

### 3. Manual Commands (All Platforms)

#### Explore the Data

```bash
# Run exploratory data analysis
jupyter notebook notebooks/01_explore.ipynb

# Or generate EDA report programmatically
cd notebooks
python -c "import nbformat; exec(open('01_explore.ipynb').read())"
```

### 3. Train Models

#### Quick Baseline Model
```bash
# Train logistic regression baseline
python src/models/train.py quick-baseline

# Train random forest baseline  
python src/models/baseline.py --model-type random_forest
```

#### Quick Transformer Model
```bash
# Train RoBERTa model (requires GPU for reasonable speed)
python src/models/train.py quick-transformer --model-name roberta-base --epochs 3

# Train smaller model for CPU
python src/models/train.py quick-transformer --model-name distilbert-base-uncased --batch-size 8
```

#### Custom Training with Configuration
```bash
# Create configuration template
python src/models/train.py template --output my_config.yaml

# Edit my_config.yaml, then train
python src/models/train.py train --config my_config.yaml
```

### 4. Run Demo Application

```bash
# Start Streamlit app
streamlit run app/streamlit_app.py

# Open browser to http://localhost:8501
```

## 📁 Project Structure

```
.
├── dataset/                    # Raw CSV files
├── data/                      # Processed data (auto-generated)
│   ├── processed/             # Cleaned datasets
│   └── hydrated/              # Tweet hydration outputs (if needed)
├── notebooks/
│   └── 01_explore.ipynb       # Exploratory Data Analysis
├── src/
│   ├── data/
│   │   ├── load_data.py       # Data loading utilities
│   │   ├── preprocess.py      # Text preprocessing
│   │   └── hydrate_tweets.py  # Twitter API hydration (optional)
│   ├── features/
│   │   └── featurize.py       # Feature engineering (TF-IDF, BERT, classical)
│   ├── models/
│   │   ├── baseline.py        # Traditional ML models
│   │   ├── transformer.py     # Transformer models  
│   │   └── train.py           # Training orchestration
│   ├── eval/
│   │   └── evaluate.py        # Model evaluation metrics
│   └── explain/
│       └── explain.py         # Model explainability (SHAP, LIME)
├── app/
│   └── streamlit_app.py       # Demo web application
├── tests/
│   ├── test_data_loading.py   # Unit tests for data loading
│   └── test_preprocess.py     # Unit tests for preprocessing
├── reports/                   # Generated reports and visualizations
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container deployment
├── Makefile                   # Build automation
├── README.md                  # This file
├── LICENSE                    # MIT License
├── model_card.md              # Model documentation
└── ethics.md                  # Ethical guidelines and limitations
```

## 🛠️ Usage Examples

### Data Loading and Preprocessing

```python
from src.data.load_data import DataLoader
from src.data.preprocess import preprocess_tweets

# Load data
loader = DataLoader()
texts, labels = loader.get_text_and_labels(use_clean=False)

# Preprocess
cleaned_texts, filtered_labels, features = preprocess_tweets(
    texts, labels, 
    config={
        'remove_urls': True,
        'expand_contractions': True,
        'lowercase': True
    }
)
```

### Feature Engineering

```python
from src.features.featurize import FeaturePipeline

# Create feature pipeline
pipeline = FeaturePipeline(
    use_tfidf=True,
    use_bert=True,
    use_classical=True
)

# Extract features
features, feature_names = pipeline.fit_transform(texts, labels)
```

### Model Training

```python
from src.models.baseline import BaselineModel
from src.models.transformer import TransformerModel

# Train baseline model
baseline = BaselineModel(model_type='logistic')
baseline.fit(texts, labels)

# Train transformer model  
transformer = TransformerModel()
transformer.train(texts, labels)
```

### Model Evaluation

```python
from src.eval.evaluate import evaluate_model

# Evaluate model
results = evaluate_model(model, test_texts, test_labels)
print(f"Accuracy: {results['accuracy']:.3f}")
print(f"F1-Score: {results['f1']:.3f}")
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_data_loading.py -v
```

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t mental-health-classifier .

# Run container with demo app
docker run -p 8501:8501 mental-health-classifier

# Run with custom command
docker run mental-health-classifier python src/models/train.py quick-baseline
```

## 📈 Model Performance

### Baseline Model (Logistic Regression + TF-IDF)
- **Accuracy**: ~85-90%
- **F1-Score**: ~85-88%
- **Training time**: ~2-5 minutes
- **Inference**: <1ms per tweet

### Transformer Model (RoBERTa)
- **Accuracy**: ~90-95%
- **F1-Score**: ~90-93%
- **Training time**: ~30-60 minutes (GPU)
- **Inference**: ~10-50ms per tweet

## 🔍 Model Explainability

The project includes several explainability features:

1. **Feature Importance**: Top TF-IDF features and classical feature weights
2. **SHAP Values**: Local and global explanations for individual predictions
3. **Attention Visualization**: Transformer attention maps (for supported models)
4. **Example Analysis**: Misclassified examples and decision boundaries

```python
from src.explain.explain import explain_prediction

# Get explanation for a prediction
explanation = explain_prediction(model, "I feel so hopeless today...")
explanation.show()  # Display SHAP plot
```

## ⚠️ Ethical Considerations

This project includes comprehensive ethical guidelines:

### ⚠️ **Medical Disclaimer**
- **NOT for clinical use**: This model is not validated for medical diagnosis
- **Research only**: Intended for academic and research purposes
- **Seek professional help**: Direct users to qualified mental health professionals

### 🔒 **Privacy & Data Protection**
- **No personal data storage**: The demo app doesn't store user inputs
- **Anonymization required**: Remove identifying information from training data
- **GDPR compliance**: Follow data protection regulations

### 🎯 **Bias and Fairness**
- **Population bias**: Training data may not represent all demographics
- **Language bias**: Optimized for English text
- **Platform bias**: Based on Twitter/X data patterns

### 📋 **Usage Guidelines**
- Always display mental health resources and crisis hotlines
- Include disclaimer about model limitations
- Provide human-in-the-loop recommendations
- Regular model auditing and bias testing

See [ethics.md](ethics.md) for detailed guidelines.

## 📚 Documentation

- **[Model Card](model_card.md)**: Detailed model documentation
- **[Ethics Guide](ethics.md)**: Ethical considerations and guidelines
- **[API Documentation](docs/)**: Code documentation and examples
- **[Dataset Schema](reports/dataset_schema.json)**: Data structure documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Include ethical impact assessment

## 📞 Crisis Resources

If you or someone you know is in crisis, please contact:

**🇺🇸 United States:**
- National Suicide Prevention Lifeline: 988 or 1-800-273-8255
- Crisis Text Line: Text HOME to 741741

**🇬🇧 United Kingdom:**
- Samaritans: 116 123

**🌍 International:**
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

**Online Resources:**
- Crisis Text Line: https://www.crisistextline.org/
- National Alliance on Mental Illness: https://www.nami.org/

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset contributors and researchers in mental health NLP
- Hugging Face for transformer models and libraries
- scikit-learn and pandas communities
- Mental health advocates and organizations

## 📧 Contact

For questions about this project:
- Open an issue on GitHub
- Check the documentation
- Review ethical guidelines before use

**Remember**: This tool is for research purposes only. Always prioritize professional mental health support over automated predictions.

---

**⚠️ Final Reminder**: This model should never replace professional mental health assessment, diagnosis, or treatment. It is a research tool designed to demonstrate NLP techniques for mental health text analysis.