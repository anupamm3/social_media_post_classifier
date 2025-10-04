"""
Simple model wrapper for the Streamlit app.
"""

import joblib
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SimpleModel:
    """Simple model wrapper for demo purposes."""
    
    def __init__(self, model_path="models/baseline"):
        """Initialize the model."""
        self.model = None
        self.vectorizer = None
        self.model_path = Path(model_path)
        self.load_model()
    
    def load_model(self):
        """Load the trained model, vectorizer, and optional components."""
        try:
            model_file = self.model_path / "logistic_model.joblib"
            vectorizer_file = self.model_path / "tfidf_vectorizer.joblib"
            
            # First, try to ensure model exists (auto-train if needed)
            if not (model_file.exists() and vectorizer_file.exists()):
                try:
                    from auto_train import ensure_model_exists
                    if ensure_model_exists():
                        pass  # Model was created successfully
                    else:
                        return False
                except ImportError:
                    return False
            
            if model_file.exists() and vectorizer_file.exists():
                self.model = joblib.load(model_file)
                self.vectorizer = joblib.load(vectorizer_file)
                
                # Load optimized components if available
                feature_selector_file = self.model_path / "feature_selector.joblib"
                text_preprocessor_file = self.model_path / "text_preprocessor.pkl"
                
                self.feature_selector = None
                self.text_preprocessor = None
                
                if feature_selector_file.exists():
                    self.feature_selector = joblib.load(feature_selector_file)
                    
                if text_preprocessor_file.exists():
                    import pickle
                    with open(text_preprocessor_file, 'rb') as f:
                        self.text_preprocessor = pickle.load(f)
                return True
            else:
                print(f"Model files not found at {self.model_path}")
                return False
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _preprocess_texts(self, texts):
        """Apply enhanced text preprocessing if available."""
        if self.text_preprocessor is not None:
            return [self.text_preprocessor(text) for text in texts]
        else:
            # Fallback to basic preprocessing
            import re
            processed = []
            for text in texts:
                if text is None or text == '':
                    processed.append('')
                    continue
                text = str(text).lower()
                text = re.sub(r'http\S+|www\S+|@\w+', '', text)
                text = re.sub(r'#(\w+)', r'\1', text)
                text = re.sub(r'\s+', ' ', text).strip()
                processed.append(text)
            return processed
    
    def predict(self, texts):
        """Make predictions on a list of texts."""
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model not loaded")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Apply enhanced preprocessing
        texts_processed = self._preprocess_texts(texts)
        
        # Transform texts with TF-IDF
        X = self.vectorizer.transform(texts_processed)
        
        # Apply feature selection if available
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_proba(self, texts):
        """Get prediction probabilities."""
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model not loaded")
            
        if isinstance(texts, str):
            texts = [texts]
        
        # Apply enhanced preprocessing
        texts_processed = self._preprocess_texts(texts)
        
        # Transform texts with TF-IDF
        X = self.vectorizer.transform(texts_processed)
        
        # Apply feature selection if available
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X)
        
        return probabilities
    
    def is_loaded(self):
        """Check if model is loaded."""
        return self.model is not None and self.vectorizer is not None