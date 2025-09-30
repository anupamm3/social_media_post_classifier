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
        """Load the trained model and vectorizer."""
        try:
            model_file = self.model_path / "logistic_model.joblib"
            vectorizer_file = self.model_path / "tfidf_vectorizer.joblib"
            
            if model_file.exists() and vectorizer_file.exists():
                self.model = joblib.load(model_file)
                self.vectorizer = joblib.load(vectorizer_file)
                return True
            else:
                print(f"Model files not found at {self.model_path}")
                return False
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, texts):
        """Make predictions on a list of texts."""
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model not loaded")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Transform texts
        X = self.vectorizer.transform(texts)
        
        # Get predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_proba(self, texts):
        """Get prediction probabilities."""
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model not loaded")
            
        if isinstance(texts, str):
            texts = [texts]
        
        # Transform texts
        X = self.vectorizer.transform(texts)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X)
        
        return probabilities
    
    def is_loaded(self):
        """Check if model is loaded."""
        return self.model is not None and self.vectorizer is not None