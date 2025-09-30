"""
Auto-trainer for deployment: Trains the model if not available.
This ensures the model is available even in deployment environments.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path
import streamlit as st

def ensure_model_exists():
    """
    Ensures model files exist, trains if necessary.
    Returns True if model is available, False otherwise.
    """
    
    # Check if model files already exist
    model_dir = Path("models/baseline")
    model_path = model_dir / "logistic_model.joblib"
    vectorizer_path = model_dir / "tfidf_vectorizer.joblib"
    
    if model_path.exists() and vectorizer_path.exists():
        return True
    
    # If models don't exist, try to train them
    try:
        st.info("🤖 Model files not found. Training model automatically...")
        
        # Create models directory
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset files exist
        dataset_dir = Path("dataset")
        d_tweets_path = dataset_dir / "d_tweets.csv"
        non_d_tweets_path = dataset_dir / "non_d_tweets.csv"
        
        if not (d_tweets_path.exists() and non_d_tweets_path.exists()):
            st.error("❌ Dataset files not found. Cannot train model.")
            return False
        
        # Load and combine datasets
        st.info("📊 Loading training data...")
        df_d = pd.read_csv(d_tweets_path)
        df_nd = pd.read_csv(non_d_tweets_path)
        
        # Add labels
        df_d['label'] = 1
        df_nd['label'] = 0
        
        # Combine datasets
        df_combined = pd.concat([df_d, df_nd], ignore_index=True)
        
        # Prepare features and labels
        X = df_combined['tweet'].fillna('')
        y = df_combined['label']
        
        st.info("🔧 Training AI model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Train model
        model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save models
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.3f}")
        return True
        
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        return False

def get_model_info():
    """Get information about the trained model."""
    model_dir = Path("models/baseline")
    model_path = model_dir / "logistic_model.joblib"
    
    if model_path.exists():
        # Get file size
        size_mb = model_path.stat().st_size / (1024 * 1024)
        return {
            "available": True,
            "size_mb": size_mb,
            "path": str(model_path)
        }
    else:
        return {"available": False}

if __name__ == "__main__":
    # Test the auto-trainer
    print("Testing auto-trainer...")
    success = ensure_model_exists()
    if success:
        print("✅ Model is available!")
        info = get_model_info()
        print(f"Model size: {info['size_mb']:.2f} MB")
    else:
        print("❌ Failed to ensure model availability")