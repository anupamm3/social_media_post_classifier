"""
Baseline model for mental health tweet classification.

This module implements a simple but effective baseline using TF-IDF features
and traditional machine learning algorithms (Logistic Regression, Random Forest).
Includes proper cross-validation, class balancing, and model persistence.
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import argparse
import yaml

# Import our custom modules
import sys
sys.path.append('..')
from src.data.load_data import DataLoader
from src.data.preprocess import preprocess_tweets
from src.features.featurize import TFIDFFeaturizer, ClassicalFeaturizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModel:
    """Baseline model using TF-IDF + traditional ML algorithms."""
    
    def __init__(self, 
                 model_type: str = 'logistic',
                 tfidf_params: Dict = None,
                 model_params: Dict = None,
                 use_classical_features: bool = True,
                 random_state: int = 42):
        """
        Initialize baseline model.
        
        Args:
            model_type: 'logistic' or 'random_forest'
            tfidf_params: Parameters for TF-IDF vectorizer
            model_params: Parameters for the ML model
            use_classical_features: Whether to include classical NLP features
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.use_classical_features = use_classical_features
        self.random_state = random_state
        
        # Default parameters
        self.tfidf_params = tfidf_params or {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95,
            'sublinear_tf': True
        }
        
        # Initialize feature extractors
        self.tfidf_featurizer = TFIDFFeaturizer(**self.tfidf_params)
        if self.use_classical_features:
            self.classical_featurizer = ClassicalFeaturizer()
        
        # Initialize ML model
        if model_type == 'logistic':
            default_params = {
                'random_state': random_state,
                'class_weight': 'balanced',
                'max_iter': 1000,
                'C': 1.0
            }
            default_params.update(model_params or {})
            self.model = LogisticRegression(**default_params)
            
        elif model_type == 'random_forest':
            default_params = {
                'random_state': random_state,
                'class_weight': 'balanced',
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2
            }
            default_params.update(model_params or {})
            self.model = RandomForestClassifier(**default_params)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.fitted = False
        self.feature_names = []
        self.training_stats = {}
        
        logger.info(f"Baseline model initialized: {model_type}")
        logger.info(f"TF-IDF params: {self.tfidf_params}")
        logger.info(f"Model params: {self.model.get_params()}")
    
    def _extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract features from texts."""
        # TF-IDF features
        if not self.tfidf_featurizer.fitted:
            tfidf_features = self.tfidf_featurizer.fit_transform(texts)
        else:
            tfidf_features = self.tfidf_featurizer.transform(texts)
        
        features_list = [tfidf_features]
        feature_names = [f"tfidf_{name}" for name in self.tfidf_featurizer.get_feature_names()]
        
        # Classical features
        if self.use_classical_features:
            classical_df = self.classical_featurizer.extract_batch_features(texts)
            classical_features = classical_df.values
            features_list.append(classical_features)
            
            classical_names = [f"classical_{name}" for name in classical_df.columns]
            feature_names.extend(classical_names)
        
        # Combine features
        if len(features_list) > 1:
            combined_features = np.hstack(features_list)
        else:
            combined_features = features_list[0]
        
        self.feature_names = feature_names
        return combined_features
    
    def fit(self, texts: List[str], labels: List[int]) -> 'BaselineModel':
        """
        Fit the baseline model.
        
        Args:
            texts: List of training texts
            labels: List of training labels
            
        Returns:
            Self for chaining
        """
        logger.info(f"Training baseline model on {len(texts)} samples...")
        
        # Extract features
        X = self._extract_features(texts)
        y = np.array(labels)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Class distribution: {class_dist}")
        
        # Fit model
        self.model.fit(X, y)
        self.fitted = True
        
        # Calculate training statistics
        train_pred = self.model.predict(X)
        train_accuracy = accuracy_score(y, train_pred)
        
        self.training_stats = {
            'n_samples': len(texts),
            'n_features': X.shape[1],
            'class_distribution': class_dist,
            'train_accuracy': train_accuracy,
            'feature_names': self.feature_names
        }
        
        logger.info(f"Model fitted successfully!")
        logger.info(f"Training accuracy: {train_accuracy:.3f}")
        
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict labels for texts."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._extract_features(texts)
        return self.model.predict(X)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict class probabilities for texts."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._extract_features(texts)
        return self.model.predict_proba(X)
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            texts: List of test texts
            labels: List of test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {len(texts)} samples...")
        
        # Make predictions
        y_pred = self.predict(texts)
        y_proba = self.predict_proba(texts)
        y_true = np.array(labels)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba[:, 1])
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'accuracy': accuracy,
            'roc_auc': auc,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist()
        }
        
        logger.info(f"Evaluation complete:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  ROC-AUC: {auc:.3f}")
        
        return results
    
    def cross_validate(self, 
                      texts: List[str], 
                      labels: List[int], 
                      cv: int = 5,
                      scoring: List[str] = None) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            texts: List of texts
            labels: List of labels
            cv: Number of CV folds
            scoring: List of scoring metrics
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        # Extract features
        X = self._extract_features(texts)
        y = np.array(labels)
        
        # Default scoring
        if scoring is None:
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Stratified K-Fold for imbalanced data
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Perform cross-validation
        cv_results = {}
        for metric in scoring:
            scores = cross_val_score(self.model, X, y, cv=cv_splitter, scoring=metric)
            cv_results[metric] = {
                'scores': scores.tolist(),
                'mean': scores.mean(),
                'std': scores.std()
            }
            logger.info(f"{metric}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        return cv_results
    
    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get top important features."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        if self.model_type == 'logistic':
            # For logistic regression, use coefficients
            importance = np.abs(self.model.coef_[0])
        elif self.model_type == 'random_forest':
            # For random forest, use feature importances
            importance = self.model.feature_importances_
        
        # Get top features
        top_indices = np.argsort(importance)[::-1][:top_n]
        top_features = [(self.feature_names[i], importance[i]) for i in top_indices]
        
        return top_features
    
    def save(self, directory: str):
        """Save fitted model and components."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        if not self.fitted:
            raise ValueError("Cannot save unfitted model")
        
        # Save model
        joblib.dump(self.model, directory / "model.pkl")
        
        # Save TF-IDF vectorizer
        self.tfidf_featurizer.save(directory / "tfidf_vectorizer.pkl")
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'use_classical_features': self.use_classical_features,
            'random_state': self.random_state,
            'tfidf_params': self.tfidf_params,
            'model_params': self.model.get_params(),
            'fitted': self.fitted,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats
        }
        
        with open(directory / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Baseline model saved to {directory}")
    
    def load(self, directory: str):
        """Load fitted model and components."""
        directory = Path(directory)
        
        # Load metadata
        with open(directory / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.model_type = metadata['model_type']
        self.use_classical_features = metadata['use_classical_features']
        self.random_state = metadata['random_state']
        self.tfidf_params = metadata['tfidf_params']
        self.fitted = metadata['fitted']
        self.feature_names = metadata['feature_names']
        self.training_stats = metadata['training_stats']
        
        # Load model
        self.model = joblib.load(directory / "model.pkl")
        
        # Load TF-IDF vectorizer
        self.tfidf_featurizer = TFIDFFeaturizer(**self.tfidf_params)
        self.tfidf_featurizer.load(directory / "tfidf_vectorizer.pkl")
        
        # Reinitialize classical featurizer
        if self.use_classical_features:
            self.classical_featurizer = ClassicalFeaturizer()
        
        logger.info(f"Baseline model loaded from {directory}")


def hyperparameter_search(texts: List[str], 
                         labels: List[int],
                         model_type: str = 'logistic',
                         cv: int = 5) -> Dict[str, Any]:
    """
    Perform hyperparameter search for baseline model.
    
    Args:
        texts: Training texts
        labels: Training labels
        model_type: Type of model to search
        cv: Number of CV folds
        
    Returns:
        Best parameters and CV results
    """
    logger.info(f"Performing hyperparameter search for {model_type} model...")
    
    # Parameter grids
    if model_type == 'logistic':
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'class_weight': ['balanced', None],
            'max_iter': [1000, 2000]
        }
        base_model = LogisticRegression(random_state=42)
        
    elif model_type == 'random_forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'class_weight': ['balanced', None],
            'min_samples_split': [2, 5, 10]
        }
        base_model = RandomForestClassifier(random_state=42)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create baseline model for feature extraction
    baseline = BaselineModel(model_type=model_type, random_state=42)
    X = baseline._extract_features(texts)
    y = np.array(labels)
    
    # Grid search
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=cv, 
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
    
    return results


def train_baseline_model(config_path: str = None) -> BaselineModel:
    """
    Train baseline model with configuration.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Trained baseline model
    """
    # Default configuration
    default_config = {
        'data': {
            'use_clean': False,
            'test_size': 0.2,
            'random_state': 42
        },
        'preprocessing': {
            'remove_urls': True,
            'remove_mentions': False,
            'expand_contractions': True,
            'lowercase': True,
            'min_length': 3
        },
        'model': {
            'type': 'logistic',
            'tfidf_params': {
                'max_features': 5000,
                'ngram_range': [1, 2],
                'min_df': 2,
                'max_df': 0.95
            },
            'model_params': {
                'C': 1.0,
                'class_weight': 'balanced'
            },
            'use_classical_features': True
        },
        'evaluation': {
            'cv_folds': 5,
            'run_hyperparameter_search': False
        }
    }
    
    # Load configuration if provided
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Merge with defaults
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey not in config[key]:
                        config[key][subkey] = subvalue
    else:
        config = default_config
        if config_path:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
    
    logger.info("Configuration:")
    logger.info(json.dumps(config, indent=2))
    
    # Load data
    logger.info("Loading data...")
    loader = DataLoader()
    texts, labels = loader.get_text_and_labels(use_clean=config['data']['use_clean'])
    
    # Preprocess data
    logger.info("Preprocessing data...")
    cleaned_texts, filtered_labels, _ = preprocess_tweets(
        texts, labels, config['preprocessing']
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts, filtered_labels,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=filtered_labels
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Hyperparameter search (optional)
    if config['evaluation']['run_hyperparameter_search']:
        logger.info("Running hyperparameter search...")
        search_results = hyperparameter_search(
            X_train, y_train, 
            model_type=config['model']['type'],
            cv=config['evaluation']['cv_folds']
        )
        
        # Update model params with best found
        config['model']['model_params'].update(search_results['best_params'])
        logger.info(f"Updated model params: {config['model']['model_params']}")
    
    # Train model
    logger.info("Training baseline model...")
    model = BaselineModel(
        model_type=config['model']['type'],
        tfidf_params=config['model']['tfidf_params'],
        model_params=config['model']['model_params'],
        use_classical_features=config['model']['use_classical_features'],
        random_state=config['data']['random_state']
    )
    
    model.fit(X_train, y_train)
    
    # Cross-validation
    logger.info("Performing cross-validation...")
    cv_results = model.cross_validate(
        X_train, y_train, 
        cv=config['evaluation']['cv_folds']
    )
    
    # Test evaluation
    logger.info("Evaluating on test set...")
    test_results = model.evaluate(X_test, y_test)
    
    # Feature importance
    top_features = model.get_feature_importance(top_n=20)
    logger.info("Top 20 important features:")
    for i, (feature, importance) in enumerate(top_features, 1):
        logger.info(f"  {i:2d}. {feature:30s} {importance:.4f}")
    
    # Save results
    results = {
        'config': config,
        'cv_results': cv_results,
        'test_results': test_results,
        'top_features': top_features,
        'training_stats': model.training_stats
    }
    
    # Save model and results
    model_dir = Path("models/baseline")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(model_dir)
    
    with open(model_dir / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Model and results saved to {model_dir}")
    
    return model


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Train baseline model for mental health tweet classification")
    parser.add_argument('--config', type=str, help="Path to configuration YAML file")
    parser.add_argument('--model-type', type=str, choices=['logistic', 'random_forest'], 
                       default='logistic', help="Type of baseline model")
    parser.add_argument('--output-dir', type=str, default='models/baseline',
                       help="Directory to save trained model")
    
    args = parser.parse_args()
    
    # Train model
    model = train_baseline_model(args.config)
    
    logger.info("✅ Baseline model training complete!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("\\nTo use the model:")
    logger.info("  from src.models.baseline import BaselineModel")
    logger.info("  model = BaselineModel()")
    logger.info(f"  model.load('{args.output_dir}')")
    logger.info("  predictions = model.predict(texts)")


if __name__ == "__main__":
    main()