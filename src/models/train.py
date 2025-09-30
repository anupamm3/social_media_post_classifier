"""
Training utilities and orchestration for mental health tweet classification models.

This module provides high-level training functions that coordinate data loading,
preprocessing, feature extraction, model training, and evaluation for both
baseline and transformer models.
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, asdict
import argparse
from datetime import datetime

# Import our modules
from baseline import BaselineModel, TransformerConfig
from transformer import TransformerModel
import sys
sys.path.append('..')
from src.data.load_data import DataLoader
from src.data.preprocess import preprocess_tweets

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Experiment info
    experiment_name: str = "mental_health_classification"
    description: str = "Depression vs non-depression tweet classification"
    
    # Data configuration
    data_config: Dict = None
    
    # Model configuration
    model_type: str = "baseline"  # "baseline" or "transformer"
    model_config: Dict = None
    
    # Training configuration
    random_state: int = 42
    cross_validation_folds: int = 5
    
    # Output configuration
    output_dir: str = "experiments"
    save_predictions: bool = True
    save_models: bool = True
    
    def __post_init__(self):
        """Set default configurations."""
        if self.data_config is None:
            self.data_config = {
                'use_clean': False,
                'test_size': 0.2,
                'preprocessing': {
                    'remove_urls': True,
                    'remove_mentions': False,
                    'expand_contractions': True,
                    'lowercase': True,
                    'min_length': 3
                }
            }
        
        if self.model_config is None:
            if self.model_type == "baseline":
                self.model_config = {
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
                }
            elif self.model_type == "transformer":
                self.model_config = {
                    'model_name': 'roberta-base',
                    'batch_size': 16,
                    'num_epochs': 3,
                    'learning_rate': 2e-5,
                    'max_length': 128
                }


class ExperimentTracker:
    """Track and log training experiments."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize experiment tracker."""
        self.config = config
        self.start_time = datetime.now()
        
        # Create experiment directory
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(config.output_dir) / f"{config.experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logs
        self.logs = {
            'config': asdict(config),
            'start_time': self.start_time.isoformat(),
            'stages': [],
            'metrics': {},
            'artifacts': []
        }
        
        logger.info(f"Experiment initialized: {self.experiment_dir}")
    
    def log_stage(self, stage: str, details: Dict = None):
        """Log a training stage."""
        stage_info = {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.logs['stages'].append(stage_info)
        logger.info(f"Stage completed: {stage}")
    
    def log_metrics(self, metrics: Dict, stage: str = "final"):
        """Log metrics for a stage."""
        self.logs['metrics'][stage] = metrics
        logger.info(f"Metrics logged for {stage}: {metrics}")
    
    def save_artifact(self, artifact_path: Path, description: str):
        """Register an artifact."""
        self.logs['artifacts'].append({
            'path': str(artifact_path.relative_to(self.experiment_dir)),
            'description': description,
            'timestamp': datetime.now().isoformat()
        })
    
    def finalize(self):
        """Finalize experiment and save logs."""
        self.logs['end_time'] = datetime.now().isoformat()
        self.logs['duration_seconds'] = (datetime.now() - self.start_time).total_seconds()
        
        # Save experiment log
        log_path = self.experiment_dir / "experiment_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.logs, f, indent=2, default=str)
        
        logger.info(f"Experiment completed: {self.experiment_dir}")
        logger.info(f"Duration: {self.logs['duration_seconds']:.1f} seconds")
        
        return log_path


class ModelTrainer:
    """High-level model training orchestrator."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize model trainer."""
        self.config = config
        self.tracker = ExperimentTracker(config)
        
        # Set random seeds for reproducibility
        np.random.seed(config.random_state)
        
        try:
            import torch
            torch.manual_seed(config.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config.random_state)
        except ImportError:
            pass
    
    def load_and_preprocess_data(self) -> Tuple[List[str], List[int]]:
        """Load and preprocess data according to configuration."""
        self.tracker.log_stage("data_loading_started")
        
        # Load raw data
        loader = DataLoader()
        texts, labels = loader.get_text_and_labels(
            use_clean=self.config.data_config['use_clean']
        )
        
        logger.info(f"Loaded {len(texts)} samples")
        
        # Preprocess if configured
        if 'preprocessing' in self.config.data_config:
            self.tracker.log_stage("preprocessing_started")
            
            cleaned_texts, filtered_labels, features_df = preprocess_tweets(
                texts, labels, self.config.data_config['preprocessing']
            )
            
            logger.info(f"After preprocessing: {len(cleaned_texts)} samples")
            
            # Save preprocessing stats
            preprocessing_stats = {
                'original_samples': len(texts),
                'filtered_samples': len(cleaned_texts),
                'filter_rate': 1 - (len(cleaned_texts) / len(texts))
            }
            
            self.tracker.log_stage("preprocessing_completed", preprocessing_stats)
            
            return cleaned_texts, filtered_labels
        
        self.tracker.log_stage("data_loading_completed")
        return texts, labels
    
    def train_baseline_model(self, texts: List[str], labels: List[int]) -> BaselineModel:
        """Train baseline model."""
        self.tracker.log_stage("baseline_training_started")
        
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=self.config.data_config.get('test_size', 0.2),
            random_state=self.config.random_state,
            stratify=labels
        )
        
        # Initialize model
        model = BaselineModel(
            model_type=self.config.model_config.get('type', 'logistic'),
            tfidf_params=self.config.model_config.get('tfidf_params', {}),
            model_params=self.config.model_config.get('model_params', {}),
            use_classical_features=self.config.model_config.get('use_classical_features', True),
            random_state=self.config.random_state
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Cross-validation
        self.tracker.log_stage("cross_validation_started")
        cv_results = model.cross_validate(X_train, y_train, cv=self.config.cross_validation_folds)
        self.tracker.log_metrics(cv_results, "cross_validation")
        
        # Test evaluation
        self.tracker.log_stage("test_evaluation_started")
        test_results = model.evaluate(X_test, y_test)
        self.tracker.log_metrics(test_results, "test")
        
        # Feature importance
        top_features = model.get_feature_importance(top_n=20)
        
        # Save model and results
        if self.config.save_models:
            model_dir = self.tracker.experiment_dir / "baseline_model"
            model.save(model_dir)
            self.tracker.save_artifact(model_dir, "Trained baseline model")
        
        # Save detailed results
        results = {
            'model_type': 'baseline',
            'config': self.config.model_config,
            'training_stats': model.training_stats,
            'cv_results': cv_results,
            'test_results': test_results,
            'top_features': top_features
        }
        
        results_path = self.tracker.experiment_dir / "baseline_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.tracker.save_artifact(results_path, "Baseline model results")
        
        self.tracker.log_stage("baseline_training_completed")
        
        return model
    
    def train_transformer_model(self, texts: List[str], labels: List[int]) -> TransformerModel:
        """Train transformer model."""
        try:
            from transformer import TransformerModel, TransformerConfig
        except ImportError:
            logger.error("Transformer dependencies not available")
            raise
        
        self.tracker.log_stage("transformer_training_started")
        
        # Create transformer config
        transformer_config = TransformerConfig(
            output_dir=str(self.tracker.experiment_dir / "transformer_model"),
            **self.config.model_config
        )
        
        # Initialize and train model
        model = TransformerModel(transformer_config)
        training_stats = model.train(texts, labels)
        
        # Log training metrics
        self.tracker.log_metrics(training_stats, "training")
        
        # Save detailed results
        results = {
            'model_type': 'transformer',
            'config': asdict(transformer_config),
            'training_stats': training_stats
        }
        
        results_path = self.tracker.experiment_dir / "transformer_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.tracker.save_artifact(results_path, "Transformer model results")
        
        if self.config.save_models:
            model_dir = Path(transformer_config.output_dir)
            self.tracker.save_artifact(model_dir, "Trained transformer model")
        
        self.tracker.log_stage("transformer_training_completed")
        
        return model
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete training experiment."""
        logger.info(f"Starting experiment: {self.config.experiment_name}")
        logger.info(f"Model type: {self.config.model_type}")
        
        try:
            # Load and preprocess data
            texts, labels = self.load_and_preprocess_data()
            
            # Train model based on type
            if self.config.model_type == "baseline":
                model = self.train_baseline_model(texts, labels)
            elif self.config.model_type == "transformer":
                model = self.train_transformer_model(texts, labels)
            else:
                raise ValueError(f"Unknown model type: {self.config.model_type}")
            
            # Save predictions if requested
            if self.config.save_predictions:
                self.tracker.log_stage("saving_predictions_started")
                
                # Make predictions on full dataset
                if self.config.model_type == "baseline":
                    predictions = model.predict(texts)
                    probabilities = model.predict_proba(texts)
                else:
                    predictions, probabilities = model.predict(texts)
                
                # Save predictions
                predictions_df = pd.DataFrame({
                    'text': texts,
                    'true_label': labels,
                    'predicted_label': predictions,
                    'probability_depression': probabilities[:, 1] if probabilities.ndim > 1 else probabilities,
                    'probability_non_depression': probabilities[:, 0] if probabilities.ndim > 1 else 1 - probabilities
                })
                
                predictions_path = self.tracker.experiment_dir / "predictions.csv"
                predictions_df.to_csv(predictions_path, index=False)
                self.tracker.save_artifact(predictions_path, "Model predictions on full dataset")
                
                self.tracker.log_stage("saving_predictions_completed")
            
            # Finalize experiment
            log_path = self.tracker.finalize()
            
            logger.info("✅ Experiment completed successfully!")
            return {
                'experiment_dir': str(self.tracker.experiment_dir),
                'log_path': str(log_path),
                'model': model
            }
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            self.tracker.log_stage("experiment_failed", {'error': str(e)})
            self.tracker.finalize()
            raise


def train_model_from_config(config_path: str) -> Dict[str, Any]:
    """
    Train model from configuration file.
    
    Args:
        config_path: Path to YAML or JSON configuration file
        
    Returns:
        Experiment results
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load configuration
    if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    
    # Create training config
    config = TrainingConfig(**config_dict)
    
    # Run training
    trainer = ModelTrainer(config)
    results = trainer.run_experiment()
    
    return results


def create_config_template(output_path: str = "training_config_template.yaml"):
    """Create a configuration template file."""
    template_config = TrainingConfig()
    
    # Convert to dict and add comments
    config_dict = asdict(template_config)
    
    yaml_content = """# Mental Health Tweet Classification Training Configuration
# 
# This file configures all aspects of model training including data preprocessing,
# model parameters, and experiment settings.

# Experiment metadata
experiment_name: "mental_health_classification"
description: "Depression vs non-depression tweet classification"

# Data configuration
data_config:
  use_clean: false  # Use preprocessed data or raw data
  test_size: 0.2   # Fraction of data for testing
  
  # Text preprocessing settings
  preprocessing:
    remove_urls: true
    remove_mentions: false
    expand_contractions: true
    lowercase: true
    min_length: 3

# Model configuration
model_type: "baseline"  # "baseline" or "transformer"

# Baseline model configuration (used if model_type is "baseline")
model_config:
  type: "logistic"  # "logistic" or "random_forest"
  use_classical_features: true
  
  # TF-IDF parameters
  tfidf_params:
    max_features: 5000
    ngram_range: [1, 2]
    min_df: 2
    max_df: 0.95
  
  # Model parameters
  model_params:
    C: 1.0
    class_weight: "balanced"

# Transformer model configuration (used if model_type is "transformer")
# model_config:
#   model_name: "roberta-base"
#   batch_size: 16
#   num_epochs: 3
#   learning_rate: 2e-5
#   max_length: 128
#   use_class_weights: true

# Training configuration
random_state: 42
cross_validation_folds: 5

# Output configuration
output_dir: "experiments"
save_predictions: true
save_models: true
"""
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"Configuration template saved to: {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Train mental health tweet classification models")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', type=str, required=True,
                             help="Path to configuration file")
    
    # Template command
    template_parser = subparsers.add_parser('template', help='Create configuration template')
    template_parser.add_argument('--output', type=str, default='training_config_template.yaml',
                                help="Output path for template")
    
    # Quick train commands
    quick_parser = subparsers.add_parser('quick-baseline', help='Quick baseline training')
    quick_parser.add_argument('--output-dir', type=str, default='experiments',
                             help="Output directory")
    
    quick_transformer_parser = subparsers.add_parser('quick-transformer', help='Quick transformer training')
    quick_transformer_parser.add_argument('--model-name', type=str, default='roberta-base',
                                         help="Transformer model name")
    quick_transformer_parser.add_argument('--batch-size', type=int, default=16,
                                         help="Batch size")
    quick_transformer_parser.add_argument('--epochs', type=int, default=3,
                                         help="Number of epochs")
    quick_transformer_parser.add_argument('--output-dir', type=str, default='experiments',
                                         help="Output directory")
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Train from config file
        results = train_model_from_config(args.config)
        logger.info(f"Training completed! Results saved to: {results['experiment_dir']}")
        
    elif args.command == 'template':
        # Create config template
        create_config_template(args.output)
        
    elif args.command == 'quick-baseline':
        # Quick baseline training
        config = TrainingConfig(
            model_type="baseline",
            output_dir=args.output_dir,
            experiment_name="quick_baseline"
        )
        
        trainer = ModelTrainer(config)
        results = trainer.run_experiment()
        logger.info(f"Baseline training completed! Results: {results['experiment_dir']}")
        
    elif args.command == 'quick-transformer':
        # Quick transformer training
        config = TrainingConfig(
            model_type="transformer",
            output_dir=args.output_dir,
            experiment_name="quick_transformer",
            model_config={
                'model_name': args.model_name,
                'batch_size': args.batch_size,
                'num_epochs': args.epochs
            }
        )
        
        trainer = ModelTrainer(config)
        results = trainer.run_experiment()
        logger.info(f"Transformer training completed! Results: {results['experiment_dir']}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()