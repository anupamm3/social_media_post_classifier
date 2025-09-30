"""
Transformer-based model for mental health tweet classification.

This module implements fine-tuning of pre-trained transformer models (BERT, RoBERTa, etc.)
using Hugging Face transformers and datasets libraries. Includes proper training loops,
GPU optimization, and model persistence.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from dataclasses import dataclass

# Hugging Face imports with error handling
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback,
        DataCollatorWithPadding, get_linear_schedule_with_warmup
    )
    from datasets import Dataset
    import torch.optim as optim
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers library not available. Install with: pip install transformers datasets torch")

# Import our custom modules
import sys
sys.path.append('..')
from src.data.load_data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for transformer model training."""
    
    # Model settings
    model_name: str = "roberta-base"
    num_labels: int = 2
    max_length: int = 128
    
    # Training settings
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Data settings
    test_size: float = 0.2
    validation_size: float = 0.1
    
    # Optimization settings
    use_class_weights: bool = True
    gradient_accumulation_steps: int = 1
    fp16: bool = False  # Enable for GPU memory efficiency
    dataloader_num_workers: int = 0
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 2
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    
    # Output
    output_dir: str = "models/transformer"
    save_total_limit: int = 2
    
    # Random seed
    seed: int = 42


class TransformerDataset:
    """Dataset preparation for transformer models."""
    
    def __init__(self, tokenizer, max_length: int = 128):
        """
        Initialize dataset handler.
        
        Args:
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def tokenize_function(self, examples):
        """Tokenize texts for transformer input."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def prepare_dataset(self, texts: List[str], labels: List[int]) -> Dataset:
        """
        Prepare dataset from texts and labels.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            
        Returns:
            Hugging Face Dataset object
        """
        # Create dataset
        dataset = Dataset.from_dict({
            "text": texts,
            "label": labels
        })
        
        # Tokenize
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        return tokenized_dataset


class WeightedTrainer(Trainer):
    """Custom Trainer with class weighting for imbalanced datasets."""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with class weighting."""
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute weighted loss
        if self.class_weights is not None:
            # Convert class weights to tensor
            if isinstance(self.class_weights, dict):
                weights = torch.tensor([self.class_weights[i] for i in range(len(self.class_weights))], 
                                     dtype=torch.float, device=logits.device)
            else:
                weights = torch.tensor(self.class_weights, dtype=torch.float, device=logits.device)
            
            # Create weighted loss function
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            # Standard loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


class TransformerModel:
    """Transformer model for mental health tweet classification."""
    
    def __init__(self, config: TransformerConfig = None):
        """
        Initialize transformer model.
        
        Args:
            config: Model configuration
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required. Install with: pip install transformers datasets torch")
        
        self.config = config or TransformerConfig()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            return_dict=True
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Dataset handler
        self.dataset_handler = TransformerDataset(self.tokenizer, self.config.max_length)
        
        # Training components
        self.trainer = None
        self.training_stats = {}
        
        logger.info(f"Transformer model initialized: {self.config.model_name}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        # AUC requires probabilities
        try:
            probabilities = torch.softmax(torch.tensor(eval_pred[0]), dim=1)[:, 1].numpy()
            auc = roc_auc_score(labels, probabilities)
        except:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def prepare_data(self, 
                    texts: List[str], 
                    labels: List[int]) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare train, validation, and test datasets.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        from sklearn.model_selection import train_test_split
        
        # First split: separate test set
        texts_temp, texts_test, labels_temp, labels_test = train_test_split(
            texts, labels,
            test_size=self.config.test_size,
            random_state=self.config.seed,
            stratify=labels
        )
        
        # Second split: separate validation from training
        texts_train, texts_val, labels_train, labels_val = train_test_split(
            texts_temp, labels_temp,
            test_size=self.config.validation_size / (1 - self.config.test_size),
            random_state=self.config.seed,
            stratify=labels_temp
        )
        
        logger.info(f"Dataset splits:")
        logger.info(f"  Training: {len(texts_train)} samples")
        logger.info(f"  Validation: {len(texts_val)} samples")
        logger.info(f"  Test: {len(texts_test)} samples")
        
        # Create datasets
        train_dataset = self.dataset_handler.prepare_dataset(texts_train, labels_train)
        val_dataset = self.dataset_handler.prepare_dataset(texts_val, labels_val)
        test_dataset = self.dataset_handler.prepare_dataset(texts_test, labels_test)
        
        return train_dataset, val_dataset, test_dataset
    
    def calculate_class_weights(self, labels: List[int]) -> Dict[int, float]:
        """Calculate class weights for imbalanced dataset."""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=labels
        )
        
        weight_dict = {i: weight for i, weight in zip(classes, class_weights)}
        logger.info(f"Class weights: {weight_dict}")
        
        return weight_dict
    
    def train(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """
        Train the transformer model.
        
        Args:
            texts: List of training texts
            labels: List of training labels
            
        Returns:
            Training statistics and metrics
        """
        logger.info(f"Training transformer model on {len(texts)} samples...")
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset = self.prepare_data(texts, labels)
        
        # Calculate class weights
        class_weights = None
        if self.config.use_class_weights:
            class_weights = self.calculate_class_weights(labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            learning_rate=self.config.learning_rate,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            seed=self.config.seed,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )
        
        # Callbacks
        callbacks = []
        if self.config.early_stopping:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience))
        
        # Initialize trainer
        if class_weights:
            self.trainer = WeightedTrainer(
                class_weights=class_weights,
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
                callbacks=callbacks
            )
        else:
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
                callbacks=callbacks
            )
        
        # Train model
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Final evaluation on test set
        logger.info("Evaluating on test set...")
        test_results = self.trainer.evaluate(eval_dataset=test_dataset)
        
        # Compile training statistics
        self.training_stats = {
            'train_runtime': train_result.metrics['train_runtime'],
            'train_samples_per_second': train_result.metrics['train_samples_per_second'],
            'train_loss': train_result.metrics['train_loss'],
            'eval_loss': test_results['eval_loss'],
            'eval_accuracy': test_results['eval_accuracy'],
            'eval_precision': test_results['eval_precision'],
            'eval_recall': test_results['eval_recall'],
            'eval_f1': test_results['eval_f1'],
            'eval_auc': test_results['eval_auc'],
            'total_training_steps': train_result.global_step,
            'config': self.config.__dict__
        }
        
        # Save training stats
        with open(Path(self.config.output_dir) / "training_stats.json", 'w') as f:
            json.dump(self.training_stats, f, indent=2, default=str)
        
        logger.info("Training complete!")
        logger.info(f"Final test accuracy: {test_results['eval_accuracy']:.3f}")
        logger.info(f"Final test F1: {test_results['eval_f1']:.3f}")
        logger.info(f"Model saved to: {self.config.output_dir}")
        
        return self.training_stats
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on texts.
        
        Args:
            texts: List of texts to predict
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.trainer is None:
            raise ValueError("Model must be trained or loaded first")
        
        # Prepare dataset
        predict_dataset = self.dataset_handler.prepare_dataset(texts, [0] * len(texts))  # Dummy labels
        
        # Make predictions
        predictions = self.trainer.predict(predict_dataset)
        
        # Extract predictions and probabilities
        logits = predictions.predictions
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        preds = np.argmax(logits, axis=1)
        
        return preds, probs
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            texts: List of test texts
            labels: List of test labels
            
        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Model must be trained or loaded first")
        
        # Prepare dataset
        eval_dataset = self.dataset_handler.prepare_dataset(texts, labels)
        
        # Evaluate
        results = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        return results
    
    @classmethod
    def load_model(cls, model_path: str) -> 'TransformerModel':
        """
        Load a trained transformer model.
        
        Args:
            model_path: Path to saved model directory
            
        Returns:
            Loaded TransformerModel instance
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")
        
        model_path = Path(model_path)
        
        # Load config if exists
        config_path = model_path / "training_stats.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                training_stats = json.load(f)
                config_dict = training_stats.get('config', {})
                config = TransformerConfig(**config_dict)
        else:
            config = TransformerConfig()
            config.model_name = str(model_path)
        
        # Create instance
        instance = cls(config)
        
        # Load model and tokenizer
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        instance.model.to(instance.device)
        
        # Create dummy trainer for predictions
        training_args = TrainingArguments(
            output_dir=str(model_path),
            per_device_eval_batch_size=config.batch_size
        )
        
        instance.trainer = Trainer(
            model=instance.model,
            args=training_args,
            tokenizer=instance.tokenizer,
            compute_metrics=instance.compute_metrics
        )
        
        logger.info(f"Model loaded from {model_path}")
        return instance


def train_transformer_model(config_path: str = None) -> TransformerModel:
    """
    Train transformer model with configuration.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Trained TransformerModel
    """
    # Load configuration
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = TransformerConfig(**config_dict)
    else:
        config = TransformerConfig()
        if config_path:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
    
    logger.info("Transformer configuration:")
    logger.info(json.dumps(config.__dict__, indent=2))
    
    # Load data
    logger.info("Loading data...")
    loader = DataLoader()
    texts, labels = loader.get_text_and_labels(use_clean=False)
    
    # Initialize and train model
    model = TransformerModel(config)
    training_stats = model.train(texts, labels)
    
    logger.info("✅ Transformer model training complete!")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train transformer model for mental health tweet classification")
    parser.add_argument('--config', type=str, help="Path to configuration JSON file")
    parser.add_argument('--model-name', type=str, default='roberta-base', 
                       help="Hugging Face model name")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs")
    parser.add_argument('--learning-rate', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--output-dir', type=str, default='models/transformer',
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Create config from args if no config file
    if not args.config:
        config = TransformerConfig(
            model_name=args.model_name,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir
        )
        
        # Initialize and train
        model = TransformerModel(config)
        
        # Load data
        loader = DataLoader()
        texts, labels = loader.get_text_and_labels(use_clean=False)
        
        # Train
        model.train(texts, labels)
    else:
        # Use config file
        train_transformer_model(args.config)
    
    print("\\n✅ Training complete!")
    print(f"Model saved to: {args.output_dir}")