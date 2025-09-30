"""
Evaluation module for mental health tweet classification models.

Provides comprehensive metrics, visualizations, and performance analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from datetime import datetime
import warnings

# Core ML libraries
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score,
        precision_recall_curve, roc_curve, average_precision_score
    )
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some evaluation features will be limited.")

# Visualization libraries  
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib/seaborn not available. Visualizations will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive plots will be limited.")

class ModelEvaluator:
    """
    Comprehensive model evaluation for mental health classification.
    
    Provides metrics, visualizations, and detailed performance analysis
    with focus on ethical considerations for sensitive classification tasks.
    """
    
    def __init__(self, 
                 class_names: Optional[List[str]] = None,
                 positive_class: Union[str, int] = 1,
                 save_dir: Optional[Path] = None):
        """
        Initialize evaluator.
        
        Args:
            class_names: Names for classes (e.g., ['Non-Depression', 'Depression'])
            positive_class: Positive class for binary classification
            save_dir: Directory to save evaluation results
        """
        self.class_names = class_names or ['Non-Depression', 'Depression']
        self.positive_class = positive_class
        self.save_dir = Path(save_dir) if save_dir else Path("evaluation_results")
        self.save_dir.mkdir(exist_ok=True)
        
        self.results_history = []
        
    def evaluate_model(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray, 
                      y_proba: Optional[np.ndarray] = None,
                      model_name: str = "Unknown",
                      dataset_name: str = "Unknown") -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            y_proba: Prediction probabilities (for binary classification)
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing all evaluation metrics and metadata
        """
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for evaluation")
            
        results = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(y_true),
            'n_classes': len(np.unique(y_true)),
            'class_distribution': self._get_class_distribution(y_true)
        }
        
        # Basic metrics
        results.update(self._calculate_basic_metrics(y_true, y_pred))
        
        # Advanced metrics (if probabilities available)
        if y_proba is not None:
            results.update(self._calculate_probabilistic_metrics(y_true, y_proba))
        
        # Per-class metrics
        results['per_class_metrics'] = self._calculate_per_class_metrics(y_true, y_pred)
        
        # Confusion matrix
        results['confusion_matrix'] = self._calculate_confusion_matrix(y_true, y_pred)
        
        # Fairness and bias metrics
        results['fairness_analysis'] = self._analyze_fairness(y_true, y_pred, y_proba)
        
        # Save results
        self._save_results(results)
        self.results_history.append(results)
        
        return results
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Binary classification specific metrics
        if len(np.unique(y_true)) == 2:
            metrics.update({
                'precision_positive': precision_score(y_true, y_pred, pos_label=self.positive_class, zero_division=0),
                'recall_positive': recall_score(y_true, y_pred, pos_label=self.positive_class, zero_division=0),
                'f1_positive': f1_score(y_true, y_pred, pos_label=self.positive_class, zero_division=0),
                'specificity': self._calculate_specificity(y_true, y_pred)
            })
        
        return metrics
    
    def _calculate_probabilistic_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate metrics that require prediction probabilities."""
        
        metrics = {}
        
        # Binary classification metrics
        if len(np.unique(y_true)) == 2:
            # Extract positive class probabilities
            if y_proba.ndim == 2:
                pos_proba = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba[:, self.positive_class]
            else:
                pos_proba = y_proba
            
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, pos_proba)
                metrics['average_precision'] = average_precision_score(y_true, pos_proba)
            except ValueError as e:
                warnings.warn(f"Could not calculate ROC AUC or Average Precision: {e}")
        
        return metrics
    
    def _calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each class."""
        
        try:
            report = classification_report(y_true, y_pred, 
                                         target_names=self.class_names,
                                         output_dict=True, 
                                         zero_division=0)
            
            # Extract per-class metrics
            per_class = {}
            for i, class_name in enumerate(self.class_names):
                if class_name in report:
                    per_class[class_name] = report[class_name]
                elif str(i) in report:
                    per_class[class_name] = report[str(i)]
            
            return per_class
            
        except Exception as e:
            warnings.warn(f"Could not calculate per-class metrics: {e}")
            return {}
    
    def _calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate and format confusion matrix."""
        
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'matrix': cm.tolist(),
            'normalized': (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).tolist(),
            'class_names': self.class_names
        }
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.0
    
    def _get_class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """Get class distribution in the dataset."""
        unique, counts = np.unique(y, return_counts=True)
        distribution = {}
        
        for i, count in enumerate(counts):
            class_name = self.class_names[unique[i]] if unique[i] < len(self.class_names) else f"Class_{unique[i]}"
            distribution[class_name] = int(count)
        
        return distribution
    
    def _analyze_fairness(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze potential fairness issues in mental health classification.
        
        Note: This is a simplified analysis. Proper fairness evaluation
        requires demographic information and domain expertise.
        """
        
        fairness_analysis = {
            'class_balance': self._analyze_class_balance(y_true),
            'prediction_bias': self._analyze_prediction_bias(y_true, y_pred),
            'recommendations': []
        }
        
        # Add recommendations based on analysis
        if fairness_analysis['class_balance']['imbalance_ratio'] > 3:
            fairness_analysis['recommendations'].append(
                "Significant class imbalance detected. Consider balancing techniques or cost-sensitive learning."
            )
        
        if fairness_analysis['prediction_bias']['bias_magnitude'] > 0.1:
            fairness_analysis['recommendations'].append(
                "Potential prediction bias detected. Review model training and validation procedures."
            )
        
        # General ethical recommendations
        fairness_analysis['recommendations'].extend([
            "Always validate model performance across diverse populations",
            "Consider cultural and linguistic factors in mental health classification",
            "Ensure proper human oversight in any deployment scenario",
            "Regular revalidation needed as language and social context evolve"
        ])
        
        return fairness_analysis
    
    def _analyze_class_balance(self, y_true: np.ndarray) -> Dict[str, float]:
        """Analyze class balance in the dataset."""
        unique, counts = np.unique(y_true, return_counts=True)
        
        if len(counts) >= 2:
            imbalance_ratio = max(counts) / min(counts)
            minority_class_pct = min(counts) / sum(counts) * 100
        else:
            imbalance_ratio = 1.0
            minority_class_pct = 100.0
        
        return {
            'imbalance_ratio': imbalance_ratio,
            'minority_class_percentage': minority_class_pct,
            'is_balanced': imbalance_ratio < 2.0
        }
    
    def _analyze_prediction_bias(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Analyze potential prediction bias."""
        
        # Calculate prediction rates for each class
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        
        if len(unique_true) >= 2 and len(unique_pred) >= 2:
            true_positive_rate = np.mean(y_true == self.positive_class)
            pred_positive_rate = np.mean(y_pred == self.positive_class)
            bias_magnitude = abs(true_positive_rate - pred_positive_rate)
        else:
            bias_magnitude = 0.0
        
        return {
            'bias_magnitude': bias_magnitude,
            'true_positive_rate': float(np.mean(y_true == self.positive_class)),
            'predicted_positive_rate': float(np.mean(y_pred == self.positive_class))
        }
    
    def cross_validate_model(self,
                           model: Any,
                           X: np.ndarray,
                           y: np.ndarray,
                           cv_folds: int = 5,
                           scoring: str = 'f1_macro') -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model: Trained model with fit/predict methods
            X: Features
            y: Labels
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for cross-validation")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        try:
            cv_scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
            
            results = {
                'cv_scores': cv_scores.tolist(),
                'mean_score': float(cv_scores.mean()),
                'std_score': float(cv_scores.std()),
                'scoring_metric': scoring,
                'cv_folds': cv_folds
            }
            
            return results
            
        except Exception as e:
            warnings.warn(f"Cross-validation failed: {e}")
            return {'error': str(e)}
    
    def create_evaluation_report(self, results: Dict[str, Any]) -> str:
        """
        Create a comprehensive evaluation report.
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Formatted evaluation report as string
        """
        
        report_lines = [
            "=" * 80,
            f"MENTAL HEALTH TWEET CLASSIFIER - EVALUATION REPORT",
            "=" * 80,
            "",
            f"Model: {results.get('model_name', 'Unknown')}",
            f"Dataset: {results.get('dataset_name', 'Unknown')}",
            f"Evaluation Time: {results.get('timestamp', 'Unknown')}",
            f"Total Samples: {results.get('n_samples', 0):,}",
            f"Number of Classes: {results.get('n_classes', 0)}",
            "",
            "CLASS DISTRIBUTION:",
            "-" * 40
        ]
        
        # Class distribution
        class_dist = results.get('class_distribution', {})
        for class_name, count in class_dist.items():
            pct = (count / results.get('n_samples', 1)) * 100
            report_lines.append(f"  {class_name}: {count:,} ({pct:.1f}%)")
        
        report_lines.extend([
            "",
            "PERFORMANCE METRICS:",
            "-" * 40,
            f"  Accuracy: {results.get('accuracy', 0):.4f}",
            f"  Precision (Macro): {results.get('precision_macro', 0):.4f}",
            f"  Recall (Macro): {results.get('recall_macro', 0):.4f}",
            f"  F1-Score (Macro): {results.get('f1_macro', 0):.4f}"
        ])
        
        # Binary classification metrics
        if 'roc_auc' in results:
            report_lines.extend([
                f"  ROC AUC: {results.get('roc_auc', 0):.4f}",
                f"  Average Precision: {results.get('average_precision', 0):.4f}",
                f"  Specificity: {results.get('specificity', 0):.4f}"
            ])
        
        # Per-class metrics
        per_class = results.get('per_class_metrics', {})
        if per_class:
            report_lines.extend([
                "",
                "PER-CLASS METRICS:",
                "-" * 40
            ])
            
            for class_name, metrics in per_class.items():
                if isinstance(metrics, dict):
                    report_lines.extend([
                        f"  {class_name}:",
                        f"    Precision: {metrics.get('precision', 0):.4f}",
                        f"    Recall: {metrics.get('recall', 0):.4f}",
                        f"    F1-Score: {metrics.get('f1-score', 0):.4f}",
                        f"    Support: {metrics.get('support', 0)}"
                    ])
        
        # Fairness analysis
        fairness = results.get('fairness_analysis', {})
        if fairness:
            report_lines.extend([
                "",
                "FAIRNESS ANALYSIS:",
                "-" * 40
            ])
            
            class_balance = fairness.get('class_balance', {})
            report_lines.extend([
                f"  Imbalance Ratio: {class_balance.get('imbalance_ratio', 0):.2f}",
                f"  Minority Class %: {class_balance.get('minority_class_percentage', 0):.1f}%",
                f"  Is Balanced: {class_balance.get('is_balanced', False)}"
            ])
            
            recommendations = fairness.get('recommendations', [])
            if recommendations:
                report_lines.extend([
                    "",
                    "RECOMMENDATIONS:",
                    "-" * 40
                ])
                for i, rec in enumerate(recommendations, 1):
                    report_lines.append(f"  {i}. {rec}")
        
        report_lines.extend([
            "",
            "ETHICAL CONSIDERATIONS:",
            "-" * 40,
            "  • This model is for research purposes only",
            "  • Not intended for clinical diagnosis or medical decisions", 
            "  • Requires human oversight and professional validation",
            "  • Regular revalidation needed as data/context evolves",
            "  • Consider cultural and demographic biases in deployment",
            "",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{results.get('model_name', 'unknown')}_{timestamp}.json"
        
        filepath = self.save_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Evaluation results saved to: {filepath}")
            
        except Exception as e:
            warnings.warn(f"Could not save results: {e}")
    
    def visualize_results(self, results: Dict[str, Any], save_plots: bool = True) -> Dict[str, Any]:
        """
        Create visualizations for evaluation results.
        
        Args:
            results: Evaluation results
            save_plots: Whether to save plots to file
            
        Returns:
            Dictionary of plot objects/paths
        """
        
        plots = {}
        
        if MATPLOTLIB_AVAILABLE:
            plots.update(self._create_matplotlib_plots(results, save_plots))
        
        if PLOTLY_AVAILABLE:
            plots.update(self._create_plotly_plots(results, save_plots))
        
        return plots
    
    def _create_matplotlib_plots(self, results: Dict[str, Any], save_plots: bool) -> Dict[str, Any]:
        """Create matplotlib visualizations."""
        
        plots = {}
        
        # Set style
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'available') and 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Confusion Matrix
        cm_data = results.get('confusion_matrix', {})
        if cm_data and 'matrix' in cm_data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Raw confusion matrix
            cm = np.array(cm_data['matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=cm_data.get('class_names', []), 
                       yticklabels=cm_data.get('class_names', []), ax=ax1)
            ax1.set_title('Confusion Matrix (Raw Counts)')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
            
            # Normalized confusion matrix
            cm_norm = np.array(cm_data['normalized'])
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=cm_data.get('class_names', []),
                       yticklabels=cm_data.get('class_names', []), ax=ax2)
            ax2.set_title('Confusion Matrix (Normalized)')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            
            plt.tight_layout()
            
            if save_plots:
                plot_path = self.save_dir / f"confusion_matrix_{results.get('model_name', 'unknown')}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plots['confusion_matrix_path'] = str(plot_path)
            
            plots['confusion_matrix_fig'] = fig
        
        return plots
    
    def _create_plotly_plots(self, results: Dict[str, Any], save_plots: bool) -> Dict[str, Any]:
        """Create plotly visualizations."""
        
        plots = {}
        
        # Interactive metrics dashboard
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Performance Metrics', 'Class Distribution', 
                              'Per-Class Performance', 'Model Summary'),
                specs=[[{"type": "bar"}, {"type": "pie"}],
                       [{"type": "bar"}, {"type": "table"}]]
            )
            
            # Performance metrics
            metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            metric_values = [results.get(m, 0) for m in metrics]
            
            fig.add_trace(
                go.Bar(x=metrics, y=metric_values, name="Metrics"),
                row=1, col=1
            )
            
            # Class distribution
            class_dist = results.get('class_distribution', {})
            if class_dist:
                fig.add_trace(
                    go.Pie(labels=list(class_dist.keys()), 
                          values=list(class_dist.values()), 
                          name="Class Distribution"),
                    row=1, col=2
                )
            
            # Per-class metrics
            per_class = results.get('per_class_metrics', {})
            if per_class:
                classes = list(per_class.keys())
                f1_scores = [per_class[c].get('f1-score', 0) if isinstance(per_class[c], dict) else 0 
                           for c in classes]
                
                fig.add_trace(
                    go.Bar(x=classes, y=f1_scores, name="F1-Score"),
                    row=2, col=1
                )
            
            fig.update_layout(height=800, showlegend=True, 
                            title_text=f"Evaluation Dashboard - {results.get('model_name', 'Unknown')}")
            
            if save_plots:
                plot_path = self.save_dir / f"dashboard_{results.get('model_name', 'unknown')}.html"
                fig.write_html(str(plot_path))
                plots['dashboard_path'] = str(plot_path)
            
            plots['dashboard_fig'] = fig
            
        except Exception as e:
            warnings.warn(f"Could not create plotly dashboard: {e}")
        
        return plots

def evaluate_mental_health_model(model: Any,
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                model_name: str = "Unknown",
                                save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        save_dir: Directory to save results
        
    Returns:
        Comprehensive evaluation results
    """
    
    evaluator = ModelEvaluator(save_dir=save_dir)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    y_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            pass
    
    # Evaluate
    results = evaluator.evaluate_model(
        y_test, y_pred, y_proba, 
        model_name=model_name,
        dataset_name="Test Set"
    )
    
    # Create report
    report = evaluator.create_evaluation_report(results)
    print(report)
    
    # Create visualizations
    if MATPLOTLIB_AVAILABLE or PLOTLY_AVAILABLE:
        plots = evaluator.visualize_results(results, save_plots=True)
        print(f"\nVisualizations created: {list(plots.keys())}")
    
    return results

if __name__ == "__main__":
    # Example usage
    print("Mental Health Tweet Classifier - Evaluation Module")
    print("This module provides comprehensive evaluation capabilities.")
    print("\nFeatures:")
    print("- Classification metrics (accuracy, precision, recall, F1)")
    print("- Confusion matrices and ROC analysis") 
    print("- Fairness and bias analysis")
    print("- Cross-validation support")
    print("- Interactive visualizations")
    print("- Ethical considerations and recommendations")
    print("\nUsage:")
    print("  from src.eval.evaluate import evaluate_mental_health_model")
    print("  results = evaluate_mental_health_model(model, X_test, y_test)")