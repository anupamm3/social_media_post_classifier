"""
Model explainability module for mental health tweet classification.

Provides interpretability tools including SHAP, LIME, and attention visualization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import json
from datetime import datetime

# Core visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib/seaborn not available. Visualizations will be limited.")

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. SHAP explanations will be disabled.")

# LIME for local interpretability
try:
    import lime
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. LIME explanations will be disabled.")

# Text processing
try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK not available. Some text analysis features limited.")

class ModelExplainer:
    """
    Comprehensive model explainability for mental health classification.
    
    Provides multiple interpretability methods with focus on responsible
    AI practices for sensitive applications.
    """
    
    def __init__(self, 
                 model: Any,
                 feature_names: Optional[List[str]] = None,
                 class_names: Optional[List[str]] = None,
                 save_dir: Optional[str] = None):
        """
        Initialize explainer.
        
        Args:
            model: Trained model to explain
            feature_names: Names of input features
            class_names: Names of output classes
            save_dir: Directory to save explanations
        """
        self.model = model
        self.feature_names = feature_names or []
        self.class_names = class_names or ['Non-Depression', 'Depression']
        self.save_dir = Path(save_dir) if save_dir else Path("explanations")
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        self._setup_explainers()
    
    def _setup_explainers(self):
        """Initialize SHAP and LIME explainers if available."""
        
        # Setup SHAP
        if SHAP_AVAILABLE:
            try:
                # Try to create appropriate SHAP explainer
                if hasattr(self.model, 'predict_proba'):
                    self.shap_explainer = shap.Explainer(self.model.predict_proba)
                else:
                    self.shap_explainer = shap.Explainer(self.model.predict)
            except Exception as e:
                warnings.warn(f"Could not initialize SHAP explainer: {e}")
        
        # Setup LIME
        if LIME_AVAILABLE:
            try:
                self.lime_explainer = LimeTextExplainer(
                    class_names=self.class_names,
                    feature_selection='auto',
                    discretize_continuous=True
                )
            except Exception as e:
                warnings.warn(f"Could not initialize LIME explainer: {e}")
    
    def explain_prediction(self,
                          text_input: Union[str, List[str]],
                          method: str = "all",
                          max_features: int = 20) -> Dict[str, Any]:
        """
        Generate explanations for model predictions.
        
        Args:
            text_input: Text or list of texts to explain
            method: Explanation method ('shap', 'lime', 'feature_importance', 'all')
            max_features: Maximum number of features to show
            
        Returns:
            Dictionary containing explanations
        """
        
        if isinstance(text_input, str):
            text_input = [text_input]
        
        explanations = {
            'input_texts': text_input,
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'explanations': []
        }
        
        for i, text in enumerate(text_input):
            text_explanation = {
                'text_id': i,
                'original_text': text,
                'prediction': self._get_prediction(text),
                'methods': {}
            }
            
            # SHAP explanations
            if method in ['shap', 'all'] and self.shap_explainer is not None:
                try:
                    shap_exp = self._explain_with_shap(text, max_features)
                    text_explanation['methods']['shap'] = shap_exp
                except Exception as e:
                    warnings.warn(f"SHAP explanation failed: {e}")
            
            # LIME explanations
            if method in ['lime', 'all'] and self.lime_explainer is not None:
                try:
                    lime_exp = self._explain_with_lime(text, max_features)
                    text_explanation['methods']['lime'] = lime_exp
                except Exception as e:
                    warnings.warn(f"LIME explanation failed: {e}")
            
            # Feature importance (for traditional ML models)
            if method in ['feature_importance', 'all']:
                try:
                    feat_imp = self._explain_feature_importance(text, max_features)
                    text_explanation['methods']['feature_importance'] = feat_imp
                except Exception as e:
                    warnings.warn(f"Feature importance explanation failed: {e}")
            
            explanations['explanations'].append(text_explanation)
        
        # Save explanations
        self._save_explanations(explanations)
        
        return explanations
    
    def _get_prediction(self, text: str) -> Dict[str, Any]:
        """Get model prediction for a single text."""
        
        try:
            # Get prediction
            pred = self.model.predict([text])[0]
            pred_class = self.class_names[pred] if pred < len(self.class_names) else f"Class_{pred}"
            
            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba([text])[0]
                probabilities = {
                    self.class_names[i] if i < len(self.class_names) else f"Class_{i}": float(prob)
                    for i, prob in enumerate(proba)
                }
            
            return {
                'predicted_class': pred_class,
                'predicted_label': int(pred),
                'probabilities': probabilities,
                'confidence': float(max(proba)) if probabilities else None
            }
            
        except Exception as e:
            warnings.warn(f"Could not get prediction: {e}")
            return {'error': str(e)}
    
    def _explain_with_shap(self, text: str, max_features: int) -> Dict[str, Any]:
        """Generate SHAP explanation for text."""
        
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return {'error': 'SHAP not available'}
        
        try:
            # Get SHAP values
            shap_values = self.shap_explainer([text])
            
            # Extract feature contributions
            if hasattr(shap_values, 'values'):
                values = shap_values.values[0]  # First (and only) instance
                if values.ndim > 1:
                    values = values[:, 1]  # Positive class for binary classification
            else:
                values = shap_values[0]
            
            # Get feature names (or indices if not available)
            if self.feature_names and len(self.feature_names) == len(values):
                features = self.feature_names
            else:
                features = [f"feature_{i}" for i in range(len(values))]
            
            # Create feature importance pairs
            feature_importance = list(zip(features, values))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            feature_importance = feature_importance[:max_features]
            
            return {
                'method': 'shap',
                'feature_importance': [
                    {'feature': feat, 'importance': float(imp)}
                    for feat, imp in feature_importance
                ],
                'base_value': float(getattr(shap_values, 'base_values', [0])[0]),
                'explanation_text': self._generate_shap_text_explanation(feature_importance)
            }
            
        except Exception as e:
            return {'error': f'SHAP explanation failed: {e}'}
    
    def _explain_with_lime(self, text: str, max_features: int) -> Dict[str, Any]:
        """Generate LIME explanation for text."""
        
        if not LIME_AVAILABLE or self.lime_explainer is None:
            return {'error': 'LIME not available'}
        
        try:
            # Create prediction function for LIME
            def predict_fn(texts):
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(texts)
                else:
                    # Convert predictions to probabilities
                    preds = self.model.predict(texts)
                    n_classes = len(self.class_names)
                    proba = np.zeros((len(texts), n_classes))
                    for i, pred in enumerate(preds):
                        proba[i, pred] = 1.0
                    return proba
            
            # Generate LIME explanation
            exp = self.lime_explainer.explain_instance(
                text, predict_fn, num_features=max_features
            )
            
            # Extract feature importance
            feature_importance = [
                {'feature': feat, 'importance': float(imp)}
                for feat, imp in exp.as_list()
            ]
            
            return {
                'method': 'lime',
                'feature_importance': feature_importance,
                'prediction_probability': float(exp.predict_proba[1]),  # Positive class
                'explanation_text': self._generate_lime_text_explanation(feature_importance)
            }
            
        except Exception as e:
            return {'error': f'LIME explanation failed: {e}'}
    
    def _explain_feature_importance(self, text: str, max_features: int) -> Dict[str, Any]:
        """Generate feature importance explanation."""
        
        try:
            # This is a simplified feature importance for traditional ML models
            # For more sophisticated analysis, would need access to feature extraction pipeline
            
            # Get basic text statistics
            words = text.lower().split()
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Simple heuristics for mental health keywords
            depression_keywords = {
                'sad', 'depressed', 'hopeless', 'empty', 'worthless', 'suicide',
                'death', 'die', 'hate', 'lonely', 'alone', 'cry', 'crying',
                'tired', 'exhausted', 'numb', 'dark', 'pain', 'hurt'
            }
            
            positive_keywords = {
                'happy', 'joy', 'love', 'great', 'amazing', 'wonderful',
                'excited', 'blessed', 'thankful', 'grateful', 'good', 'best'
            }
            
            feature_scores = []
            
            # Score words based on presence in keyword sets
            for word, count in word_counts.items():
                if word in depression_keywords:
                    score = count * 2.0  # Higher weight for depression keywords
                    feature_scores.append((word, score))
                elif word in positive_keywords:
                    score = count * -1.0  # Negative weight for positive keywords
                    feature_scores.append((word, score))
                else:
                    score = count * 0.1  # Small weight for other words
                    feature_scores.append((word, score))
            
            # Sort by absolute importance
            feature_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            feature_scores = feature_scores[:max_features]
            
            return {
                'method': 'feature_importance',
                'feature_importance': [
                    {'feature': feat, 'importance': float(imp)}
                    for feat, imp in feature_scores
                ],
                'explanation_text': self._generate_feature_importance_explanation(feature_scores),
                'note': 'Simplified heuristic-based feature importance'
            }
            
        except Exception as e:
            return {'error': f'Feature importance failed: {e}'}
    
    def _generate_shap_text_explanation(self, feature_importance: List[Tuple]) -> str:
        """Generate human-readable SHAP explanation."""
        
        if not feature_importance:
            return "No significant features identified."
        
        explanation = "SHAP Analysis:\n"
        
        # Separate positive and negative contributions
        positive_features = [(f, i) for f, i in feature_importance if i > 0]
        negative_features = [(f, i) for f, i in feature_importance if i < 0]
        
        if positive_features:
            explanation += f"Features supporting depression classification:\n"
            for feat, imp in positive_features[:5]:
                explanation += f"  • {feat}: +{imp:.3f}\n"
        
        if negative_features:
            explanation += f"Features supporting non-depression classification:\n"
            for feat, imp in negative_features[:5]:
                explanation += f"  • {feat}: {imp:.3f}\n"
        
        return explanation
    
    def _generate_lime_text_explanation(self, feature_importance: List[Dict]) -> str:
        """Generate human-readable LIME explanation."""
        
        if not feature_importance:
            return "No significant features identified."
        
        explanation = "LIME Analysis:\n"
        explanation += "Key words/phrases influencing the prediction:\n"
        
        for item in feature_importance[:10]:
            feat = item['feature']
            imp = item['importance']
            direction = "supporting depression" if imp > 0 else "supporting non-depression"
            explanation += f"  • '{feat}': {direction} (weight: {imp:.3f})\n"
        
        return explanation
    
    def _generate_feature_importance_explanation(self, feature_scores: List[Tuple]) -> str:
        """Generate human-readable feature importance explanation."""
        
        if not feature_scores:
            return "No significant features identified."
        
        explanation = "Feature Importance Analysis:\n"
        explanation += "Words with highest impact on classification:\n"
        
        for feat, imp in feature_scores[:10]:
            if imp > 0:
                direction = "increases depression likelihood"
            else:
                direction = "decreases depression likelihood"
            explanation += f"  • '{feat}': {direction} (score: {imp:.2f})\n"
        
        return explanation
    
    def create_explanation_report(self, explanations: Dict[str, Any]) -> str:
        """
        Create a comprehensive explanation report.
        
        Args:
            explanations: Explanation results
            
        Returns:
            Formatted explanation report
        """
        
        report_lines = [
            "=" * 80,
            "MENTAL HEALTH TWEET CLASSIFIER - EXPLANATION REPORT",
            "=" * 80,
            "",
            f"Analysis Time: {explanations.get('timestamp', 'Unknown')}",
            f"Method(s): {explanations.get('method', 'Unknown')}",
            f"Number of Texts: {len(explanations.get('explanations', []))}",
            ""
        ]
        
        for i, exp in enumerate(explanations.get('explanations', [])):
            report_lines.extend([
                f"TEXT {i+1}:",
                "-" * 40,
                f"Original: {exp.get('original_text', '')[:100]}{'...' if len(exp.get('original_text', '')) > 100 else ''}",
                ""
            ])
            
            # Prediction info
            pred = exp.get('prediction', {})
            if pred and 'predicted_class' in pred:
                report_lines.extend([
                    f"Prediction: {pred['predicted_class']}",
                    f"Confidence: {pred.get('confidence', 'N/A'):.3f}" if pred.get('confidence') else "Confidence: N/A"
                ])
                
                if pred.get('probabilities'):
                    report_lines.append("Probabilities:")
                    for class_name, prob in pred['probabilities'].items():
                        report_lines.append(f"  {class_name}: {prob:.3f}")
            
            report_lines.append("")
            
            # Method explanations
            methods = exp.get('methods', {})
            for method_name, method_results in methods.items():
                if 'error' in method_results:
                    report_lines.extend([
                        f"{method_name.upper()} EXPLANATION:",
                        f"  Error: {method_results['error']}",
                        ""
                    ])
                    continue
                
                report_lines.extend([
                    f"{method_name.upper()} EXPLANATION:",
                    "-" * 30
                ])
                
                # Feature importance
                if 'feature_importance' in method_results:
                    report_lines.append("Top Features:")
                    for feat_info in method_results['feature_importance'][:10]:
                        feat = feat_info['feature']
                        imp = feat_info['importance']
                        report_lines.append(f"  {feat}: {imp:.4f}")
                
                # Explanation text
                if 'explanation_text' in method_results:
                    report_lines.extend([
                        "",
                        "Explanation:",
                        method_results['explanation_text'],
                        ""
                    ])
        
        # Ethical considerations
        report_lines.extend([
            "",
            "INTERPRETABILITY CONSIDERATIONS:",
            "-" * 40,
            "• Explanations are approximations of model behavior",
            "• Local explanations may not reflect global model patterns", 
            "• Consider multiple explanation methods for robustness",
            "• Human expertise required to validate AI explanations",
            "• Explanations should not replace clinical judgment",
            "",
            "RESPONSIBLE USE GUIDELINES:",
            "-" * 40,
            "• Use explanations to improve model understanding, not medical decisions",
            "• Consider explanation uncertainty and limitations",
            "• Validate explanations with domain experts",
            "• Regular auditing of explanation quality recommended",
            "• Be aware of potential biases in explanation methods",
            "",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def visualize_explanations(self, explanations: Dict[str, Any], save_plots: bool = True) -> Dict[str, Any]:
        """
        Create visualizations for model explanations.
        
        Args:
            explanations: Explanation results
            save_plots: Whether to save plots
            
        Returns:
            Dictionary of plot objects/paths
        """
        
        if not MATPLOTLIB_AVAILABLE:
            warnings.warn("Matplotlib not available. Skipping visualizations.")
            return {}
        
        plots = {}
        
        for i, exp in enumerate(explanations.get('explanations', [])):
            methods = exp.get('methods', {})
            
            # Create feature importance plots for each method
            for method_name, method_results in methods.items():
                if 'feature_importance' in method_results and 'error' not in method_results:
                    
                    fig_path = self._plot_feature_importance(
                        method_results['feature_importance'],
                        f"{method_name.upper()} - Text {i+1}",
                        save_plots,
                        f"{method_name}_text_{i+1}"
                    )
                    
                    if fig_path:
                        plots[f"{method_name}_text_{i+1}"] = fig_path
        
        return plots
    
    def _plot_feature_importance(self, 
                                feature_importance: List[Dict],
                                title: str,
                                save_plot: bool = True,
                                filename: str = "feature_importance") -> Optional[str]:
        """Create feature importance visualization."""
        
        if not feature_importance:
            return None
        
        # Prepare data
        features = [item['feature'] for item in feature_importance[:15]]  # Top 15
        importances = [item['importance'] for item in feature_importance[:15]]
        
        # Create colors based on importance direction
        colors = ['red' if imp > 0 else 'blue' for imp in importances]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
        
        # Customize plot
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(title)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add legend
        red_patch = plt.Rectangle((0,0),1,1, fc="red", alpha=0.7)
        blue_patch = plt.Rectangle((0,0),1,1, fc="blue", alpha=0.7)
        plt.legend([red_patch, blue_patch], 
                  ['Supports Depression', 'Supports Non-Depression'],
                  loc='lower right')
        
        plt.tight_layout()
        
        # Save if requested
        if save_plot:
            plot_path = self.save_dir / f"{filename}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return None
    
    def _save_explanations(self, explanations: Dict[str, Any]):
        """Save explanations to file."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"explanations_{timestamp}.json"
        
        filepath = self.save_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(explanations, f, indent=2, default=str)
            
            print(f"Explanations saved to: {filepath}")
            
        except Exception as e:
            warnings.warn(f"Could not save explanations: {e}")

def explain_mental_health_prediction(model: Any,
                                   text_input: Union[str, List[str]],
                                   method: str = "all",
                                   class_names: Optional[List[str]] = None,
                                   save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for explaining mental health predictions.
    
    Args:
        model: Trained model
        text_input: Text(s) to explain
        method: Explanation method
        class_names: Class names
        save_dir: Save directory
        
    Returns:
        Explanation results
    """
    
    explainer = ModelExplainer(
        model=model,
        class_names=class_names,
        save_dir=save_dir
    )
    
    explanations = explainer.explain_prediction(text_input, method=method)
    
    # Create and display report
    report = explainer.create_explanation_report(explanations)
    print(report)
    
    # Create visualizations
    if MATPLOTLIB_AVAILABLE:
        plots = explainer.visualize_explanations(explanations, save_plots=True)
        if plots:
            print(f"\nVisualization plots created: {list(plots.keys())}")
    
    return explanations

if __name__ == "__main__":
    print("Mental Health Tweet Classifier - Explainability Module")
    print("This module provides model interpretability capabilities.")
    print("\nFeatures:")
    print("- SHAP explanations for global/local interpretability")
    print("- LIME explanations for local text interpretability") 
    print("- Feature importance analysis")
    print("- Visualization of explanations")
    print("- Comprehensive explanation reports")
    print("- Ethical guidelines for responsible interpretation")
    print("\nUsage:")
    print("  from src.eval.explain import explain_mental_health_prediction")
    print("  explanations = explain_mental_health_prediction(model, text)")