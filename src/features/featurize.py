"""
Feature engineering for mental health tweet classification.

This module provides comprehensive feature extraction capabilities including:
- TF-IDF vectorization with configurable parameters
- Sentence-BERT embeddings 
- Classical NLP features (sentiment, readability, etc.)
- Feature caching and serialization
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib

# Import libraries with error handling for optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. BERT embeddings will be skipped.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("vaderSentiment not available. Sentiment features will be limited.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFIDFFeaturizer:
    """TF-IDF feature extraction with configurable parameters."""
    
    def __init__(self, 
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: Union[int, float] = 2,
                 max_df: Union[int, float] = 0.95,
                 stop_words: str = 'english',
                 lowercase: bool = True,
                 sublinear_tf: bool = True):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams to consider
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms  
            stop_words: Stop words to remove
            lowercase: Convert to lowercase
            sublinear_tf: Apply sublinear tf scaling
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.sublinear_tf = sublinear_tf
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            lowercase=lowercase,
            sublinear_tf=sublinear_tf
        )
        self.fitted = False
        
    def fit(self, texts: List[str]) -> 'TFIDFFeaturizer':
        """Fit TF-IDF vectorizer on texts."""
        logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} texts...")
        self.vectorizer.fit(texts)
        self.fitted = True
        
        # Log vocabulary info
        vocab_size = len(self.vectorizer.vocabulary_)
        feature_names = self.vectorizer.get_feature_names_out()
        logger.info(f"TF-IDF vocabulary size: {vocab_size}")
        logger.info(f"Sample features: {feature_names[:10]}")
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF vectors."""
        if not self.fitted:
            raise ValueError("TF-IDF vectorizer must be fitted first")
        
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizer and transform texts."""
        return self.fit(texts).transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """Get TF-IDF feature names."""
        if not self.fitted:
            raise ValueError("TF-IDF vectorizer must be fitted first")
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_top_features_by_class(self, 
                                 texts: List[str], 
                                 labels: List[int], 
                                 top_n: int = 20) -> Dict[int, List[Tuple[str, float]]]:
        """Get top TF-IDF features for each class."""
        if not self.fitted:
            self.fit(texts)
        
        tfidf_matrix = self.transform(texts)
        feature_names = self.get_feature_names()
        
        # Calculate mean TF-IDF scores by class
        unique_labels = sorted(set(labels))
        top_features = {}
        
        for label in unique_labels:
            class_mask = np.array(labels) == label
            class_tfidf = tfidf_matrix[class_mask]
            
            # Mean TF-IDF score per feature for this class
            mean_scores = class_tfidf.mean(axis=0)
            
            # Get top features
            top_indices = np.argsort(mean_scores)[::-1][:top_n]
            top_features[label] = [(feature_names[i], mean_scores[i]) for i in top_indices]
        
        return top_features
    
    def save(self, path: str):
        """Save fitted vectorizer."""
        if not self.fitted:
            raise ValueError("Cannot save unfitted vectorizer")
        
        joblib.dump(self.vectorizer, path)
        logger.info(f"TF-IDF vectorizer saved to {path}")
    
    def load(self, path: str):
        """Load fitted vectorizer."""
        self.vectorizer = joblib.load(path)
        self.fitted = True
        logger.info(f"TF-IDF vectorizer loaded from {path}")


class BERTFeaturizer:
    """Sentence-BERT embedding feature extraction."""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 cache_dir: Optional[str] = None,
                 device: str = 'cpu'):
        """
        Initialize BERT featurizer.
        
        Args:
            model_name: Sentence-BERT model name
            cache_dir: Directory to cache embeddings
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required for BERT features. Install with: pip install sentence-transformers")
        
        # Initialize model
        logger.info(f"Loading Sentence-BERT model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Model loaded on device: {self.model.device}")
        
        # Create cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts to BERT embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        # Check cache first
        cache_path = None
        if self.cache_dir:
            text_hash = hash(str(texts))  # Simple hash for caching
            cache_path = self.cache_dir / f"bert_embeddings_{text_hash}.npy"
            
            if cache_path.exists():
                logger.info(f"Loading cached embeddings from {cache_path}")
                return np.load(cache_path)
        
        # Encode embeddings
        logger.info(f"Encoding {len(texts)} texts with {self.model_name}...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Cache embeddings
        if cache_path:
            np.save(cache_path, embeddings)
            logger.info(f"Embeddings cached to {cache_path}")
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def encode_and_reduce(self, 
                         texts: List[str], 
                         n_components: int = 50,
                         method: str = 'pca') -> np.ndarray:
        """
        Encode texts and apply dimensionality reduction.
        
        Args:
            texts: List of texts to encode
            n_components: Number of components for reduction
            method: Reduction method ('pca' or 'umap')
            
        Returns:
            Reduced embeddings
        """
        from sklearn.decomposition import PCA
        
        # Get full embeddings
        embeddings = self.encode(texts)
        
        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
            
            explained_variance = reducer.explained_variance_ratio_.sum()
            logger.info(f"PCA reduction: {embeddings.shape[1]} -> {n_components} dims")
            logger.info(f"Explained variance: {explained_variance:.3f}")
            
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=n_components, random_state=42)
                reduced_embeddings = reducer.fit_transform(embeddings)
                logger.info(f"UMAP reduction: {embeddings.shape[1]} -> {n_components} dims")
            except ImportError:
                logger.warning("UMAP not available, falling back to PCA")
                return self.encode_and_reduce(texts, n_components, 'pca')
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        return reduced_embeddings


class ClassicalFeaturizer:
    """Extract classical NLP and linguistic features."""
    
    def __init__(self):
        """Initialize classical feature extractor."""
        self.sentiment_analyzer = vader_analyzer if VADER_AVAILABLE else None
        
    def extract_basic_features(self, text: str) -> Dict[str, float]:
        """Extract basic statistical features."""
        if not text or not isinstance(text, str):
            return {
                'char_count': 0, 'word_count': 0, 'sentence_count': 0,
                'avg_word_length': 0, 'punct_count': 0, 'uppercase_ratio': 0
            }
        
        words = text.split()
        sentences = len([s for s in text.split('.') if s.strip()])
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': max(1, sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'punct_count': sum(1 for c in text if c in '.,!?;:"'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
    
    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment features using VADER."""
        if not self.sentiment_analyzer or not text:
            return {'sentiment_compound': 0, 'sentiment_pos': 0, 
                   'sentiment_neg': 0, 'sentiment_neu': 0}
        
        scores = self.sentiment_analyzer.polarity_scores(text)
        return {
            'sentiment_compound': scores['compound'],
            'sentiment_pos': scores['pos'],
            'sentiment_neg': scores['neg'],
            'sentiment_neu': scores['neu']
        }
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic and stylistic features."""
        import re
        
        if not text:
            return {
                'exclamation_count': 0, 'question_count': 0, 'caps_words': 0,
                'hashtag_count': 0, 'mention_count': 0, 'url_count': 0
            }
        
        return {
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_words': len([w for w in text.split() if w.isupper() and len(w) > 1]),
            'hashtag_count': len(re.findall(r'#\\w+', text)),
            'mention_count': len(re.findall(r'@\\w+', text)),
            'url_count': len(re.findall(r'http\\S+|www\\S+', text))
        }
    
    def extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability and complexity features.""" 
        if not text or not isinstance(text, str):
            return {'flesch_reading_ease': 0, 'avg_syllables': 0, 'complex_words': 0}
        
        words = text.split()
        sentences = max(1, len([s for s in text.split('.') if s.strip()]))
        
        # Simple syllable counting (approximation)
        def count_syllables(word):
            word = word.lower().strip(".,!?;:")
            vowels = 'aeiouy'
            syllable_count = 0
            previous_was_vowel = False
            
            for char in word:
                if char in vowels:
                    if not previous_was_vowel:
                        syllable_count += 1
                    previous_was_vowel = True
                else:
                    previous_was_vowel = False
            
            if word.endswith('e'):
                syllable_count -= 1
            
            return max(1, syllable_count)
        
        if not words:
            return {'flesch_reading_ease': 0, 'avg_syllables': 0, 'complex_words': 0}
        
        total_syllables = sum(count_syllables(word) for word in words)
        avg_syllables = total_syllables / len(words)
        complex_words = sum(1 for word in words if count_syllables(word) >= 3)
        
        # Flesch Reading Ease (simplified)
        asl = len(words) / sentences  # Average sentence length
        asw = avg_syllables  # Average syllables per word
        flesch = 206.835 - (1.015 * asl) - (84.6 * asw)
        flesch = max(0, min(100, flesch))
        
        return {
            'flesch_reading_ease': flesch,
            'avg_syllables': avg_syllables,
            'complex_words': complex_words / len(words) if words else 0
        }
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """Extract all classical features for a single text."""
        features = {}
        features.update(self.extract_basic_features(text))
        features.update(self.extract_sentiment_features(text))
        features.update(self.extract_linguistic_features(text))
        features.update(self.extract_readability_features(text))
        
        return features
    
    def extract_batch_features(self, texts: List[str]) -> pd.DataFrame:
        """Extract classical features for a batch of texts."""
        logger.info(f"Extracting classical features for {len(texts)} texts...")
        
        features_list = []
        for text in texts:
            features = self.extract_all_features(text)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(features_df.columns)} classical features")
        logger.info(f"Feature names: {features_df.columns.tolist()}")
        
        return features_df


class FeaturePipeline:
    """Complete feature engineering pipeline combining multiple feature types."""
    
    def __init__(self, 
                 use_tfidf: bool = True,
                 use_bert: bool = True,
                 use_classical: bool = True,
                 tfidf_params: Dict = None,
                 bert_params: Dict = None,
                 scale_features: bool = True):
        """
        Initialize feature pipeline.
        
        Args:
            use_tfidf: Whether to include TF-IDF features
            use_bert: Whether to include BERT embeddings
            use_classical: Whether to include classical features
            tfidf_params: Parameters for TF-IDF vectorizer
            bert_params: Parameters for BERT encoder
            scale_features: Whether to scale numerical features
        """
        self.use_tfidf = use_tfidf
        self.use_bert = use_bert and SENTENCE_TRANSFORMERS_AVAILABLE
        self.use_classical = use_classical
        self.scale_features = scale_features
        
        # Initialize feature extractors
        if self.use_tfidf:
            tfidf_params = tfidf_params or {}
            self.tfidf_featurizer = TFIDFFeaturizer(**tfidf_params)
        
        if self.use_bert:
            bert_params = bert_params or {}
            self.bert_featurizer = BERTFeaturizer(**bert_params)
        
        if self.use_classical:
            self.classical_featurizer = ClassicalFeaturizer()
        
        # Scaler for numerical features
        if self.scale_features:
            self.scaler = StandardScaler()
            self.scaler_fitted = False
        
        self.fitted = False
        self.feature_names = []
        
    def fit(self, texts: List[str], labels: List[int] = None) -> 'FeaturePipeline':
        """Fit feature extractors on training data."""
        logger.info(f"Fitting feature pipeline on {len(texts)} texts...")
        
        # Fit TF-IDF
        if self.use_tfidf:
            self.tfidf_featurizer.fit(texts)
        
        # BERT doesn't need fitting, but we can pre-compute embeddings
        if self.use_bert:
            logger.info("Pre-computing BERT embeddings for training data...")
            # This will cache the embeddings
            _ = self.bert_featurizer.encode(texts)
        
        # Classical features don't need fitting
        
        self.fitted = True
        logger.info("Feature pipeline fitted successfully!")
        return self
    
    def transform(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Transform texts to feature vectors."""
        if not self.fitted and self.use_tfidf:
            raise ValueError("Pipeline must be fitted before transform")
        
        logger.info(f"Transforming {len(texts)} texts to features...")
        
        all_features = []
        feature_names = []
        
        # TF-IDF features
        if self.use_tfidf:
            logger.info("Extracting TF-IDF features...")
            tfidf_features = self.tfidf_featurizer.transform(texts)
            all_features.append(tfidf_features)
            
            tfidf_names = [f"tfidf_{name}" for name in self.tfidf_featurizer.get_feature_names()]
            feature_names.extend(tfidf_names)
            logger.info(f"TF-IDF features shape: {tfidf_features.shape}")
        
        # BERT features
        if self.use_bert:
            logger.info("Extracting BERT embeddings...")
            bert_features = self.bert_featurizer.encode(texts)
            all_features.append(bert_features)
            
            bert_names = [f"bert_{i}" for i in range(bert_features.shape[1])]
            feature_names.extend(bert_names)
            logger.info(f"BERT features shape: {bert_features.shape}")
        
        # Classical features
        if self.use_classical:
            logger.info("Extracting classical features...")
            classical_df = self.classical_featurizer.extract_batch_features(texts)
            classical_features = classical_df.values
            all_features.append(classical_features)
            
            classical_names = [f"classical_{name}" for name in classical_df.columns]
            feature_names.extend(classical_names)
            logger.info(f"Classical features shape: {classical_features.shape}")
        
        # Combine all features
        if len(all_features) > 1:
            combined_features = np.hstack(all_features)
        else:
            combined_features = all_features[0]
        
        # Scale features if requested
        if self.scale_features and self.use_classical:
            # Only scale classical features (not TF-IDF or BERT which are already normalized)
            if not hasattr(self, 'scaler_fitted') or not self.scaler_fitted:
                # Fit scaler on classical features only
                if self.use_classical:
                    classical_start = (tfidf_features.shape[1] if self.use_tfidf else 0) + (bert_features.shape[1] if self.use_bert else 0)
                    classical_end = combined_features.shape[1]
                    self.scaler.fit(combined_features[:, classical_start:classical_end])
                    self.scaler_fitted = True
            
            # Apply scaling
            if self.use_classical:
                classical_start = (tfidf_features.shape[1] if self.use_tfidf else 0) + (bert_features.shape[1] if self.use_bert else 0)
                classical_end = combined_features.shape[1]
                combined_features[:, classical_start:classical_end] = self.scaler.transform(combined_features[:, classical_start:classical_end])
        
        self.feature_names = feature_names
        
        logger.info(f"Final feature matrix shape: {combined_features.shape}")
        return combined_features, feature_names
    
    def fit_transform(self, texts: List[str], labels: List[int] = None) -> Tuple[np.ndarray, List[str]]:
        """Fit pipeline and transform texts."""
        return self.fit(texts, labels).transform(texts)
    
    def save(self, directory: str):
        """Save fitted pipeline."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save TF-IDF vectorizer
        if self.use_tfidf and self.tfidf_featurizer.fitted:
            self.tfidf_featurizer.save(directory / "tfidf_vectorizer.pkl")
        
        # Save scaler
        if self.scale_features and hasattr(self, 'scaler_fitted') and self.scaler_fitted:
            joblib.dump(self.scaler, directory / "feature_scaler.pkl")
        
        # Save pipeline metadata
        metadata = {
            'use_tfidf': self.use_tfidf,
            'use_bert': self.use_bert,
            'use_classical': self.use_classical,
            'scale_features': self.scale_features,
            'fitted': self.fitted,
            'feature_names': self.feature_names
        }
        
        with open(directory / "pipeline_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Feature pipeline saved to {directory}")
    
    def load(self, directory: str):
        """Load fitted pipeline."""
        directory = Path(directory)
        
        # Load metadata
        with open(directory / "pipeline_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.use_tfidf = metadata['use_tfidf']
        self.use_bert = metadata['use_bert']
        self.use_classical = metadata['use_classical']
        self.scale_features = metadata['scale_features']
        self.fitted = metadata['fitted']
        self.feature_names = metadata['feature_names']
        
        # Load TF-IDF vectorizer
        if self.use_tfidf:
            self.tfidf_featurizer = TFIDFFeaturizer()
            self.tfidf_featurizer.load(directory / "tfidf_vectorizer.pkl")
        
        # Load scaler
        if self.scale_features:
            scaler_path = directory / "feature_scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                self.scaler_fitted = True
        
        # Reinitialize other featurizers
        if self.use_bert and SENTENCE_TRANSFORMERS_AVAILABLE:
            self.bert_featurizer = BERTFeaturizer()
        
        if self.use_classical:
            self.classical_featurizer = ClassicalFeaturizer()
        
        logger.info(f"Feature pipeline loaded from {directory}")


def create_features(texts: List[str], 
                   labels: List[int] = None,
                   output_format: str = 'numpy',
                   save_path: Optional[str] = None) -> Union[np.ndarray, pd.DataFrame]:
    """
    Convenience function to create features with default settings.
    
    Args:
        texts: List of texts to featurize
        labels: Optional labels for supervised feature selection
        output_format: 'numpy' or 'dataframe'
        save_path: Path to save features (parquet format)
        
    Returns:
        Feature matrix as numpy array or DataFrame
    """
    # Create default pipeline
    pipeline = FeaturePipeline(
        use_tfidf=True,
        use_bert=SENTENCE_TRANSFORMERS_AVAILABLE,
        use_classical=True,
        tfidf_params={'max_features': 5000, 'ngram_range': (1, 2)},
        bert_params={'model_name': 'all-MiniLM-L6-v2'} if SENTENCE_TRANSFORMERS_AVAILABLE else None
    )
    
    # Extract features
    features, feature_names = pipeline.fit_transform(texts, labels)
    
    if output_format == 'dataframe':
        features_df = pd.DataFrame(features, columns=feature_names)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_parquet(save_path, index=False)
            logger.info(f"Features saved to {save_path}")
        
        return features_df
    
    else:  # numpy
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as npz with feature names
            np.savez_compressed(save_path, 
                              features=features, 
                              feature_names=feature_names)
            logger.info(f"Features saved to {save_path}")
        
        return features


if __name__ == "__main__":
    # Example usage and testing
    print("Feature Engineering Module Test")
    print("=" * 40)
    
    # Sample texts
    sample_texts = [
        "I'm feeling really depressed today... everything seems hopeless 😢",
        "Great workout at the gym! Feeling energized and positive! 💪 #fitness",
        "Why is everything so difficult? I can't handle this anymore.",
        "Love this beautiful sunny day! Perfect for a walk in the park ☀️",
        "Can't sleep again... my mind won't stop racing with negative thoughts"
    ]
    
    sample_labels = [1, 0, 1, 0, 1]  # depression, non-depression, etc.
    
    print(f"\\nTesting with {len(sample_texts)} sample texts...")
    
    # Test individual feature extractors
    print("\\n" + "="*50)
    print("TESTING INDIVIDUAL FEATURE EXTRACTORS")
    print("="*50)
    
    # TF-IDF features
    print("\\n1. TF-IDF Features:")
    tfidf_featurizer = TFIDFFeaturizer(max_features=100)
    tfidf_features = tfidf_featurizer.fit_transform(sample_texts)
    print(f"   Shape: {tfidf_features.shape}")
    print(f"   Sample feature names: {tfidf_featurizer.get_feature_names()[:5]}")
    
    # Classical features
    print("\\n2. Classical Features:")
    classical_featurizer = ClassicalFeaturizer()
    classical_df = classical_featurizer.extract_batch_features(sample_texts)
    print(f"   Shape: {classical_df.shape}")
    print(f"   Feature names: {classical_df.columns.tolist()}")
    print("   Sample values:")
    print(classical_df.head())
    
    # BERT features (if available)
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\\n3. BERT Features:")
        try:
            bert_featurizer = BERTFeaturizer(model_name='all-MiniLM-L6-v2')
            bert_features = bert_featurizer.encode(sample_texts[:3])  # Just first 3 for speed
            print(f"   Shape: {bert_features.shape}")
            print(f"   Sample embedding: {bert_features[0][:5]}")
        except Exception as e:
            print(f"   BERT feature extraction failed: {e}")
    else:
        print("\\n3. BERT Features: Not available (install sentence-transformers)")
    
    # Test complete pipeline
    print("\\n" + "="*50)
    print("TESTING COMPLETE FEATURE PIPELINE")
    print("="*50)
    
    # Create pipeline
    pipeline = FeaturePipeline(
        use_tfidf=True,
        use_bert=SENTENCE_TRANSFORMERS_AVAILABLE,
        use_classical=True,
        tfidf_params={'max_features': 50, 'ngram_range': (1, 1)}  # Small for demo
    )
    
    # Extract features
    all_features, feature_names = pipeline.fit_transform(sample_texts, sample_labels)
    
    print(f"Combined features shape: {all_features.shape}")
    print(f"Number of feature types: {len([x for x in [pipeline.use_tfidf, pipeline.use_bert, pipeline.use_classical] if x])}")
    print(f"Feature name examples: {feature_names[:5]}")
    
    # Test convenience function
    print("\\n" + "="*50)
    print("TESTING CONVENIENCE FUNCTION")
    print("="*50)
    
    features_df = create_features(sample_texts, sample_labels, output_format='dataframe')
    print(f"Features DataFrame shape: {features_df.shape}")
    print(f"Feature columns: {features_df.columns[:5].tolist()}")
    
    print("\\n✅ Feature engineering module test complete!")
    print("\\nNext steps:")
    print("- Use FeaturePipeline for training models")
    print("- Save/load fitted pipelines for reproducibility")
    print("- Experiment with different feature combinations")