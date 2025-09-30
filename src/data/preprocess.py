"""
Tweet preprocessing utilities for mental health classification.

This module provides comprehensive preprocessing functions for tweet text,
including cleaning, normalization, language detection, and tokenization.
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

# Import libraries with error handling for optional dependencies
try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    logging.warning("emoji library not available. Emoji processing will be limited.")

try:
    from langdetect import detect, LangDetectError
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available. Language filtering will be skipped.")

try:
    import spacy
    # Try to load English model
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        SPACY_AVAILABLE = False
        logging.warning("spaCy English model not available. Using basic tokenization.")
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Using basic tokenization.")

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. HF tokenization will be skipped.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("vaderSentiment not available. Sentiment analysis will be skipped.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TweetPreprocessor:
    """Comprehensive tweet preprocessing for mental health classification."""
    
    def __init__(self, 
                 remove_urls: bool = True,
                 remove_mentions: bool = False,
                 remove_hashtags: bool = False,
                 expand_contractions: bool = True,
                 normalize_emojis: bool = True,
                 lowercase: bool = True,
                 remove_extra_whitespace: bool = True,
                 min_length: int = 3,
                 language_filter: str = "en"):
        """
        Initialize tweet preprocessor.
        
        Args:
            remove_urls: Remove URLs from tweets
            remove_mentions: Remove @mentions from tweets
            remove_hashtags: Remove #hashtags from tweets  
            expand_contractions: Expand contractions (don't -> do not)
            normalize_emojis: Convert emojis to text descriptions
            lowercase: Convert to lowercase
            remove_extra_whitespace: Remove extra whitespace
            min_length: Minimum text length after preprocessing
            language_filter: Language to filter for ("en" or None to skip)
        """
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.expand_contractions = expand_contractions
        self.normalize_emojis = normalize_emojis
        self.lowercase = lowercase
        self.remove_extra_whitespace = remove_extra_whitespace
        self.min_length = min_length
        self.language_filter = language_filter
        
        # Contractions dictionary
        self.contractions = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot", "couldn't": "could not",
            "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
            "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
            "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
            "let's": "let us", "mustn't": "must not", "shan't": "shall not", "she'd": "she would",
            "she'll": "she will", "she's": "she is", "shouldn't": "should not", "that's": "that is",
            "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are",
            "they've": "they have", "we'd": "we would", "we're": "we are", "we've": "we have",
            "weren't": "were not", "what's": "what is", "where's": "where is", "who's": "who is",
            "won't": "will not", "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
            "you're": "you are", "you've": "you have"
        }
        
        logger.info("TweetPreprocessor initialized")
        
    def detect_language(self, text: str) -> str:
        """Detect language of text."""
        if not LANGDETECT_AVAILABLE:
            return "unknown"
        
        try:
            return detect(text)
        except LangDetectError:
            return "unknown"
    
    def remove_urls_from_text(self, text: str) -> str:
        """Remove URLs from text."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        return text
    
    def remove_mentions_from_text(self, text: str) -> str:
        """Remove @mentions from text."""
        text = re.sub(r'@\w+', '', text)
        return text
    
    def remove_hashtags_from_text(self, text: str) -> str:
        """Remove #hashtags from text.""" 
        text = re.sub(r'#\w+', '', text)
        return text
    
    def expand_contractions_in_text(self, text: str) -> str:
        """Expand contractions in text."""
        words = text.split()
        expanded_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower in self.contractions:
                expanded_words.append(self.contractions[word_lower])
            else:
                expanded_words.append(word)
        return ' '.join(expanded_words)
    
    def normalize_emojis_in_text(self, text: str) -> str:
        """Convert emojis to text descriptions."""
        if not EMOJI_AVAILABLE:
            # Basic emoji removal if emoji package not available
            text = re.sub(r'[^\w\s]', '', text)
            return text
        
        # Convert emojis to text
        text = emoji.demojize(text, delimiters=(" ", " "))
        return text
    
    def clean_text(self, text: str) -> str:
        """Apply all text cleaning steps."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Apply cleaning steps in order
        if self.remove_urls:
            text = self.remove_urls_from_text(text)
            
        if self.remove_mentions:
            text = self.remove_mentions_from_text(text)
            
        if self.remove_hashtags:
            text = self.remove_hashtags_from_text(text)
            
        if self.expand_contractions:
            text = self.expand_contractions_in_text(text)
            
        if self.normalize_emojis:
            text = self.normalize_emojis_in_text(text)
            
        if self.lowercase:
            text = text.lower()
            
        # Remove extra punctuation and whitespace
        text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
        
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def filter_by_language(self, texts: List[str]) -> List[bool]:
        """Filter texts by language. Returns boolean mask."""
        if not self.language_filter or not LANGDETECT_AVAILABLE:
            return [True] * len(texts)
        
        mask = []
        for text in texts:
            lang = self.detect_language(text)
            mask.append(lang == self.language_filter)
        
        return mask
    
    def filter_by_length(self, texts: List[str]) -> List[bool]:
        """Filter texts by minimum length. Returns boolean mask."""
        return [len(text.split()) >= self.min_length for text in texts]
    
    def preprocess_batch(self, texts: List[str]) -> Tuple[List[str], List[bool]]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of raw texts
            
        Returns:
            Tuple of (cleaned_texts, valid_mask)
        """
        logger.info(f"Preprocessing {len(texts)} texts...")
        
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Apply filters
        length_mask = self.filter_by_length(cleaned_texts)
        language_mask = self.filter_by_language(cleaned_texts)
        
        # Combine masks
        valid_mask = [l and lang for l, lang in zip(length_mask, language_mask)]
        
        logger.info(f"After preprocessing: {sum(valid_mask)}/{len(texts)} texts remain")
        logger.info(f"Filtered out: {len(texts) - sum(valid_mask)} texts")
        
        return cleaned_texts, valid_mask


class TweetTokenizer:
    """Tweet tokenization utilities."""
    
    def __init__(self, method: str = "basic"):
        """
        Initialize tokenizer.
        
        Args:
            method: Tokenization method ("basic", "spacy", "transformers")
        """
        self.method = method
        
        if method == "transformers" and TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif method == "spacy" and SPACY_AVAILABLE:
            self.tokenizer = nlp
        else:
            self.method = "basic"
            
        logger.info(f"TweetTokenizer initialized with method: {self.method}")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        if self.method == "transformers" and TRANSFORMERS_AVAILABLE:
            tokens = self.tokenizer.tokenize(text)
            return tokens
        elif self.method == "spacy" and SPACY_AVAILABLE:
            doc = self.tokenizer(text)
            tokens = [token.text for token in doc if not token.is_space]
            return tokens
        else:
            # Basic tokenization
            tokens = text.split()
            return tokens
    
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """Tokenize batch of texts."""
        return [self.tokenize(text) for text in texts]


class FeatureExtractor:
    """Extract additional features from tweets."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.sentiment_analyzer = vader_analyzer if VADER_AVAILABLE else None
    
    def extract_basic_features(self, text: str) -> Dict:
        """Extract basic statistical features."""
        features = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        }
        return features
    
    def extract_emoji_features(self, text: str) -> Dict:
        """Extract emoji-related features."""
        if not EMOJI_AVAILABLE:
            return {'emoji_count': 0}
        
        emoji_count = len([c for c in text if c in emoji.UNICODE_EMOJI['en']])
        
        return {
            'emoji_count': emoji_count,
            'has_emojis': emoji_count > 0
        }
    
    def extract_sentiment_features(self, text: str) -> Dict:
        """Extract sentiment features using VADER."""
        if not self.sentiment_analyzer:
            return {'sentiment_compound': 0, 'sentiment_positive': 0, 
                   'sentiment_negative': 0, 'sentiment_neutral': 0}
        
        scores = self.sentiment_analyzer.polarity_scores(text)
        return {
            'sentiment_compound': scores['compound'],
            'sentiment_positive': scores['pos'],
            'sentiment_negative': scores['neg'], 
            'sentiment_neutral': scores['neu']
        }
    
    def extract_all_features(self, text: str) -> Dict:
        """Extract all available features."""
        features = {}
        features.update(self.extract_basic_features(text))
        features.update(self.extract_emoji_features(text))
        features.update(self.extract_sentiment_features(text))
        
        return features


def preprocess_tweets(texts: List[str], 
                     labels: List[int] = None,
                     config: Dict = None) -> Tuple[List[str], List[int], pd.DataFrame]:
    """
    Main preprocessing function.
    
    Args:
        texts: List of raw tweet texts
        labels: List of corresponding labels (optional)
        config: Configuration dictionary for preprocessing
        
    Returns:
        Tuple of (cleaned_texts, filtered_labels, features_df)
    """
    if config is None:
        config = {}
    
    # Initialize preprocessor
    preprocessor = TweetPreprocessor(**config)
    
    # Preprocess texts
    cleaned_texts, valid_mask = preprocessor.preprocess_batch(texts)
    
    # Filter based on valid mask
    filtered_texts = [text for text, valid in zip(cleaned_texts, valid_mask) if valid]
    filtered_labels = [label for label, valid in zip(labels or [], valid_mask) if valid] if labels else None
    
    # Extract additional features
    feature_extractor = FeatureExtractor()
    features_list = [feature_extractor.extract_all_features(text) for text in filtered_texts]
    features_df = pd.DataFrame(features_list)
    
    logger.info(f"Preprocessing complete. Final dataset: {len(filtered_texts)} samples")
    
    return filtered_texts, filtered_labels, features_df


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "I'm feeling really depressed today... can't seem to do anything right 😢",
        "Just finished an amazing workout! Feeling great! 💪 #fitness #motivation",
        "Why is everything so hard? I don't understand what's happening to me.",
        "Love this new coffee shop! Great vibes ☕",
    ]
    
    sample_labels = [1, 0, 1, 0]  # depression, non-depression, depression, non-depression
    
    # Test preprocessing
    print("Original texts:")
    for i, text in enumerate(sample_texts):
        print(f"{i}: {text}")
    
    print("\n" + "="*50)
    print("PREPROCESSING")
    print("="*50)
    
    cleaned_texts, cleaned_labels, features_df = preprocess_tweets(sample_texts, sample_labels)
    
    print("\nCleaned texts:")
    for i, text in enumerate(cleaned_texts):
        print(f"{i}: {text}")
    
    print(f"\nLabels: {cleaned_labels}")
    
    print("\nExtracted features:")
    print(features_df.head())
    
    # Test tokenization
    print("\n" + "="*50)
    print("TOKENIZATION")
    print("="*50)
    
    tokenizer = TweetTokenizer("basic")
    for text in cleaned_texts[:2]:
        tokens = tokenizer.tokenize(text)
        print(f"Text: {text}")
        print(f"Tokens: {tokens}")
        print()