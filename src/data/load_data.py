"""
Data loading utilities for the mental health tweet classifier.

This module provides functions to load and combine the tweet datasets,
automatically inferring schema and handling both raw and preprocessed data.
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Load and combine mental health tweet datasets."""
    
    def __init__(self, dataset_dir: str = "dataset"):
        """Initialize data loader with dataset directory."""
        self.dataset_dir = Path(dataset_dir)
        self.schema_path = Path("reports/dataset_schema.json")
        
    def load_schema(self) -> Dict:
        """Load dataset schema information."""
        if self.schema_path.exists():
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Schema file not found at {self.schema_path}")
            return {}
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load raw (unprocessed) tweet data.
        
        Returns:
            Tuple of (features_df, labels_series)
        """
        logger.info("Loading raw tweet data...")
        
        # Load depression tweets
        d_tweets_path = self.dataset_dir / "d_tweets.csv"
        non_d_tweets_path = self.dataset_dir / "non_d_tweets.csv"
        
        if not d_tweets_path.exists() or not non_d_tweets_path.exists():
            raise FileNotFoundError("Required dataset files not found")
        
        # Load datasets
        d_tweets = pd.read_csv(d_tweets_path)
        non_d_tweets = pd.read_csv(non_d_tweets_path)
        
        # Add labels
        d_tweets['label'] = 1  # depression
        non_d_tweets['label'] = 0  # non-depression
        
        # Combine datasets
        combined_df = pd.concat([d_tweets, non_d_tweets], ignore_index=True)
        
        logger.info(f"Loaded {len(d_tweets)} depression tweets and {len(non_d_tweets)} non-depression tweets")
        logger.info(f"Total samples: {len(combined_df)}")
        
        # Extract features and labels
        features = combined_df.drop('label', axis=1)
        labels = combined_df['label']
        
        return features, labels
    
    def load_clean_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load preprocessed (clean) tweet data.
        
        Returns:
            Tuple of (features_df, labels_series)
        """
        logger.info("Loading preprocessed tweet data...")
        
        # Load clean datasets
        clean_d_path = self.dataset_dir / "clean_d_tweets.csv" 
        clean_non_d_path = self.dataset_dir / "clean_non_d_tweets.csv"
        
        if not clean_d_path.exists() or not clean_non_d_path.exists():
            raise FileNotFoundError("Clean dataset files not found")
        
        # Load datasets
        clean_d_tweets = pd.read_csv(clean_d_path)
        clean_non_d_tweets = pd.read_csv(clean_non_d_path)
        
        # Add labels
        clean_d_tweets['label'] = 1  # depression
        clean_non_d_tweets['label'] = 0  # non-depression
        
        # Combine datasets
        combined_df = pd.concat([clean_d_tweets, clean_non_d_tweets], ignore_index=True)
        
        logger.info(f"Loaded {len(clean_d_tweets)} clean depression tweets and {len(clean_non_d_tweets)} clean non-depression tweets")
        logger.info(f"Total clean samples: {len(combined_df)}")
        
        # Extract features and labels
        features = combined_df.drop('label', axis=1)
        labels = combined_df['label']
        
        return features, labels
    
    def get_text_and_labels(self, use_clean: bool = False) -> Tuple[List[str], List[int]]:
        """
        Get just the text and labels for modeling.
        
        Args:
            use_clean: Whether to use preprocessed text
            
        Returns:
            Tuple of (texts, labels)
        """
        if use_clean:
            features, labels = self.load_clean_data()
        else:
            features, labels = self.load_raw_data()
        
        texts = features['tweet'].tolist()
        labels_list = labels.tolist()
        
        return texts, labels_list
    
    def get_user_level_data(self, use_clean: bool = False, max_tweets_per_user: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aggregate data at user level for user-level predictions.
        
        Args:
            use_clean: Whether to use preprocessed text
            max_tweets_per_user: Maximum tweets to consider per user
            
        Returns:
            Tuple of (user_features_df, user_labels_series)
        """
        if use_clean:
            features, labels = self.load_clean_data()
        else:
            features, labels = self.load_raw_data()
        
        # Combine for easier processing
        df = features.copy()
        df['label'] = labels
        
        # Group by user and aggregate
        user_data = []
        
        for user_id, user_tweets in df.groupby('user_id'):
            # Take top N tweets by engagement (nlikes + nretweets)
            user_tweets['engagement'] = user_tweets['nlikes'].fillna(0) + user_tweets['nretweets'].fillna(0)
            top_tweets = user_tweets.nlargest(max_tweets_per_user, 'engagement')
            
            # Aggregate text
            combined_text = ' [SEP] '.join(top_tweets['tweet'].tolist())
            
            # User-level features
            user_features = {
                'user_id': user_id,
                'username': top_tweets['username'].iloc[0],
                'name': top_tweets['name'].iloc[0],
                'tweet_count': len(user_tweets),
                'combined_text': combined_text,
                'avg_likes': user_tweets['nlikes'].mean(),
                'avg_retweets': user_tweets['nretweets'].mean(),
                'avg_replies': user_tweets['nreplies'].mean(),
                'total_engagement': user_tweets['engagement'].sum(),
            }
            
            # User label (majority vote, but in this dataset users are consistent)
            user_label = user_tweets['label'].mode().iloc[0]
            
            user_data.append((user_features, user_label))
        
        # Convert to DataFrame
        user_features_list = [item[0] for item in user_data]
        user_labels_list = [item[1] for item in user_data]
        
        user_df = pd.DataFrame(user_features_list)
        user_labels = pd.Series(user_labels_list)
        
        logger.info(f"Created user-level dataset with {len(user_df)} users")
        logger.info(f"User-level class distribution: {user_labels.value_counts().to_dict()}")
        
        return user_df, user_labels
    
    def save_processed_data(self, texts: List[str], labels: List[int], output_path: str = "data/processed/processed.csv"):
        """Save processed text and labels to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")


def load_data(use_clean: bool = False, user_level: bool = False) -> Tuple[List[str], List[int]]:
    """
    Convenience function to load data.
    
    Args:
        use_clean: Whether to use preprocessed text
        user_level: Whether to return user-level aggregated data
        
    Returns:
        Tuple of (texts, labels)
    """
    loader = DataLoader()
    
    if user_level:
        user_df, user_labels = loader.get_user_level_data(use_clean=use_clean)
        texts = user_df['combined_text'].tolist()
        labels = user_labels.tolist()
    else:
        texts, labels = loader.get_text_and_labels(use_clean=use_clean)
    
    return texts, labels


if __name__ == "__main__":
    # Example usage and testing
    loader = DataLoader()
    
    # Load schema
    schema = loader.load_schema()
    print("Dataset Schema:")
    print(json.dumps(schema, indent=2))
    
    # Load raw data
    print("\n" + "="*50)
    print("RAW DATA")
    print("="*50)
    texts_raw, labels_raw = loader.get_text_and_labels(use_clean=False)
    print(f"Raw data: {len(texts_raw)} samples")
    print(f"Label distribution: {pd.Series(labels_raw).value_counts().to_dict()}")
    print(f"Sample text: {texts_raw[0][:100]}...")
    
    # Load clean data  
    print("\n" + "="*50)
    print("CLEAN DATA")
    print("="*50)
    texts_clean, labels_clean = loader.get_text_and_labels(use_clean=True)
    print(f"Clean data: {len(texts_clean)} samples")
    print(f"Label distribution: {pd.Series(labels_clean).value_counts().to_dict()}")
    print(f"Sample text: {texts_clean[0][:100]}...")
    
    # User-level data
    print("\n" + "="*50)
    print("USER-LEVEL DATA")
    print("="*50)
    user_df, user_labels = loader.get_user_level_data(use_clean=False)
    print(f"User-level data: {len(user_df)} users")
    print(f"User label distribution: {user_labels.value_counts().to_dict()}")
    print(f"Sample user text: {user_df['combined_text'].iloc[0][:100]}...")