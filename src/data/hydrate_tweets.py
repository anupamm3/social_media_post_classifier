"""
Twitter/X tweet hydration utility for mental health tweet classifier.

This module provides functionality to hydrate tweet IDs into full tweet objects
using the Twitter/X API. Since the current dataset already contains full text,
this is provided for completeness and future use cases.

IMPORTANT: Twitter/X API access requires authentication and may be paid.
This script will handle missing/deleted tweets gracefully.
"""

import pandas as pd
import json
import time
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

# Import tweepy with error handling
try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    logging.warning("tweepy not available. Tweet hydration will not work.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TweetHydrator:
    """Hydrate tweet IDs using Twitter/X API."""
    
    def __init__(self):
        """Initialize Twitter API client from environment variables."""
        
        if not TWEEPY_AVAILABLE:
            raise ImportError("tweepy is required for tweet hydration. Install with: pip install tweepy")
        
        # Get API credentials from environment variables
        api_key = os.getenv('TWITTER_API_KEY')
        api_secret = os.getenv('TWITTER_API_SECRET') 
        bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        # Check for required credentials
        if not bearer_token:
            raise ValueError(
                "Missing Twitter API credentials. Please set environment variables:\\n"
                "- TWITTER_API_KEY\\n"
                "- TWITTER_API_SECRET\\n" 
                "- TWITTER_BEARER_TOKEN\\n"
                "- TWITTER_ACCESS_TOKEN (optional)\\n"
                "- TWITTER_ACCESS_TOKEN_SECRET (optional)\\n\\n"
                "Note: Twitter/X API access may require paid subscription."
            )
        
        try:
            # Initialize Twitter API v2 client
            self.client = tweepy.Client(
                bearer_token=bearer_token,
                consumer_key=api_key,
                consumer_secret=api_secret,
                access_token=access_token,
                access_token_secret=access_token_secret,
                wait_on_rate_limit=True
            )
            
            logger.info("Twitter API client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Twitter API client: {e}")
            raise
    
    def test_api_connection(self) -> bool:
        """Test API connection and authentication."""
        try:
            # Try to get user info (a simple API call)
            user = self.client.get_me()
            if user:
                logger.info("API connection successful")
                return True
            else:
                logger.error("API connection failed - no user returned")
                return False
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False
    
    def hydrate_tweets_batch(self, tweet_ids: List[str], batch_size: int = 100) -> List[Dict]:
        """
        Hydrate a batch of tweet IDs.
        
        Args:
            tweet_ids: List of tweet ID strings
            batch_size: Number of tweets to fetch per API call (max 100)
            
        Returns:
            List of tweet dictionaries with available data
        """
        tweets_data = []
        
        # Process in batches
        for i in range(0, len(tweet_ids), batch_size):
            batch_ids = tweet_ids[i:i + batch_size]
            
            logger.info(f"Hydrating batch {i//batch_size + 1}: tweets {i+1}-{min(i+batch_size, len(tweet_ids))}")
            
            try:
                # Fetch tweets with comprehensive fields
                response = self.client.get_tweets(
                    ids=batch_ids,
                    tweet_fields=[
                        'created_at', 'author_id', 'public_metrics', 
                        'context_annotations', 'conversation_id', 'lang',
                        'possibly_sensitive', 'referenced_tweets', 'reply_settings'
                    ],
                    user_fields=['username', 'name', 'verified', 'public_metrics'],
                    expansions=['author_id']
                )
                
                if response.data:
                    # Create user lookup
                    users = {user.id: user for user in response.includes.get('users', [])}
                    
                    for tweet in response.data:
                        user = users.get(tweet.author_id)
                        
                        tweet_data = {
                            'id': tweet.id,
                            'text': tweet.text,
                            'created_at': tweet.created_at.isoformat() if tweet.created_at else None,
                            'author_id': tweet.author_id,
                            'username': user.username if user else None,
                            'name': user.name if user else None,
                            'conversation_id': tweet.conversation_id,
                            'language': tweet.lang,
                            'retweet_count': tweet.public_metrics.get('retweet_count', 0) if tweet.public_metrics else 0,
                            'like_count': tweet.public_metrics.get('like_count', 0) if tweet.public_metrics else 0,
                            'reply_count': tweet.public_metrics.get('reply_count', 0) if tweet.public_metrics else 0,
                            'quote_count': tweet.public_metrics.get('quote_count', 0) if tweet.public_metrics else 0,
                            'possibly_sensitive': tweet.possibly_sensitive,
                            'hydrated_at': pd.Timestamp.now().isoformat(),
                            'status': 'success'
                        }
                        
                        tweets_data.append(tweet_data)
                
                # Track errors (deleted/private tweets)
                if response.errors:
                    for error in response.errors:
                        error_data = {
                            'id': error.get('resource_id'),
                            'text': None,
                            'error_message': error.get('detail', 'Unknown error'),
                            'error_type': error.get('title', 'Error'),
                            'hydrated_at': pd.Timestamp.now().isoformat(),
                            'status': 'error'
                        }
                        tweets_data.append(error_data)
                
                # Rate limit handling (tweepy handles this automatically with wait_on_rate_limit=True)
                time.sleep(1)  # Small delay to be respectful
                
            except Exception as e:
                logger.error(f"Error hydrating batch {i//batch_size + 1}: {e}")
                
                # Add error records for failed batch
                for tweet_id in batch_ids:
                    error_data = {
                        'id': tweet_id,
                        'text': None,
                        'error_message': str(e),
                        'error_type': 'API Error',
                        'hydrated_at': pd.Timestamp.now().isoformat(),
                        'status': 'error'
                    }
                    tweets_data.append(error_data)
        
        return tweets_data
    
    def hydrate_from_csv(self, csv_path: str, id_column: str = 'id', output_dir: str = 'data/hydrated') -> str:
        """
        Hydrate tweets from a CSV file containing tweet IDs.
        
        Args:
            csv_path: Path to CSV file with tweet IDs
            id_column: Name of column containing tweet IDs
            output_dir: Directory to save hydrated data
            
        Returns:
            Path to output file
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        
        if id_column not in df.columns:
            raise ValueError(f"Column '{id_column}' not found in CSV. Available columns: {df.columns.tolist()}")
        
        # Extract tweet IDs
        tweet_ids = df[id_column].astype(str).tolist()
        logger.info(f"Found {len(tweet_ids)} tweet IDs in {csv_path}")
        
        # Hydrate tweets
        hydrated_data = self.hydrate_tweets_batch(tweet_ids)
        
        # Create output dataframe
        hydrated_df = pd.DataFrame(hydrated_data)
        
        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_filename = Path(csv_path).stem
        output_path = output_dir / f"{input_filename}_hydrated.csv"
        
        hydrated_df.to_csv(output_path, index=False)
        
        # Log results
        success_count = len(hydrated_df[hydrated_df['status'] == 'success'])
        error_count = len(hydrated_df[hydrated_df['status'] == 'error'])
        
        logger.info(f"Hydration complete:")
        logger.info(f"  Successfully hydrated: {success_count} tweets")
        logger.info(f"  Errors/deleted tweets: {error_count} tweets")
        logger.info(f"  Output saved to: {output_path}")
        
        return str(output_path)
    
    def save_hydration_report(self, hydrated_df: pd.DataFrame, output_path: str):
        """Generate and save hydration report."""
        
        report = {
            'hydration_summary': {
                'total_tweets': len(hydrated_df),
                'successful': len(hydrated_df[hydrated_df['status'] == 'success']),
                'failed': len(hydrated_df[hydrated_df['status'] == 'error']),
                'success_rate': len(hydrated_df[hydrated_df['status'] == 'success']) / len(hydrated_df) * 100
            },
            'error_breakdown': hydrated_df[hydrated_df['status'] == 'error']['error_type'].value_counts().to_dict(),
            'language_distribution': hydrated_df[hydrated_df['status'] == 'success']['language'].value_counts().to_dict(),
            'hydration_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Hydration report saved to: {output_path}")


def hydrate_dataset(dataset_dir: str = "dataset", output_dir: str = "data/hydrated"):
    """
    Hydrate all CSV files in the dataset directory that appear to contain tweet IDs.
    
    Args:
        dataset_dir: Directory containing CSV files with tweet IDs
        output_dir: Directory to save hydrated data
    """
    if not TWEEPY_AVAILABLE:
        logger.error("tweepy not available. Cannot hydrate tweets.")
        logger.info("To install tweepy: pip install tweepy")
        return
    
    # Check for API credentials
    required_env_vars = ['TWITTER_BEARER_TOKEN']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please set your Twitter API credentials:")
        logger.info("export TWITTER_API_KEY='your_api_key'")
        logger.info("export TWITTER_API_SECRET='your_api_secret'") 
        logger.info("export TWITTER_BEARER_TOKEN='your_bearer_token'")
        logger.info("\\nNote: Twitter/X API access may require paid subscription.")
        return
    
    try:
        # Initialize hydrator
        hydrator = TweetHydrator()
        
        # Test API connection
        if not hydrator.test_api_connection():
            logger.error("API connection failed. Please check your credentials.")
            return
        
        # Find CSV files
        dataset_path = Path(dataset_dir)
        csv_files = list(dataset_path.glob("*.csv"))
        
        if not csv_files:
            logger.error(f"No CSV files found in {dataset_dir}")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        # Process each CSV file
        for csv_file in csv_files:
            logger.info(f"Processing {csv_file}")
            
            try:
                # Check if file has tweet IDs (vs full text)
                df_sample = pd.read_csv(csv_file, nrows=5)
                
                # Look for ID column
                id_column = None
                for col in ['id', 'tweet_id', 'status_id']:
                    if col in df_sample.columns:
                        id_column = col
                        break
                
                if not id_column:
                    logger.info(f"No tweet ID column found in {csv_file}, skipping")
                    continue
                
                # Check if already has full text
                if 'tweet' in df_sample.columns or 'text' in df_sample.columns:
                    text_col = 'tweet' if 'tweet' in df_sample.columns else 'text'
                    sample_text = df_sample[text_col].iloc[0]
                    
                    if isinstance(sample_text, str) and len(sample_text) > 20:
                        logger.info(f"{csv_file} already contains full text, skipping hydration")
                        continue
                
                # Hydrate tweets
                output_path = hydrator.hydrate_from_csv(csv_file, id_column, output_dir)
                
                # Generate report
                hydrated_df = pd.read_csv(output_path)
                report_path = Path(output_path).parent / f"{Path(output_path).stem}_report.json"
                hydrator.save_hydration_report(hydrated_df, report_path)
                
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")
                continue
        
        logger.info("Dataset hydration complete!")
        
    except Exception as e:
        logger.error(f"Hydration failed: {e}")


if __name__ == "__main__":
    # Example usage
    print("Twitter/X Tweet Hydration Utility")
    print("=" * 40)
    
    print("\\nThis script hydrates tweet IDs into full tweet objects using the Twitter/X API.")
    print("\\nRequired environment variables:")
    print("- TWITTER_API_KEY")
    print("- TWITTER_API_SECRET")
    print("- TWITTER_BEARER_TOKEN")
    print("- TWITTER_ACCESS_TOKEN (optional)")
    print("- TWITTER_ACCESS_TOKEN_SECRET (optional)")
    
    print("\\nIMPORTANT:")
    print("- Twitter/X API access may require paid subscription")
    print("- This script handles deleted/private tweets gracefully")
    print("- Rate limits are automatically handled")
    
    print("\\nCurrent dataset analysis:")
    
    # Check if current dataset needs hydration
    dataset_dir = Path("dataset")
    if dataset_dir.exists():
        csv_files = list(dataset_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files in dataset/")
        
        for csv_file in csv_files[:2]:  # Check first 2 files
            try:
                df_sample = pd.read_csv(csv_file, nrows=3)
                print(f"\\n{csv_file.name}:")
                print(f"  Columns: {df_sample.columns.tolist()}")
                
                if 'tweet' in df_sample.columns:
                    sample_text = df_sample['tweet'].iloc[0]
                    if isinstance(sample_text, str):
                        print(f"  Sample text: {sample_text[:100]}...")
                        print("  -> Already contains full text, hydration not needed")
                    else:
                        print("  -> May need hydration")
                        
            except Exception as e:
                print(f"  Error reading {csv_file}: {e}")
    
    print("\\nTo run hydration (if needed):")
    print("python src/data/hydrate_tweets.py")
    
    # Uncomment to actually run hydration
    # hydrate_dataset()