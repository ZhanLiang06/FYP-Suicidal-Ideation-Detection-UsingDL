import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_analyze_data():
    """Load and perform comprehensive analysis on both datasets"""
    
    print("="*80)
    print("COMPREHENSIVE DATA ANALYSIS FOR SUICIDAL IDEATION DETECTION")
    print("="*80)
    
    # Load datasets
    try:
        twitter_df = pd.read_csv('twitter-suicidal_data.csv')
        reddit_df = pd.read_csv('suicidal_ideation_reddit_annotated.csv')
        
        print(f"\n1. RAW DATA OVERVIEW")
        print("-" * 50)
        
        # Twitter Data Analysis
        print(f"\nTWITTER DATASET:")
        print(f"Shape: {twitter_df.shape}")
        print(f"Columns: {twitter_df.columns.tolist()}")
        print(f"Data Types:\n{twitter_df.dtypes}")
        print(f"Missing Values:\n{twitter_df.isnull().sum()}")
        
        # Reddit Data Analysis
        print(f"\nREDDIT DATASET:")
        print(f"Shape: {reddit_df.shape}")
        print(f"Columns: {reddit_df.columns.tolist()}")
        print(f"Data Types:\n{reddit_df.dtypes}")
        print(f"Missing Values:\n{reddit_df.isnull().sum()}")
        
        return twitter_df, reddit_df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def analyze_text_features(df, dataset_name):
    """Analyze text features and characteristics"""
    
    print(f"\n2. TEXT ANALYSIS - {dataset_name.upper()}")
    print("-" * 50)
    
    # Identify text column
    text_col = None
    for col in df.columns:
        if 'text' in col.lower() or 'content' in col.lower() or 'post' in col.lower():
            text_col = col
            break
    
    if text_col is None:
        print("No text column found")
        return
    
    # Basic text statistics
    df['text_length'] = df[text_col].astype(str).str.len()
    df['word_count'] = df[text_col].astype(str).str.split().str.len()
    
    print(f"Text Length Statistics:")
    print(f"Mean: {df['text_length'].mean():.2f}")
    print(f"Median: {df['text_length'].median():.2f}")
    print(f"Std: {df['text_length'].std():.2f}")
    print(f"Min: {df['text_length'].min()}")
    print(f"Max: {df['text_length'].max()}")
    
    print(f"\nWord Count Statistics:")
    print(f"Mean: {df['word_count'].mean():.2f}")
    print(f"Median: {df['word_count'].median():.2f}")
    print(f"Std: {df['word_count'].std():.2f}")
    
    # Sentiment analysis
    print(f"\nSentiment Analysis:")
    sentiments = []
    for text in df[text_col].astype(str).head(1000):  # Sample for speed
        try:
            blob = TextBlob(text)
            sentiments.append(blob.sentiment.polarity)
        except:
            sentiments.append(0)
    
    sentiment_scores = pd.Series(sentiments)
    print(f"Mean Sentiment: {sentiment_scores.mean():.3f}")
    print(f"Sentiment Std: {sentiment_scores.std():.3f}")
    
    # Common words analysis
    all_text = ' '.join(df[text_col].astype(str).head(1000))
    words = re.findall(r'\b\w+\b', all_text.lower())
    word_freq = Counter(words)
    
    print(f"\nMost Common Words (top 10):")
    for word, count in word_freq.most_common(10):
        print(f"  {word}: {count}")
    
    return df

def analyze_labels(df, dataset_name):
    """Analyze label distribution and characteristics"""
    
    print(f"\n3. LABEL ANALYSIS - {dataset_name.upper()}")
    print("-" * 50)
    
    # Find label column
    label_col = None
    for col in df.columns:
        if 'label' in col.lower() or 'class' in col.lower() or 'target' in col.lower():
            label_col = col
            break
    
    if label_col is None:
        print("No label column found")
        return
    
    print(f"Label Distribution:")
    label_counts = df[label_col].value_counts()
    print(label_counts)
    
    print(f"\nLabel Percentages:")
    label_percentages = (label_counts / len(df)) * 100
    for label, percentage in label_percentages.items():
        print(f"  {label}: {percentage:.2f}%")
    
    # Check for class imbalance
    if len(label_counts) == 2:
        imbalance_ratio = label_counts.max() / label_counts.min()
        print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 2:
            print("⚠️  Significant class imbalance detected!")
    
    return df

def create_visualizations(df_twitter, df_reddit):
    """Create comprehensive visualizations"""
    
    print(f"\n4. DATA VISUALIZATIONS")
    print("-" * 50)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Data Analysis Visualizations', fontsize=16)
    
    # Text length distributions
    if 'text_length' in df_twitter.columns:
        axes[0,0].hist(df_twitter['text_length'], bins=50, alpha=0.7, label='Twitter')
        axes[0,0].set_title('Text Length Distribution - Twitter')
        axes[0,0].set_xlabel('Text Length')
        axes[0,0].set_ylabel('Frequency')
    
    if 'text_length' in df_reddit.columns:
        axes[0,1].hist(df_reddit['text_length'], bins=50, alpha=0.7, label='Reddit')
        axes[0,1].set_title('Text Length Distribution - Reddit')
        axes[0,1].set_xlabel('Text Length')
        axes[0,1].set_ylabel('Frequency')
    
    # Word count distributions
    if 'word_count' in df_twitter.columns:
        axes[0,2].hist(df_twitter['word_count'], bins=30, alpha=0.7)
        axes[0,2].set_title('Word Count Distribution - Twitter')
        axes[0,2].set_xlabel('Word Count')
        axes[0,2].set_ylabel('Frequency')
    
    # Label distributions
    twitter_label_col = None
    for col in df_twitter.columns:
        if 'label' in col.lower() or 'class' in col.lower():
            twitter_label_col = col
            break
    
    if twitter_label_col:
        label_counts = df_twitter[twitter_label_col].value_counts()
        axes[1,0].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title('Label Distribution - Twitter')
    
    reddit_label_col = None
    for col in df_reddit.columns:
        if 'label' in col.lower() or 'class' in col.lower():
            reddit_label_col = col
            break
    
    if reddit_label_col:
        label_counts = df_reddit[reddit_label_col].value_counts()
        axes[1,1].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('Label Distribution - Reddit')
    
    # Dataset comparison
    dataset_sizes = [len(df_twitter), len(df_reddit)]
    axes[1,2].bar(['Twitter', 'Reddit'], dataset_sizes, color=['skyblue', 'lightcoral'])
    axes[1,2].set_title('Dataset Sizes Comparison')
    axes[1,2].set_ylabel('Number of Samples')
    
    plt.tight_layout()
    plt.savefig('data_analysis_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'data_analysis_visualizations.png'")

def analyze_data_quality(df, dataset_name):
    """Analyze data quality issues"""
    
    print(f"\n5. DATA QUALITY ANALYSIS - {dataset_name.upper()}")
    print("-" * 50)
    
    # Missing values
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    print("Missing Values Analysis:")
    for col, missing_count in missing_data.items():
        if missing_count > 0:
            print(f"  {col}: {missing_count} ({missing_percentage[col]:.2f}%)")
    
    # Duplicate analysis
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    # Text quality analysis
    text_col = None
    for col in df.columns:
        if 'text' in col.lower() or 'content' in col.lower():
            text_col = col
            break
    
    if text_col:
        # Empty or very short texts
        empty_texts = (df[text_col].astype(str).str.strip() == '').sum()
        short_texts = (df[text_col].astype(str).str.len() < 10).sum()
        
        print(f"\nText Quality Issues:")
        print(f"  Empty texts: {empty_texts} ({empty_texts/len(df)*100:.2f}%)")
        print(f"  Very short texts (<10 chars): {short_texts} ({short_texts/len(df)*100:.2f}%)")

def generate_summary_report(df_twitter, df_reddit):
    """Generate a comprehensive summary report"""
    
    print(f"\n6. SUMMARY REPORT")
    print("="*80)
    
    print(f"\nDATASET OVERVIEW:")
    print(f"  Twitter Dataset: {len(df_twitter)} samples, {len(df_twitter.columns)} features")
    print(f"  Reddit Dataset: {len(df_reddit)} samples, {len(df_reddit.columns)} features")
    
    # Data characteristics
    print(f"\nKEY FINDINGS:")
    
    # Text characteristics
    twitter_text_col = None
    for col in df_twitter.columns:
        if 'text' in col.lower() or 'content' in col.lower():
            twitter_text_col = col
            break
    
    if twitter_text_col:
        avg_length = df_twitter[twitter_text_col].astype(str).str.len().mean()
        print(f"  Average text length (Twitter): {avg_length:.1f} characters")
    
    # Label distribution summary
    twitter_label_col = None
    for col in df_twitter.columns:
        if 'label' in col.lower() or 'class' in col.lower():
            twitter_label_col = col
            break
    
    if twitter_label_col:
        label_dist = df_twitter[twitter_label_col].value_counts()
        print(f"  Twitter label distribution: {dict(label_dist)}")
    
    reddit_label_col = None
    for col in df_reddit.columns:
        if 'label' in col.lower() or 'class' in col.lower():
            reddit_label_col = col
            break
    
    if reddit_label_col:
        label_dist = df_reddit[reddit_label_col].value_counts()
        print(f"  Reddit label distribution: {dict(label_dist)}")
    
    print(f"\nRECOMMENDATIONS:")
    print(f"  1. Consider data preprocessing for text normalization")
    print(f"  2. Address class imbalance if present")
    print(f"  3. Handle missing values appropriately")
    print(f"  4. Consider text augmentation for minority classes")
    print(f"  5. Validate data quality before model training")

def main():
    """Main analysis function"""
    
    # Load and analyze data
    df_twitter, df_reddit = load_and_analyze_data()
    
    if df_twitter is not None and df_reddit is not None:
        # Analyze text features
        df_twitter = analyze_text_features(df_twitter, "Twitter")
        df_reddit = analyze_text_features(df_reddit, "Reddit")
        
        # Analyze labels
        df_twitter = analyze_labels(df_twitter, "Twitter")
        df_reddit = analyze_labels(df_reddit, "Reddit")
        
        # Data quality analysis
        analyze_data_quality(df_twitter, "Twitter")
        analyze_data_quality(df_reddit, "Reddit")
        
        # Create visualizations
        create_visualizations(df_twitter, df_reddit)
        
        # Generate summary report
        generate_summary_report(df_twitter, df_reddit)
        
        print(f"\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
    else:
        print("Failed to load data. Please check file paths and formats.")

if __name__ == "__main__":
    main() 