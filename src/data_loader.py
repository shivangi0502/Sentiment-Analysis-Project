from datasets import load_dataset
import pandas as pd

def load_twitter_sentiment_dataset_multiclass():
    """
    Loads the 'tweet_eval' dataset with the 'sentiment' subset from Hugging Face.
    Returns a pandas DataFrame with 'text' and 'sentiment' columns.
    """
    
    dataset = load_dataset("tweet_eval", "sentiment")

    
    df_train = pd.DataFrame(dataset['train'])
    df_val = pd.DataFrame(dataset['validation'])
    df_test = pd.DataFrame(dataset['test'])

    df = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # The 'label' column represents sentiment:
    # 0: negative
    # 1: neutral
    # 2: positive

    df.rename(columns={'label': 'sentiment'}, inplace=True)

    # Filter out neutral sentiment (label 1)
    # df = df[df['label'] != 1].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Map positive (2) to 1 for binary classification
    # df['sentiment'] = df['label'].replace(2, 1) # Map positive (2) to 1

    # Rename 'text' column if needed (tweet_eval already has 'text')
    # df.rename(columns={'tweet': 'text'}, inplace=True) # tweet_eval already uses 'text'

    return df[['text', 'sentiment']]

if __name__ == '__main__':
   
    df = load_twitter_sentiment_dataset_multiclass()
    print(df.head())
    print("\nSentiment Distribution")
    print(df['sentiment'].value_counts())
    print(f"Dataset loaded with {len(df)} samples.")
    df.to_csv('data/tweet_eval_sentiment_multiclass.csv', index=False) # Changed filename
    print("Dataset saved to data/tweet_eval_sentiment_multiclass.csv")