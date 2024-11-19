"""
Script Name: 0_preprocess.py
Prepares data for LLM batch prompt creation by processing raw conversation data.

Author: Emeri Zhang
Date: 2024-11-17

Description:
This script processes raw conversation data to prepare it for LLM ingestion. Key steps include:
- Extract conversation turns and calculating average user ratings from the input text file.
- Categorizing satisfaction scores into 'Low', 'Medium', and 'High' using quantile thresholds:
  - Low: <= 30th percentile
  - High: > 90th percentile
  - Medium: Scores between these thresholds.
- Perform sentiment analysis using the VADER sentiment analyzer, classifying conversations as 'Positive', 'Neutral', or 'Negative.'
- Add a token count column, measuring the number of words in each conversation.

Input:
- Raw MWOZ.txt

Output:
- Full processed dataset saved to `conversation_data.csv` containing:
  - Conversation ID, text, average ratings, satisfaction rating, sentiment, and token count.
- Random sample (30% of the data) saved to `sample_conversation_data.csv` for testing purposes.
- Simplified evaluation dataset (`conv_id` and `satisfaction_rating`) saved to `convid_w_satisfaction_rating.csv` for metrics calculation.
"""

import pandas as pd
import warnings
from utils import process_conversation_data_debug, add_token_count_column

warnings.filterwarnings("ignore")
pd.set_option("display.width", 10000)

# ========================
# Constants
# ========================

RAW_DATA_PATH = "/Users/RiRi/Desktop/github/convo-quality/data/raw/MWOZ.txt"
PROCESSED_DATA_PATH = "/Users/RiRi/Desktop/github/convo-quality/data/LLM_ingest/conversation_data.csv"
PROCESSED_DATA_SAMPLE_PATH = "/Users/RiRi/Desktop/github/convo-quality/data/LLM_ingest/sample_conversation_data.csv"
FOR_EVAL_PATH = "/Users/RiRi/Desktop/github/convo-quality/data/LLM_ingest/convid_w_satisfaction_rating.csv"

# ========================
# Conversation Preprocessing
# ========================
def process_conversation_data_debug(file_path):
    """
    Processes raw conversation data to extract conversation turns and calculate average ratings.

    Args:
        file_path (str): Path to the raw conversation data file.

    Returns:
        pd.DataFrame: Processed conversation data as a DataFrame.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    conversations = []
    conversation_id = 1
    conversation_text = []

    for idx, line in enumerate(lines):
        line = line.strip()

        if not line:  # Skip empty lines
            continue

        parts = line.split("\t")

        # Process conversation turns
        if line.startswith("USER") or line.startswith("SYSTEM"):
            if len(parts) >= 2:
                speaker = parts[0]
                text = parts[1].replace("\n", "").strip()
                conversation_text.append(f"{speaker} {text}")

        # Process overall ratings
        if line.startswith("USER\tOVERALL"):
            try:
                overall_ratings = list(map(int, parts[3].split(",")))
                convo_avg = round(sum(overall_ratings) / len(overall_ratings), 2)
                conversations.append({
                    "conv_id": conversation_id,
                    "conv_text": "\n".join(conversation_text),
                    "average_rating": convo_avg,
                })
                conversation_id += 1
                conversation_text = []
            except (IndexError, ValueError) as e:
                print(f"Error processing OVERALL line {idx + 1}: {line}")
                print(f"Error details: {e}")

    return pd.DataFrame(conversations)


# Process the raw conversation data
conversation_df = process_conversation_data_debug(RAW_DATA_PATH)

# ========================
# Add Satisfaction Ratings
# ========================
# Define thresholds for satisfaction categorization
quantiles = conversation_df["average_rating"].quantile([i / 10 for i in range(1, 10)])
low_threshold = quantiles[0.3]
print(low_threshold)
high_threshold = quantiles[0.9]
print(high_threshold)

def categorize_satisfaction(score):
    """
    Categorizes satisfaction scores into 'Low', 'Medium', and 'High'.

    Args:
        score (float): The satisfaction score.

    Returns:
        str: The satisfaction category ('Low', 'Medium', 'High').
    """
    if score <= low_threshold:
        return "Low"
    elif score > high_threshold:
        return "High"
    else:
        return "Medium"

# Apply categorization
conversation_df["satisfaction_rating"] = conversation_df["average_rating"].apply(categorize_satisfaction)


# ========================
# Add Sentiment
# ========================
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def classify_sentiment(text):
    """Function to classify sentiment based on VADER compound score"""
    if pd.isna(text):  # Handle missing values
        return "Neutral"
    scores = analyzer.polarity_scores(text)
    if scores["compound"] >= 0.05:
        return "Positive"
    elif scores["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"
conversation_df["sentiment"] = conversation_df["conv_text"].apply(classify_sentiment)
print("Classifying Sentiment")

# ========================
# Add Token Count
# ========================
def add_token_count_column(conversation_df):
    """
    Adds a 'token_count' column to the DataFrame, containing the number of tokens in 'conv_text'.

    Args:
        conversation_df (pd.DataFrame): A DataFrame with a 'conv_text' column.

    Returns:
        pd.DataFrame: The DataFrame with an added 'token_count' column.
    """
    conversation_df["token_count"] = conversation_df["conv_text"].apply(lambda x: len(str(x).split()))
    return conversation_df

conversation_df = add_token_count_column(conversation_df)
print("Calculating Token Count")

# ========================
# Save Processed Data
# ========================
# Dump full dataset
print(conversation_df)
conversation_df.to_csv(PROCESSED_DATA_PATH, index=False)

# Generate sample containing shuffled batches of conversation IDs
shuffled_sampled_df = conversation_df.sample(frac=0.3, random_state=42).reset_index(drop=True)
shuffled_sampled_df.to_csv(PROCESSED_DATA_SAMPLE_PATH, index=False)

# Prepare and save id + rating, for evaluation metric calculation
for_eval = conversation_df[["conv_id", "satisfaction_rating"]]
for_eval.to_csv(FOR_EVAL_PATH, index=False)
