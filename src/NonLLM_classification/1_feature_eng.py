"""
Script Name: feature_eng.py
Purpose: This script performs feature eng/extraction from mwoz_1processed.csv
    Based off 1_NonLLM_Intent_and_Satisfaction.ipynb
Ouput: 
    data/output/features.csv: full data + feature set
    data/output/train.csv: 80%, train conversations
    data/output/test.csv: 20%, test conversations
Author: Emeri Zhang
Date: 2024-11-15
"""

import pandas as pd
import string
import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import brown
nltk.download("brown")
from nltk.corpus import stopwords
nltk.download('stopwords')

import spacy
nlp = spacy.load("en_core_web_sm")

#IN
INGEST_DATA_PATH = "data/output/mwoz_1processed.csv"

#OUT
PROCESSED_DATA_PATH = "data/output/features.csv"
TRAIN_DATA_PATH = "data/output/train.csv"
TEST_DATA_PATH = "data/output/train.csv"
################################################################################################################################
mwoz = pd.read_csv(INGEST_DATA_PATH)
#FEATURE: add sentiment per turn
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
mwoz["sentiment"] = mwoz["text"].apply(classify_sentiment)
print("Classifying Sentiment")

#FEATURE Measure politeness
def measure_politeness(text):
    politeness_phrases = ["please", "thank you", "would you mind", "could you", "I appreciate"]
    softeners = ["might", "perhaps", "possibly", "I was wondering if"]
    
    politeness_score = sum(1 for phrase in politeness_phrases if phrase in text.lower())
    softener_score = sum(1 for word in softeners if word in text.lower())
    
    return politeness_score + softener_score
mwoz["politeness"] = mwoz["text"].apply(measure_politeness)
print("Measuring Politeness")


#FEATURE: clause to sentence ratio
def calculate_clause_to_sentence_ratio(text):
    """
    Calculate the Clause-to-Sentence Ratio for a given text.
    """
    doc = nlp(text)
    num_sentences = len(list(doc.sents))
    num_clauses = 0

    for token in doc:
        # Identify clause indicators like VERB, AUX (auxiliary verbs)
        if token.dep_ in {"ROOT", "csubj", "csubjpass", "advcl", "relcl"}:
            num_clauses += 1

    # Avoid div by zero
    return num_clauses / num_sentences if num_sentences > 0 else 0
mwoz["clause_to_sentence_ratio"] = mwoz["text"].apply(calculate_clause_to_sentence_ratio)
print("Calculating clause to sentence ratio")

#FEATURE: 
def calculate_dialogue_depth(text):
    """
    Approximate dialogue depth by counting references to prior context.
    This is a simplified metric based on pronouns and discourse markers.
    """
    # Pronouns and discourse markers indicative of context dependency
    context_markers = {"he", "she", "it", "they", "this", "that", "these", "those", 
                       "because", "therefore", "however", "but", "so"}
    
    doc = nlp(text)
    depth = 0

    for token in doc:
        if token.text.lower() in context_markers:
            depth += 1

    return depth
mwoz["dialogue_depth"] = mwoz["text"].apply(calculate_dialogue_depth)
print("Calculating dialogue depth")

def calculate_lexical_diversity(text):
    """
    Calculate the Type-Token Ratio (TTR) for a given text.
    """
    tokens = text.split() 
    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens) if tokens else 0
mwoz["lexical_diversity"] = mwoz["text"].apply(calculate_lexical_diversity)
print("Calculating lexical diversity")

# Create a frequency distribution from the Brown corpus
brown_words = brown.words()
freq_dist = nltk.FreqDist(brown_words)

# Define thresholds for rare and common words
common_threshold = 1000  # Common if frequency > 1000
rare_threshold = 10      # Rare if frequency < 10

# Function to calculate rare/common word ratio
def calculate_rare_common_ratio(text):
    tokens = text.split()
    rare_count = sum(1 for token in tokens if freq_dist[token.lower()] < rare_threshold)
    common_count = sum(1 for token in tokens if freq_dist[token.lower()] > common_threshold)
    total_count = len(tokens)
    return {
        "rare_ratio": rare_count / total_count if total_count else 0,
        "common_ratio": common_count / total_count if total_count else 0
    }

# Apply the function to the text column
rare_common_ratios = mwoz["text"].apply(calculate_rare_common_ratio)

# Split the results into separate columns
mwoz["rare_ratio"] = rare_common_ratios.apply(lambda x: x["rare_ratio"])
# mwoz["common_ratio"] = rare_common_ratios.apply(lambda x: x["common_ratio"])
print("Calculating rare word ratios")

#Code for Punctuation Counts and Stopword Ratios
# Define stopwords
stop_words = set(stopwords.words("english"))

# Sample DataFrame for demonstration (replace with your actual data)
# mwoz = pd.DataFrame({"text": ["Hello! How are you?", "I'm fine. Thank you!", "Why...?"]})

# Function to calculate punctuation counts
def count_punctuation(text):
    punctuation_counts = {
        "?": text.count("?"),
        "!": text.count("!"),
        "...": text.count("...")
    }
    return punctuation_counts

# Function to calculate stopword ratio
def calculate_stopword_ratio(text):
    words = text.split()  # Split text into words
    stopword_count = sum(1 for word in words if word.lower() in stop_words)
    total_words = len(words)
    return stopword_count / total_words if total_words > 0 else 0

# Apply the functions to the text column
mwoz["punctuation_counts"] = mwoz["text"].apply(count_punctuation)
mwoz["stopword_ratio"] = mwoz["text"].apply(calculate_stopword_ratio)
print("Calculating punctuation and stopword ratios")

# Split punctuation counts into separate columns
punct_df = mwoz["punctuation_counts"].apply(pd.Series)
mwoz = pd.concat([mwoz, punct_df], axis=1).drop(columns=["punctuation_counts"])

# print(mwoz)
mwoz.to_csv('PROCESSED_DATA_PATH')
##############################
##SPLIT INTO TRAIN AND TEST###
##############################
from utils import stratified_train_test_split
# Load your dataset
file_path = PROCESSED_DATA_PATH 
data = pd.read_csv(file_path)

# Perform stratified split and save outputs
stratified_train_test_split(data, conv_id_col='conv_ID')

# train = pd.read_csv('data/output/train.csv')
# train
# if 'Unnamed: 0' in train.columns:
    # train = train.drop(columns=['Unnamed: 0'])

