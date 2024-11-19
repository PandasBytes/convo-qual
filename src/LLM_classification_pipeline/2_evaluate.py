"""
Script for Evaluating and Logging Classification Performance of LLM Predictions.

This script processes conversation satisfaction prediction results for multiple prompts and generates evaluation metrics.
It includes functionality for individual and batch evaluations, logging results, and maintaining a history of runs.

1. Load prompt labels from `prompts.yaml`.
2. For each prompt:
   - Load scored conversation data from CSV.
   - Preprocess and clean the data.
   - Compute evaluation metrics (accuracy, precision, recall, F1-score).
   - Append metrics to a log CSV (`eval.csv`) for tracking and analysis.
3. Results are saved incrementally to maintain a historical record of evaluations.

Usage:
- Single Prompt: Uncomment and configure the "Eval for just 1 run" section with appropriate file paths.
- Batch Run: Ensure `prompts.yaml` is properly set up and execute the script to process all runs.

Paths:
- Input data: `/data/output/LLM/sample_scored_PROMPT{label}.csv`
- Log file: `/logs/eval.csv`
- Prompt configuration: `prompts.yaml`
"""

import openai
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import yaml
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import read_yaml, extract_yaml_labels
from datetime import datetime
import os
import matplotlib.pyplot as plt
import re

PROMPTS_PATH = 'prompts.yaml'
# ========================
# Eval for Single Run
# ========================
# label = "cleaned"
# INPUT_PATH = f"/Users/RiRi/Desktop/github/convo-quality/data/output/LLM/sample_scored_PROMPT{label}.csv"


# scored = pd.read_csv(INPUT_PATH)
# #cleanup, lowercase 
# #Grab only those with valid predicted_label output
# scored = scored[~scored['predicted_label'].astype(str).str.startswith('"')]
# scored = scored.applymap(lambda x: x.lower() if isinstance(x, str) else x)
# actual = scored['satisfaction_rating']
# pred = scored['predicted_label']

# # Evaluate performance
# accuracy = accuracy_score(actual, pred)
# report = classification_report(actual, pred, output_dict=True)
# report_df = pd.DataFrame(report).transpose()
# conf_matrix = confusion_matrix(actual, pred)

# print(f"Eval for PROMPT {label}")
# print(f"Accuracy: {accuracy:.2f}")
# # print("Classification Report:")
# print(report_df)

# ==============================
# Eval for Multiple Batch Runs
# ==============================
#csv_file = "/Users/RiRi/Desktop/github/convo-quality/src/LLM_classification_pipeline/logs/eval_gpt4.csv" #gpt4o mini


import pandas as pd
from sklearn.metrics import classification_report

# Initialize or load existing log file
csv_file = "/Users/RiRi/Desktop/github/convo-quality/src/LLM_classification_pipeline/logs/eval.csv"
try:
    log_df = pd.read_csv(csv_file)
except FileNotFoundError:
    log_df = pd.DataFrame()

def log_classification_report(y_true, y_pred, run_label):
    global log_df
    # Generate the classification report as dict
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().reset_index().round(3)
    report_df["run_label"] = run_label
    log_df = pd.concat([log_df, report_df], ignore_index=True)
    log_df.to_csv(csv_file, index=False)

#get the labels of prompts, to label each run
yaml_path = PROMPTS_PATH

try:
    yaml_data = read_yaml(yaml_path)
    labels = extract_yaml_labels(yaml_data)
    print("Extracted Labels:", labels)

except Exception as e:
    print("An error occurred:", str(e))

print(labels)

for label in labels: 
    INPUT_PATH = f"/Users/RiRi/Desktop/github/convo-quality/data/output/LLM/sample_scored_PROMPT{label}.csv"
    #INPUT_PATH = '/Users/RiRi/Desktop/github/convo-quality/data/output/LLM/gpt4/sample_scored_PROMPTcleaned.csv' #gpt4o mini
    scored = pd.read_csv(INPUT_PATH)
    #cleanup, lowercase 
    #Grab only those with valid predicted_label output
    scored = scored[~scored['predicted_label'].astype(str).str.startswith('"')]
    scored = scored.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    actual = scored['satisfaction_rating']
    pred = scored['predicted_label']
    log_classification_report(actual, pred, label)
