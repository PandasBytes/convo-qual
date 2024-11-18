"""
Script Name: 1_predict.py
Generates prompts for API calls and outputs satisfaction predictions.

Author: Emeri Zhang
Date: 2024-11-17

Description:
This script dynamically generates prompts from a YAML file (`prompts.yaml`), calls an API to classify conversations into satisfaction levels (Low/Medium/High), and combines predictions with actual scores for evaluation.

Key Steps:
- Reads processed conversation data and corresponding labels.
- Divides data into batches for efficient API calls.
- Generates prompts dynamically based on predefined YAML templates.
- Logs prompts and conversation IDs for traceability.
- Calls the OpenAI API to classify conversations into satisfaction levels.
- Combines predicted satisfaction levels with actual scores for comparison and evaluation.
- Outputs predictions to structured files for further analysis.

Input Files:
- Processed conversation data: `conversation_data.csv`
- Sample conversation data: `sample_conversation_data.csv`
- Predefined prompts: `prompts.yaml`
- Actual satisfaction labels: `convid_w_satisfaction_rating.csv`

Output Files:
- Batched API responses saved to timestamped CSV files in `/output`.
- Combined predictions and actual scores stored in a summary file.
"""

import openai
import pandas as pd
import warnings
from datetime import datetime
import yaml
from utils import (
    generate_batches,
    generate_batch_prompt_from_dataframe,
    log_prompt_and_ids,
    classify_conversations,
    combine_pred_and_actual_scores,
)

warnings.filterwarnings("ignore")

# ========================
# Constants
# ========================
PROCESSED_DATA_PATH = '/Users/RiRi/Desktop/github/convo-quality/data/LLM_ingest/conversation_data.csv'
PROCESSED_DATA_SAMPLE_PATH = "/Users/RiRi/Desktop/github/convo-quality/data/LLM_ingest/sample_conversation_data.csv"
PROMPTS_PATH = 'prompts.yaml'
LABELED_DATA_PATH = "/Users/RiRi/Desktop/github/convo-quality/data/LLM_ingest/convid_w_satisfaction_rating.csv"
OUTPUT_PATH_TEMPLATE = "/Users/RiRi/Desktop/github/convo-quality/data/output/LLM/{model_name}/sample_scored_PROMPT{label}.csv"
LOG_FILE = "logs/batch_log.txt"

# #gpt4
# OUTPUT_PATH_TEMPLATE = "/Users/RiRi/Desktop/github/convo-quality/data/output/LLM/gpt4/sample_scored_PROMPT{label}.csv"
# LOG_FILE = "logs/batch_log.txt"


# ========================
# Main Script
# ========================
if __name__ == "__main__":
    # Load the preprocessed DataFrame
    conversation_df = pd.read_csv(PROCESSED_DATA_SAMPLE_PATH)
    print("Initial Conversation Data:")
    print(conversation_df.head())

    # Define parameters
    # model_name = "gpt-3.5-turbo"
    model_name = "gpt-4o-mini"
    yaml_path = PROMPTS_PATH
    label = "cleaned"
    batch_limit = 50  # Set limit for testing new prompts   

    # # Generate shuffled batches of conversation IDs
    # shuffled_sampled_df = conversation_df.sample(frac=1, random_state=42).reset_index(drop=True)
    # batches = generate_batches(yaml_path, label, shuffled_sampled_df, model_name)

    batches = generate_batches(yaml_path, label, conversation_df, model_name)

    # Initialize results container
    all_batches_scored = []

    # Process each batch
    for i, batch in enumerate(batches[:batch_limit], start=1):
        total_size = len(batches)
        sample_df = conversation_df[conversation_df["conv_id"].isin(batch)]

        print(f"Processing Batch {i} of {batch_limit}, sampled out of {total_size} total batches")
        print(f"Conversations in this batch: {sample_df['conv_id'].tolist()}")

        # Generate the batch prompt
        batch_prompt = generate_batch_prompt_from_dataframe(yaml_path, label, sample_df)

        # Save the batch prompt to a file for reference
        with open("batch_prompt.txt", "w") as txt_file:
            txt_file.write(batch_prompt)

        # Log batch information
        log_prompt_and_ids(LOG_FILE, label, sample_df["conv_id"].tolist())

        # Call the API and make classifications
        llm_label = classify_conversations(batch_prompt, model_name)

        # Combine predictions with actual scores
        scored_convos = combine_pred_and_actual_scores(LABELED_DATA_PATH, llm_label)
        print(f"Finished processing Batch {i} of {total_size}")

        # Extend the results
        all_batches_scored.extend(scored_convos)

    # Save results
    output_path = OUTPUT_PATH_TEMPLATE.format(label=label)
    total_conv_ids = len(all_batches_scored)
    print(f"{total_conv_ids} convos scored, see {output_path}")

    # Convert results to a DataFrame and save to CSV
    df = pd.DataFrame(all_batches_scored)
    df.to_csv(output_path, index=False)
