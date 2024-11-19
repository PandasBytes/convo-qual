#####################################
### For Reading in from txt to csv###
#####################################

import pandas as pd
import numpy as np

def preprocess_mwoz_data(file_path, output_path):
    """
    Preprocess the MultiWOZ dataset from a TXT file into a CSV format with enriched features.
    
    Steps:
        1. Load the raw data.
        2. Parse ratings into individual columns.
        3. Extract domain and intent from the "domain-intent" column.
        4. Identify and insert breaks for conversation boundaries.
        5. Add derived features like average ratings and conversation/turn IDs.
        6. Save the preprocessed data to a CSV file.

    Args:
        file_path (str): Path to the raw MultiWOZ data file in TXT format.
        output_path (str): Path to save the preprocessed data as a CSV file.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Step 1: Load the MWOZ data
    mwoz = pd.read_csv(file_path, sep="\t", header=None, names=[
        "speaker", "text", "domain-intent", "rating 1", "rating 2", "rating 3", "rating 4", "rating 5"
    ])

    # Step 2: Parse ratings into columns
    ratings_split = mwoz["rating 1"].str.split(",", expand=True)
    mwoz["rating 1"] = ratings_split[0]
    mwoz["rating 2"] = ratings_split[1]
    mwoz["rating 3"] = ratings_split[2]
    mwoz["rating 4"] = ratings_split[3]
    mwoz["rating 5"] = ratings_split[4]
    mwoz.replace(to_replace=[None], value=np.nan, inplace=True)

    # Splice domain vs intent
    mwoz['domain'] = mwoz['domain-intent'].str.split("-").str[0]
    mwoz['intent'] = mwoz['domain-intent'].str.split("-").str[1]

    # Step 3: Identify breaks in conversations and create break lines
    break_indices = (mwoz["speaker"] == "USER") & (mwoz["text"] == "OVERALL")
    break_rows = pd.DataFrame({
        "speaker": ["BREAK"] * break_indices.sum(),
        "text": ["--- End of Conversation ---"] * break_indices.sum(),
        "domain-intent": [None] * break_indices.sum(),
        "rating 1": [None] * break_indices.sum(),
        "rating 2": [None] * break_indices.sum(),
        "rating 3": [None] * break_indices.sum(),
        "rating 4": [None] * break_indices.sum(),
        "rating 5": [None] * break_indices.sum()
    })
    print(mwoz.shape)
    # Add average rating column
    rating_columns = ['rating 1', 'rating 2', 'rating 3', 'rating 4', 'rating 5']
    mwoz[rating_columns] = mwoz[rating_columns].apply(pd.to_numeric, errors='coerce')
    mwoz['avg_rating'] = mwoz[rating_columns].mean(axis=1).round(2)

    # Add a marker column for sorting
    mwoz["IsBreak"] = break_indices

    # Concat original data with break rows
    mwoz = pd.concat([mwoz, break_rows], ignore_index=True).sort_index(kind="merge")

    # Remove marker column and reset index
    mwoz = mwoz.drop(columns=["IsBreak"]).reset_index(drop=True)

    # Enumerate convos, convo ID for sampling
    mwoz['conv_ID'] = ((mwoz['speaker'] == 'USER') & (mwoz['text'] == 'OVERALL')).cumsum()

    # Assign "OVERALL" rows to the previous convo ID
    mwoz['conv_ID'] = mwoz['conv_ID'].where(
        ~((mwoz['speaker'] == 'USER') & (mwoz['text'] == 'OVERALL')), 
        mwoz['conv_ID'] - 1
    )

    # Exclude invalid conversation IDs
    mwoz = mwoz[mwoz['conv_ID'] != 1000]

    # Per conv_id, set turn ID for each line
    mwoz['turn_id'] = mwoz.groupby('conv_ID').cumcount()

    # Reset index for consistency
    mwoz.reset_index(inplace=True, drop=True)

    # Save the preprocessed data
    mwoz.to_csv(output_path, index=False)
    print(f'Preprocessed data saved to {output_path}')

    return mwoz



def add_weighted_average_and_features(file_path, rating_columns, output_path=None):
    """
    Adds a weighted average column and additional features like word count to the dataset.

    Args:
        file_path (str): Path to the processed CSV file.
        rating_columns (list): List of rating column names to compute the weighted average.
        output_path (str, optional): Path to save the updated dataset. If None, the file won't be saved.

    Returns:
        pd.DataFrame: Updated DataFrame with new columns added.
    """
    # Load the dataset
    mwoz = pd.read_csv(file_path)

    # Ensure all rating columns are numeric
    mwoz[rating_columns] = mwoz[rating_columns].apply(pd.to_numeric, errors='coerce')

    # Define the weighted average calculation function
    def weighted_average(row, weights):
        """
        Calculate the weighted average of a row based on given weights.
        """
        weighted_sum = np.nansum(row * weights)
        sum_weights = np.nansum([w if not np.isnan(val) else 0 for val, w in zip(row, weights)])
        return weighted_sum / sum_weights if sum_weights > 0 else np.nan

    # Calculate coverage-based weights
    coverage = mwoz[rating_columns].notna().mean()
    weights = coverage / coverage.sum()

    # Apply the weighted average function row-wise
    mwoz["wt_avg_rating"] = mwoz[rating_columns].apply(lambda row: weighted_average(row, weights), axis=1).round(2)
    print('Weighted average calculated and added as "wt_avg_rating".')

    # Add word count feature
    mwoz['word_count'] = mwoz['text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    print('Word count calculated and added as "word_count".')

    # Save the updated dataset if output_path is provided
    if output_path:
        mwoz.to_csv(output_path, index=False)
        print(f'Updated dataset saved to {output_path}.')

    return mwoz


import pandas as pd
import os
from pathlib import Path

def add_satisfaction_rating(input, low_threshold, high_threshold, output_path=None):
    """
    Categorize satisfaction scores into 'Low', 'Medium', and 'High' based on thresholds
    and add a new column to the dataset.

    Args:
        input (str): Path to the processed CSV file with a 'wt_avg_rating' column.
        low_threshold (float): Threshold for categorizing 'Low' satisfaction.
        high_threshold (float): Threshold for categorizing 'High' satisfaction.
        output_path (str, optional): Path to save the updated dataset. If None, the file won't be saved.

    Returns:
        pd.DataFrame: Updated DataFrame with the 'satisfaction_rating' column added.
    """
    # Load the dataset
    if isinstance(input, (str, Path)) and os.path.exists(input):
        mwoz = pd.read_csv(input)
    if isinstance(input, pd.DataFrame):
        mwoz = input

    # Define the categorization function
    def categorize_satisfaction(score):
        """Categorize scores into 'Low', 'Medium', and 'High'."""
        if score <= low_threshold:
            return "Low"
        elif score > high_threshold:
            return "High"
        else:
            return "Medium"

    # Apply categorization to create the new column
    mwoz["satisfaction_rating"] = mwoz["wt_avg_rating"].apply(categorize_satisfaction)
    print('Satisfaction Rating set to Low/Medium/High.')

    # Save the updated dataset if output_path is provided
    if output_path:
        mwoz.to_csv(output_path, index=False)
        print(f'Updated dataset saved to {output_path}.')

    return mwoz


#######################
### For splitting data###
#######################

from sklearn.model_selection import train_test_split

def stratified_train_test_split(data, conv_id_col='conv_ID', train_frac=0.2, random_state=42):
    """
    Splits a dataset into stratified train and test sets based on conversation IDs (conv_ID),
    and outputs them as CSV files.
    
    Args:
        data (pd.DataFrame): The full dataset.
        conv_id_col (str): The column name representing conversation IDs.
        train_frac (float): Proportion of conversations to include in the training set.
        random_state (int): Random seed for reproducibility.

    Returns:
        None
    """
    # Extract unique conversation IDs
    unique_conv_ids = data[conv_id_col].unique()
    
    # Perform train-test split on the conversation IDs
    train_ids, test_ids = train_test_split(
        unique_conv_ids, 
        train_size=train_frac, 
        random_state=random_state
    )
    
    # Split the data based on the conversation IDs
    train_data = data[data[conv_id_col].isin(train_ids)]
    test_data = data[data[conv_id_col].isin(test_ids)]
    
    # Save train and test datasets as CSV files
    train_file_path = "data/output/LLM_ingest/train.csv"
    test_file_path = "data/output/LLM_ingest/test.csv"
    
    train_data.to_csv(train_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)
    
    print(f"Train set saved to {train_file_path} with {len(train_data)} rows and {train_data[conv_id_col].nunique()} unique conversations.")
    print(f"Test set saved to {test_file_path} with {len(test_data)} rows and {test_data[conv_id_col].nunique()} unique conversations.")


