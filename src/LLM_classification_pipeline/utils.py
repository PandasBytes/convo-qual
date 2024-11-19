# ========================
# Convo Preprocessing
# ========================

def process_conversation_data_debug(file_path):
    """
    Processes a conversation dataset to extract and structure dialogue data with average ratings.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    conversations = []
    conversation_id = 1
    conversation_text = []

    for idx, line in enumerate(lines):
        line = line.strip()

        # Skip empty lines and log unexpected formats
        if not line:
            continue

        parts = line.split("\t")

        # Process dialogue lines
        if line.startswith("USER") or line.startswith("SYSTEM"):
            if len(parts) >= 2:
                speaker = parts[0]
                text = parts[1]
                cleaned_text = text.replace("\\n", "")
                conversation_text.append(f"{speaker} {cleaned_text}")
            else:
                print(f"Skipping malformed dialogue line {idx + 1}: {line}")

        # Process 'OVERALL' ratings
        elif line.startswith("USER\tOVERALL"):
            try:
                overall_ratings = list(map(int, parts[3].split(",")))
                convo_avg = round(sum(overall_ratings) / len(overall_ratings), 2)

                # Append conversation
                conversations.append({
                    "conv_id": conversation_id,
                    "conv_text": "\n".join(conversation_text),
                    "average_rating": convo_avg,
                })
                conversation_id += 1
                conversation_text = []
            except (IndexError, ValueError) as e:
                print(f"Error parsing OVERALL line {idx + 1}: {line}")
                print(f"Error details: {e}")

        else:
            print(f"Skipping unexpected line {idx + 1}: {line}")

    return pd.DataFrame(conversations)

def add_token_count_column(conversation_df):
    """
    Adds a new column to the DataFrame with the token count of the text in the 'text' column.

    Args:
        conversation_df (pd.DataFrame): A DataFrame containing a 'text' column with string data.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column 'token_count',
                      containing the number of tokens in each row of the 'text' column.
    """
    # Tokenize the text and count the number of tokens
    conversation_df['token_count'] = conversation_df['conv_text'].apply(lambda x: len(str(x).split()))
    return conversation_df


# ========================
# Prompting
# ========================
def remove_last_two_words(text):
    words = text.split()  # Split the text into a list of words
    return " ".join(words[:-2])  # Join all words except the last two

import yaml
def generate_batch_prompt_from_dataframe(yaml_path, label, dataframe):
    """
    Generates a batch prompt by combining a YAML prompt with conversation IDs and texts from a DataFrame.

    Args:
        yaml_path (str): Path to the YAML file containing the prompt templates.
        label (str): The label of the prompt to use from the YAML file.
        dataframe (pd.DataFrame): DataFrame containing conversation data with 'conv_id' and 'conv_text'.
    Returns:
        str: The batch prompt ready for use.
    estimator: https://platform.openai.com/tokenizer
    """
    # Load the prompt from YAML
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    # Find the desired prompt
    prompt_begin = None
    for entry in data.get("prompts", []):
        if entry.get("label") == label:
            prompt_begin = entry.get("prompt")
            break
    
    if not prompt_begin:
        raise ValueError(f"No prompt found for label '{label}' in the YAML file.")
    
    # Create the batch prompt
    batch_prompt = prompt_begin.strip() + "\n\n"
    for i, row in dataframe.iterrows():
        batch_prompt += f"{row['conv_id']}. \"{row['conv_text']}\"\nClassification:\n\n"
    return batch_prompt

def log_prompt_and_ids(log_file, label, conv_ids):
    """
    Logs the prompt label and conversation IDs to a log file.

    Args:
        log_file (str): Path to the log file.
        label (str): The label of the prompt.
        conv_ids (list): List of conversation IDs included in the batch.
    """
    with open(log_file, "a") as log:
        log.write(f"Prompt Label: {label}\n")
        log.write(f"Conversation IDs: {', '.join(map(str, conv_ids))}\n")
        log.write("=" * 40 + "\n")  # Separator for readability

def generate_batches(yaml_path, label, dataframe, model_name) -> list:
    """
    Generates non-overlapping batches of conversation IDs that fit within the token limit for the specified model.

    Args:
        yaml_path (str): Path to the YAML file containing prompt templates.
        label (str): Label of the prompt to use.
        dataframe (pd.DataFrame): DataFrame with 'conv_id' and 'token_count' columns.
        model_name (str): The name of the model (e.g., 'gpt-3.5-turbo', 'text-davinci-003').

    Returns:
        list: A list of batches, where each batch is a list of conversation IDs.
    """
    # Dictionary of token limits by model
    model_token_limits = {
        'gpt-3.5-turbo': 4096, #this is WRONG should be  16385
        'gpt-4o-mini': 128000 #cheaper and more capable than GPT-3.5 Turbo
    }

    # Get the token limit for the specified model
    buffer = 5
    answer_tokens = dataframe.shape[0]*2
    token_limit = model_token_limits.get(model_name)-buffer-answer_tokens
    if not token_limit:
        raise ValueError(f"Unsupported model: {model_name}")

    # Load the prompt from YAML
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    # Retrieve the prompt for the given label
    prompt_text = None
    for entry in data.get("prompts", []):
        if entry.get("label") == label:
            prompt_text = entry.get("prompt")
            break

    if not prompt_text:
        raise ValueError(f"No prompt found for label '{label}' in the YAML file.")

    # Calculate the token count of the prompt
    prompt_token_count = len(prompt_text.split())

    # Generate batches
    batches = []
    current_batch = []
    current_tokens = prompt_token_count

    for _, row in dataframe.iterrows():
        if current_tokens + row['token_count'] > token_limit:
            # Start a new batch if the token limit is exceeded
            batches.append(current_batch)
            current_batch = []
            current_tokens = prompt_token_count

        # Add conversation to the current batch
        current_batch.append(row['conv_id'])
        current_tokens += row['token_count']

    # Append the final batch if it has content
    if current_batch:
        batches.append(current_batch)

    return batches

import openai
from openai import OpenAI
from openai import OpenAIError
import pandas as pd

def classify_conversations(batch_prompt, model_name):
    """
    Calls the OpenAI API with a batch_prompt and returns a DataFrame
    containing conversation IDs and their classifications.
    
    Parameters:
        batch_prompt (str): The full prompt to send to the API.
        model_name (str): The name of the OpenAI model to use (e.g., 'gpt-3.5-turbo').
    
    Returns:
        pd.DataFrame: A DataFrame with two columns: 'conv_id' and 'classification'.
    """
    try:
        # Call OpenAI API
        client = OpenAI()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": batch_prompt}],
            max_tokens=200,  # Adjust based on the expected output size
            temperature=0  # Keep deterministic for classification
        )
        # Extract the output text
        # print(response.choices[0].message)
        response_dict = response.model_dump() 
        response_msg = response_dict["choices"][0]["message"]["content"]
        return response_msg

    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        pass

def combine_pred_and_actual_scores(labeled_path: str, predictions: str):
    def parse_pred(predictions: str) -> list:
        lines = predictions.strip().split("\n")
        data = []
        for line in lines:
            if ". " in line:
                try:
                    conv_id, label = line.split(". ", 1)
                    data.append({"conv_id": int(conv_id), "predicted_label": label.strip()})
                except ValueError as e:
                    print(f"Error parsing line: '{line}'")
                    print(f"Error details: {e}")
            elif ":" in line:  # If '. ' not found, try splitting with ':'
                try:
                    conv_id, label = line.split(":", 1)
                    data.append({"conv_id": int(conv_id.strip()), "predicted_label": label.strip()})
                except ValueError as e:
                    print(f"Error parsing line (using ':'): '{line}'")
                    print(f"Error details: {e}")
            else:
                print(f"Skipping line with unexpected format: '{line}'")
        return data

    def parse_actuals(labeled_path: str) -> list:
        import csv
        with open(labeled_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            labeled_list = [row for row in reader]
        return labeled_list

    conv_w_actual_labels = parse_actuals(labeled_path)
    predicted_labels = parse_pred(predictions)

    # Deduplicate using a dictionary keyed by `conv_id`
    combined_data_dict = {}
    for actual in conv_w_actual_labels:
        for predicted in predicted_labels:
            if str(predicted['conv_id']) == actual['conv_id']:  # Ensure types match for comparison
                combined_data_dict[actual['conv_id']] = {
                    'conv_id': actual['conv_id'],
                    'satisfaction_rating': actual['satisfaction_rating'],
                    'predicted_label': predicted['predicted_label']
                }

    # Convert dictionary back to a list
    combined_data = list(combined_data_dict.values())
    return combined_data
# ========================
# Read in yaml
# ========================

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        # Load the YAML file
        return yaml.safe_load(file)

def extract_yaml_labels(yaml_data):
    # Ensure the input is a dictionary with a 'prompts' key
    if not isinstance(yaml_data, dict) or 'prompts' not in yaml_data:
        raise ValueError("Invalid YAML structure or missing 'prompts' key.")
    
    labels = []
    for item in yaml_data['prompts']:
        if 'label' in item:
            labels.append(item['label'])
    return labels
