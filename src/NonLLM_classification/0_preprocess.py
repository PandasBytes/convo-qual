"""
Script Name: preprocess.py
Purpose: This script performs data preprocessing from txt into tabular form for the MultiWOZ dataset, 
    including calculating weighted average ratings.
    Based off 1_Preprocess_EDA.ipynb
Ouput: 
    data/mwoz_0preprocessed.csv: initial conversion of .txt to .csv
    data/mwoz_1preprocessed.csv: feature enriched
    data/output/train.csv: 80%, train conversations
    data/output/test.csv: 20%, test conversations
Author: Emeri Zhang
Date: 2024-11-16
"""

import pandas as pd
pd.set_option('display.width', 10000)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from utils import preprocess_mwoz_data, add_weighted_average_and_features, add_satisfaction_rating

# Constants
RAW_DATA_PATH = "data/raw/MWOZ.txt"
PROCESSED_DATA_PATH = "data/output/mwoz_1processed.csv"

#Read in and processd ata
preprocessed_data = preprocess_mwoz_data(RAW_DATA_PATH, PROCESSED_DATA_PATH )
print('Initial Read In from txt to csv completed + stored')

#Change score aggregation from average to weighted avg based on coverage
# List of rating columns
rating_columns = ["rating 1", "rating 2", "rating 3", "rating 4", "rating 5"]

# Add weighted average and additional features
updated_data = add_weighted_average_and_features(PROCESSED_DATA_PATH, rating_columns, PROCESSED_DATA_PATH)

# print(updated_data.head())

# Thresholds for categorization
low_threshold = 2.71  # 0.2 decile
high_threshold = 3.29  # 0.7 decile

# Add satisfaction ratings
updated_data = add_satisfaction_rating(PROCESSED_DATA_PATH, low_threshold, high_threshold, PROCESSED_DATA_PATH)

# print(updated_data.head())



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

