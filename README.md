# [Calculating Conversation Quality](https://interesting-wildflower-0cf.notion.site/Case-Valence-ML-AI-Data-Science-10c2531f31768016823ae32a9a7e5cef?pvs=27&qid=)


This repository contains scripts and data to preprocess, analyze, and evaluate the quality of task-oriented conversations, with a focus on MultiWOZ.

## Running LLM Code
  ```bash
  cd src/LLM_classification_pipeline
  python 0_preprocess.py
  python 1_predict.py
  python 2_evaluate.py 
```

## Folder Structure
    ├── data/                                       # Directory for storing raw and processed data
    │   ├── raw/                                    # Original data files (e.g. MultiWOZ dataset)
    │   ├── output/                                 # Processed files and analysis results
    │   ├── LLM_ingest/                             # Processed for LLM_classification_pipeline
    ├── myenv/                                      # Environment setup 
    ├── src/                                             
    │   ├── LLM_classification_pipeline/            # Core Task
    │   │   ├── utils.py                                # Helper functions
    │   │   ├── prompts.yaml                            # Store all prompts, for dynamic batch generation
    │   │   ├── 0_preprocess.py                         # Prep data for data/LLM_ingest/
    │   │   ├── 1_predict.py                            # Prompt generation, API call, combine pred w/ actual 
    │   │   ├── 2_evaluate.py                           # Generate evaluation metric
    │   │   ├── analysis_eval_and_errors.ipynb          # Generate charts, deep dive misclassifications
    │   ├── NonLLM_classification/                  # Dummy classic ML model + feature eng      
    │   │   ├── utils.py                                # Helper functions, mostly used in Python scripts
    │   │   ├── 0_preprocess.py                         # Python automating preprocessing steps from 1.ipynb
    │   │   ├── 0_Preprocess_EDA.ipynb                  # NB for preprocessing and EDA
    │   │   ├── 1A_NonLLM_Intent_and_Satisfaction.ipynb # NB for analyzing intent/satisfaction metrics no LLMs
    │   │   ├── 1B_NonLLM_Convo_Satisfaction.ipynb      # NB for convo level satisfaction without LLMs
    │   │   ├── 1_feature_eng.py                        # Script automating preprocessing steps from 1A.ipynb
    ├── README.md                                   # Documentation for the project
    ├── requirements.txt                            # List of Python dependencies


Analyze results using ```analysis_eval_and_errors.ipynb```

## Example EDA Visuals
generated in ```1B_NonLLM_Convo_Satisfaction.ipynb```
![alt text](image.png)