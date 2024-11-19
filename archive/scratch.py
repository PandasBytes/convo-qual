# # Plot and save confusion matrix
# classes = ['low', 'medium', 'high']
# conf_matrix_path = f"logs/conf_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
# os.makedirs("logs", exist_ok=True)
# plot_confusion_matrix(conf_matrix, classes, save_path=conf_matrix_path)
# print(f"Confusion matrix saved to {conf_matrix_path}")

# # Logging results
# log_file = "logs/satisfaction_classification_log.csv"

# log_entry = {
#     'timestamp': datetime.now(),
#     'accuracy': accuracy,
#     'precision_low': report['low']['precision'],
#     'recall_low': report['low']['recall'],
#     'f1_low': report['low']['f1-score'],
#     'precision_medium': report['medium']['precision'],
#     'recall_medium': report['medium']['recall'],
#     'f1_medium': report['medium']['f1-score'],
#     'precision_high': report['high']['precision'],
#     'recall_high': report['high']['recall'],
#     'f1_high': report['high']['f1-score'],
#     'notes': "Initial classification evaluation"
# }

# # Append log entry
# if os.path.exists(log_file):
#     logs_df = pd.read_csv(log_file)
# else:
#     logs_df = pd.DataFrame(columns=log_entry.keys())

# logs_df = logs_df._append(log_entry, ignore_index=True)
# logs_df.to_csv(log_file, index=False)

# print(f"Logged results to {log_file}")

# # Visualize improvement over time (accuracy)
# if len(logs_df) > 1:
#     plt.figure(figsize=(10, 6))
#     plt.plot(pd.to_datetime(logs_df['timestamp']), logs_df['accuracy'], marker='o', label='Accuracy')
#     plt.title("Model Accuracy Over Time")
#     plt.xlabel("Timestamp")
#     plt.ylabel("Accuracy")
#     plt.grid()
#     plt.legend()
#     plt.show()



# ========================
# EVAL for Regression
# ========================




# ##############################################
# ##### EVALS FOR LOW/MED/HIGH 3 CLASS, ML ##########
# ##############################################
# # import pandas as pd
# # from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
# # import seaborn as sns
# # import matplotlib.pyplot as plt

# # def calculate_metrics(df, true_col, pred_col):
# #     """
# #     Calculate accuracy, precision, recall, and F1-score.
    
# #     Args:
# #         df (pd.DataFrame): DataFrame containing true and predicted labels.
# #         true_col (str): Column name for true labels.
# #         pred_col (str): Column name for predicted labels.
        
# #     Returns:
# #         dict: Dictionary of evaluation metrics.
# #     """
# #     # True and predicted labels
# #     true_labels = df[true_col]
# #     predicted_labels = df[pred_col]
    
# #     # Calculate metrics
# #     metrics = classification_report(true_labels, predicted_labels, output_dict=True, zero_division=0)
# #     accuracy = accuracy_score(true_labels, predicted_labels)
    
# #     # Print metrics
# #     print("Classification Report:")
# #     print(classification_report(true_labels, predicted_labels))
# #     print(f"Accuracy: {accuracy:.2f}")
    
# #     return {
# #         "accuracy": round(accuracy,2),
# #         "precision_macro": round(metrics["macro avg"]["precision"],2),
# #         "recall_macro": round(metrics["macro avg"]["recall"],2),
# #         "f1_macro": round(metrics["macro avg"]["f1-score"],2)
# #     }

# # def plot_confusion_matrix(df, true_col, pred_col, labels=None):
# #     """
# #     Plot a confusion matrix for true and predicted labels.
    
# #     Args:
# #         df (pd.DataFrame): DataFrame containing true and predicted labels.
# #         true_col (str): Column name for true labels.
# #         pred_col (str): Column name for predicted labels.
# #         labels (list, optional): List of label names for the confusion matrix.
# #     """
# #     from sklearn.metrics import confusion_matrix
# #     import matplotlib.pyplot as plt
# #     import seaborn as sns

# #     # True and predicted labels
# #     true_labels = df[true_col]
# #     predicted_labels = df[pred_col]
    
# #     # Handle dynamic labels
# #     if labels is None:
# #         labels = sorted(set(true_labels).union(set(predicted_labels)))
    
# #     # Generate confusion matrix
# #     cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    
# #     # Plot confusion matrix
# #     plt.figure(figsize=(8, 6))
# #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
# #     plt.xlabel('Predicted Labels')
# #     plt.ylabel('True Labels')
# #     plt.title('Confusion Matrix')
# #     plt.show()


# # def evaluate_classification(df, true_col, pred_col):
# #     """
# #     Evaluate classification results with metrics and a confusion matrix.
    
# #     Args:
# #         df (pd.DataFrame): DataFrame containing true and predicted labels.
# #         true_col (str): Column name for true labels.
# #         pred_col (str): Column name for predicted labels.
        
# #     Returns:
# #         dict: Evaluation metrics including accuracy, precision, recall, and F1-score.
# #     """
# #     # Define unique labels if needed
# #     labels = df[true_col].unique().tolist()
    
# #     # Calculate and print metrics
# #     metrics = calculate_metrics(df, true_col, pred_col)
    
# #     # Plot confusion matrix
# #     plot_confusion_matrix(df, true_col, pred_col, labels=labels)
# #     return metrics
