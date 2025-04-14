import os
import pandas as pd
from Preprocessing import load_and_combine_datasets, clean_dataset, normalize_features, save_cleaned_data
from balance_data import balance_dataset
from prepare_sequence_data import prepare_data_sequences
from train_lstm_model import train_and_evaluate_lstm, load_model_for_prediction, get_user_input, predict_fall

# === Step 1: Preprocessing ===
print("Step 1: Loading and cleaning datasets...")
dataset1_path = "data/Dataset1.csv"
dataset2_path = "data/Dataset2.csv"
combined_path = "data/cleaned_combined_dataset.csv"
scaler_path = "data/feature_scaler.pkl"

if not os.path.exists(dataset1_path) or not os.path.exists(dataset2_path):
    print("Error: Dataset files not found.")
    exit(1)

df = load_and_combine_datasets(dataset1_path, dataset2_path)
df = clean_dataset(df)
feature_cols = ['Acceleration', 'AngularVelocity']
df, scaler = normalize_features(df, feature_cols)
save_cleaned_data(df, scaler, combined_path, scaler_path)
print("Cleaned and normalized data saved.")

# === Step 2: Balance Dataset ===
print("\nStep 2: Balancing dataset...")
balanced_path = balance_dataset(combined_path)
print("Balanced dataset saved as 'data/balanced_dataset.csv'")

# === Step 3: Prepare Sequence Data ===
print("\nStep 3: Preparing sequence data...")
X_train, X_val, X_test, y_train, y_val, y_test, feature_stats = prepare_data_sequences(balanced_path)

# === Step 4: Train and Evaluate LSTM ===
print("\nStep 4: Training and evaluating LSTM model...")
model = train_and_evaluate_lstm()

# === Step 5: User Prediction ===
print("\nStep 5: Predict based on user input...")
model = load_model_for_prediction("models/best_lstm_model.keras")
user_input = get_user_input()
predict_fall(model, user_input)
