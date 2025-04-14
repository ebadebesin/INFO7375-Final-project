import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_data_sequences(input_path="data/balanced_dataset.csv", timesteps=14):
    print("Loading balanced dataset...")
    df = pd.read_csv(input_path)

    # Drop non-numeric columns if present
    for col in ['Time', 'Timestamp']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Print class distribution
    if "State" in df.columns:
        print("\nSequence Preparation Class Distribution:")
        print(df["State"].value_counts())
    else:
        raise ValueError("Missing 'State' column in dataset.")

    # Separate features and labels
    feature_cols = [col for col in df.columns if col != "State"]
    X = df[feature_cols].values
    y = df["State"].values
    features = len(feature_cols)

    # Make length divisible by timesteps
    usable_len = (X.shape[0] // timesteps) * timesteps
    X = X[:usable_len]
    y = y[:usable_len]

    # Reshape into sequences
    X_seq = X.reshape((-1, timesteps, features))
    y_seq = y.reshape((-1, timesteps))[:, -1]  # Label from last timestep

    print(f"X shape: {X_seq.shape}, y shape: {y_seq.shape}")

    # Split into train/val/test sets
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_seq, y_seq, test_size=0.25, stratify=y_seq, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
    )

    # Save feature min/max for UI input range suggestion
    feature_stats = {
        'min': df[feature_cols].min().to_dict(),
        'max': df[feature_cols].max().to_dict()
    }

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_stats
