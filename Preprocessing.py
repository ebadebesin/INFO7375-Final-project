import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pickle

def load_and_combine_datasets(path1, path2):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df = pd.concat([df1, df2], ignore_index=True)
    return df

def clean_dataset(df):
    # Strip whitespace from column headers
    df.columns = df.columns.str.strip()

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Filter valid states (0 or 1)
    df = df[df['State'].isin([0, 1])]

    # Sort by Sequence and Time (if present)
    if 'Time' in df.columns and 'Sequence' in df.columns:
        df = df.sort_values(by=['Sequence', 'Time']).reset_index(drop=True)
    elif 'Sequence' in df.columns:
        df = df.sort_values(by=['Sequence']).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df

def normalize_features(df, feature_columns):
    df.columns = df.columns.str.strip()
    for col in feature_columns:
        if col not in df.columns:
            raise KeyError(f"Missing expected column: '{col}' in dataset.")

    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df, scaler

def save_cleaned_data(df, scaler, output_data_path, scaler_path):
    df.to_csv(output_data_path, index=False)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    dataset1_path = 'data/Dataset1.csv'
    dataset2_path = 'data/Dataset2.csv'
    output_path = 'data/cleaned_combined_dataset.csv'
    scaler_output_path = 'data/feature_scaler.pkl'

    if not os.path.exists(dataset1_path) or not os.path.exists(dataset2_path):
        print("Error: One or both dataset paths are invalid.")
        exit(1)

    print("Loading datasets...")
    df = load_and_combine_datasets(dataset1_path, dataset2_path)
    print(f"Combined dataset shape: {df.shape}")

    print("Cleaning dataset...")
    df = clean_dataset(df)
    print(f"Cleaned dataset shape: {df.shape}")
    print(df['State'].value_counts())

    print("Normalizing features...")
    feature_cols = ['Acceleration', 'AngularVelocity']  # Use exact column name from dataset
    df, scaler = normalize_features(df, feature_cols)

    print("Saving cleaned data and scaler...")
    save_cleaned_data(df, scaler, output_path, scaler_output_path)
    print("Preprocessing completed successfully.")
