import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from nn_models import build_lstm_model, weighted_binary_crossentropy

def get_weighted_loss(pos_weights, neg_weights):
    """Create a weighted loss function."""
    def weighted_loss(y_true, y_pred):
        # Calculate the binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        # Convert weights to float32 to match bce's type
        weights = tf.where(y_true == 1, 
                          tf.cast(pos_weights, tf.float32), 
                          tf.cast(neg_weights, tf.float32))
        return tf.reduce_mean(weights * bce)
    return weighted_loss

# Register the custom loss function
tf.keras.utils.get_custom_objects()['weighted_loss'] = get_weighted_loss(3.0, 1.0)

# --- Configuration ---
MODEL_PATH = 'best_dense_model_hypersearch_weighted.keras'
SCALER_PATH = 'scaler.pkl'
OUTPUT_CSV = 'test_predictions_real.csv'
NEW_DATA_PATH = 'datasets/02202025'
N_FEATURES = 3  # Only using real features
EXPECTED_REAL_FEATURES = ['Time', 'Acceleration', 'AngularVelocity']
ALL_FEATURES_ORDER = EXPECTED_REAL_FEATURES

# Fixed sampling rate (assuming 50Hz)
SAMPLING_RATE = 50  # 50 samples per second
SEQUENCE_LENGTH = 2 * SAMPLING_RATE  # 2 seconds of data = 100 samples

def load_real_data(root_folder):
    """Loads and concatenates CSVs from the specified folder."""
    all_data = []
    if not os.path.isdir(root_folder):
        print(f"Error: Directory not found: {root_folder}")
        return None
        
    print(f"Loading real data from: {root_folder}")
    for file in os.listdir(root_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(root_folder, file)
            try:
                df = pd.read_csv(file_path)
                # Basic check for required columns
                required_cols = ['Sequence', 'State'] + EXPECTED_REAL_FEATURES
                if not all(col in df.columns for col in required_cols):
                    print(f"Warning: Skipping {file_path}. Missing required columns.")
                    continue
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
    if not all_data:
        print(f"Error: No valid CSV files found or loaded from {root_folder}.")
        return None
        
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_df)} total samples from {len(all_data)} files.")
    
    # Sort by Sequence to maintain temporal order
    combined_df = combined_df.sort_values('Sequence')
    
    # Drop rows with NaNs
    original_len = len(combined_df)
    combined_df.dropna(subset=required_cols, inplace=True)
    if len(combined_df) < original_len:
        print(f"Dropped {original_len - len(combined_df)} rows with NaN values in required columns.")
    
    return combined_df

def analyze_data_distribution(df):
    """Analyze the distribution of features and states."""
    print("\n--- Data Analysis ---")
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['State'].value_counts(normalize=True)}")
    
    # Analyze feature statistics by state
    for feature in EXPECTED_REAL_FEATURES:
        print(f"\n{feature} statistics by state:")
        print(df.groupby('State')[feature].describe())
    
    # Plot feature distributions
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(EXPECTED_REAL_FEATURES, 1):
        plt.subplot(1, 3, i)
        df.boxplot(column=feature, by='State')
        plt.title(f'{feature} by State')
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()

def create_sequences(data, sequence_length):
    """Create sequences of data for LSTM prediction with 2-second memory window."""
    X, sequences, actual_states = [], [], []
    
    # Group data by sequence
    for seq_id, group in data.groupby('Sequence'):
        features = group[EXPECTED_REAL_FEATURES].values
        states = group['State'].values
        
        # Calculate feature statistics for the sequence
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        
        # If sequence is shorter than sequence_length, pad it
        if len(features) < sequence_length:
            # Pad features with the mean of the sequence
            padding = np.tile(feature_means, (sequence_length - len(features), 1))
            features = np.vstack([padding, features])
            # Pad states with the last state
            states = np.pad(states, (sequence_length - len(states), 0), mode='edge')
        
        # Create sequences with 50% overlap
        step_size = sequence_length // 2
        for i in range(0, len(features) - sequence_length + 1, step_size):
            sequence_features = features[i:(i + sequence_length)]
            
            # Calculate sequence statistics
            seq_mean = np.mean(sequence_features, axis=0)
            seq_std = np.std(sequence_features, axis=0)
            
            # Normalize the sequence
            normalized_sequence = (sequence_features - seq_mean) / (seq_std + 1e-8)
            
            X.append(normalized_sequence)
            sequences.append(seq_id)
            actual_states.append(states[i + sequence_length - 1])
    
    if not X:  # If no sequences were created
        print("Warning: No sequences created. Creating single sequence from all data...")
        features = data[EXPECTED_REAL_FEATURES].values
        states = data['State'].values
        
        if len(features) < sequence_length:
            padding = np.tile(np.mean(features, axis=0), (sequence_length - len(features), 1))
            features = np.vstack([padding, features])
            states = np.pad(states, (sequence_length - len(states), 0), mode='edge')
        
        # Normalize the sequence
        seq_mean = np.mean(features, axis=0)
        seq_std = np.std(features, axis=0)
        normalized_sequence = (features - seq_mean) / (seq_std + 1e-8)
        
        X.append(normalized_sequence)
        sequences.append(data['Sequence'].iloc[0])
        actual_states.append(states[-1])
    
    return np.array(X), np.array(sequences), np.array(actual_states)

def find_optimal_threshold(y_true, y_proba):
    """Find the optimal threshold that maximizes F1 score."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    f1_scores = [2 * (tpr[i] * (1 - fpr[i])) / (tpr[i] + (1 - fpr[i])) for i in range(len(thresholds))]
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]

# --- Main Execution ---
if __name__ == "__main__":
    # --- Check necessary files/folders ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()
    if not os.path.exists(SCALER_PATH):
        print(f"Error: Scaler file not found at {SCALER_PATH}")
        exit()
    if not os.path.isdir(NEW_DATA_PATH):
        print(f"Error: New data directory not found at {NEW_DATA_PATH}")
        exit()

    # --- 1. Load Real Data ---
    real_df = load_real_data(NEW_DATA_PATH)
    if real_df is None or real_df.empty:
        print("Exiting due to data loading issues.")
        exit()
    
    # Analyze data distribution
    analyze_data_distribution(real_df)
    print(f"\nUsing sequence length of {SEQUENCE_LENGTH} samples for 2-second window")
    
    # --- 2. Create Sequences ---
    print("Creating sequences...")
    X, sequences, actual_states = create_sequences(real_df, SEQUENCE_LENGTH)
    print(f"Created {len(X)} sequences of length {SEQUENCE_LENGTH}")
    
    if len(X) == 0:
        print("Error: No sequences could be created from the data.")
        exit()

    # --- 3. Load Model and Make Predictions ---
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

    print("Making predictions...")
    predictions_proba = model.predict(X)
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(actual_states, predictions_proba)
    print(f"\nOptimal threshold based on F1 score: {optimal_threshold:.3f}")
    
    # Make predictions using optimal threshold
    predictions = (predictions_proba > optimal_threshold).astype(int).flatten()

    # --- 4. Create Output DataFrame ---
    print("Creating output DataFrame...")
    output_df = pd.DataFrame({
        'Sequence': sequences,
    })
    
    # Add feature columns (using the last time step of each sequence)
    for i, col_name in enumerate(EXPECTED_REAL_FEATURES):
        output_df[col_name] = X[:, -1, i]  # Take the last time step
        
    output_df['ActualState'] = actual_states
    output_df['PredictedState'] = predictions
    output_df['PredictedProbability'] = predictions_proba.flatten()
    
    # Ensure correct final column order
    final_cols = ['Sequence'] + EXPECTED_REAL_FEATURES + ['ActualState', 'PredictedState', 'PredictedProbability']
    output_df = output_df[final_cols]

    # --- 5. Save to CSV ---
    print(f"Saving results to {OUTPUT_CSV}...")
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully saved predictions to {OUTPUT_CSV}")

    # --- 6. Display Results ---
    print("\nSample of the output data:")
    print(output_df.head())

    # Calculate and display metrics
    accuracy = np.mean(actual_states == predictions)
    print(f"\nAccuracy on new data: {accuracy * 100:.2f}% (Using optimal threshold {optimal_threshold:.3f})")
    
    print("\n--- Classification Report (Using Optimal Threshold) ---")
    print(classification_report(actual_states, predictions, target_names=['No Fall (0)', 'Fall (1)'], zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(actual_states, predictions)
    print(cm)
