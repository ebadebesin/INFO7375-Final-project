import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
from nn_models import build_dense_model, build_lstm_model, compile_model_with_weighted_loss, weighted_binary_crossentropy
from sklearn.utils.class_weight import compute_class_weight

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
MODEL_PATH = 'best_dense_model_hypersearch_weighted.keras'
SCALER_PATH = 'scaler.pkl'
TRAIN_DATA_PATH = 'datasets/01252025'
N_FEATURES = 3  # Only using real features
EXPECTED_REAL_FEATURES = ['Time', 'Acceleration', 'AngularVelocity']
ALL_FEATURES_ORDER = EXPECTED_REAL_FEATURES
SEQUENCE_LENGTH = 10  # Number of time steps to look back

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

def load_and_preprocess_data(root_folder):
    """Loads and preprocesses data from the specified folder."""
    all_data = []
    if not os.path.isdir(root_folder):
        print(f"Error: Directory not found: {root_folder}")
        return None
        
    print(f"Loading data from: {root_folder}")

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

def create_sequences(data, sequence_length):
    """Create sequences of data for LSTM training."""
    X, y = [], []
    
    # Group data by sequence
    for _, group in data.groupby('Sequence'):
        features = group[EXPECTED_REAL_FEATURES].values
        labels = group['State'].values
        
        # Create sequences
        for i in range(len(features) - sequence_length + 1):
            X.append(features[i:(i + sequence_length)])
            y.append(labels[i + sequence_length - 1])
    
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Build and compile LSTM model."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    return model

def main():
    # --- 1. Load and Preprocess Data ---
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(TRAIN_DATA_PATH)
    if data is None:
        print("Exiting due to data loading issues.")
        return
    
    # --- 2. Scale Features ---
    print("Scaling features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[EXPECTED_REAL_FEATURES])
    
    # Save the scaler
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {SCALER_PATH}")
    
    # --- 3. Create Sequences ---
    print("Creating sequences...")
    X, y = create_sequences(data, SEQUENCE_LENGTH)
    print(f"Created {len(X)} sequences of length {SEQUENCE_LENGTH}")
    
    # --- 4. Split Data ---
    print("Splitting data into train and validation sets...")
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Calculate class weights and convert to float32
    pos_weight = float(len(y_train) / (2 * np.sum(y_train)))
    neg_weight = float(len(y_train) / (2 * (len(y_train) - np.sum(y_train))))
    print(f"Class weights - Positive: {pos_weight:.2f}, Negative: {neg_weight:.2f}")
    
    # --- 5. Build and Train Model ---
    print("Building and training model...")
    model = build_lstm_model((SEQUENCE_LENGTH, N_FEATURES))
    
    # Compile model with weighted loss
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=get_weighted_loss(pos_weight, neg_weight),
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # --- 6. Evaluate Model ---
    print("\nEvaluating model...")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # --- 7. Save Final Model ---
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()