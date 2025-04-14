import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, average_precision_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import compute_sample_weight
import os
import random

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def train_and_evaluate_lstm():
    print("Loading balanced dataset...")
    df = pd.read_csv("data/balanced_dataset.csv")

    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    timesteps = 7
    feature_cols = [col for col in df.columns if col != "State"]
    features = len(feature_cols)

    X = df[feature_cols].values
    y = df["State"].values

    usable_len = (X.shape[0] // timesteps) * timesteps
    X = X[:usable_len]
    y = y[:usable_len]

    X_seq = X.reshape((-1, timesteps, features))
    y_seq = y.reshape((-1, timesteps))[:, -1]

    print(f"X shape: {X_seq.shape}, y shape: {y_seq.shape}")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_seq, y_seq, test_size=0.25, stratify=y_seq, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42)

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    checkpoint_path = "models/best_lstm_model.keras"
    os.makedirs("models", exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True)
    ]

    print("Training LSTM model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=callbacks,
        sample_weight=sample_weights,
        verbose=1
    )

    print("\nEvaluating on test set...")
    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.3).astype(int)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print("PR AUC:", average_precision_score(y_test, y_prob))

    print("\nFall predicted count:", np.sum(y_pred == 1))
    print("No-fall predicted count:", np.sum(y_pred == 0))

    print(f"\nBest LSTM model saved at: {checkpoint_path}")
    return model

def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, return_sequences=True),
        BatchNormalization(),
        Dropout(0.4),

        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.4),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    return model

def get_user_input():
    print("\nPlease enter a single set of sequence values for all features:")
    df = pd.read_csv("data/balanced_dataset.csv")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    feature_cols = [col for col in df.columns if col != "State"]
    feature_ranges = {feature: (df[feature].min(), df[feature].max()) for feature in feature_cols}
    sequence = []

    for i in range(7):
        row = []
        print(f"\n--- Input for Timestep {i+1} ---")
        for feature in feature_cols:
            min_val, max_val = feature_ranges[feature]
            while True:
                try:
                    val = float(input(f"Enter {feature} ({min_val:.2f} to {max_val:.2f}): "))
                    if min_val <= val <= max_val:
                        row.append(val)
                        break
                    else:
                        print(f"Value out of range. Please enter between {min_val:.2f} and {max_val:.2f}.")
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")
        sequence.append(row)

    return np.array(sequence).reshape((1, 7, len(feature_cols)))

def predict_fall(model, input_sequence):
    prediction = model.predict(input_sequence)
    score = prediction[0][0]
    print(f"\nRaw prediction score: {score:.4f}")
    print("Prediction Result:")
    print("FALL" if score > 0.5 else "NO FALL")

def load_model_for_prediction(model_path="models/best_lstm_model.keras"):
    return load_model(model_path)
