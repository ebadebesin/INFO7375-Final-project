import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
import pickle

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to load all CSV files from multiple folders
def load_all_csv(root_folders):
    all_data = []
    for root_folder in root_folders:
        for subdir, _, files in os.walk(root_folder):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(subdir, file)
                    print(f"Loading: {file_path}")
                    df = pd.read_csv(file_path)
                    df['Source'] = root_folder  # Add source folder info
                    all_data.append(df)
    
    if not all_data:
        raise ValueError("No CSV files found in the specified folders.")
        
    return pd.concat(all_data, ignore_index=True)

def preprocess_data(df):
    """
    Preprocess the loaded data
    """
    # Display basic information
    print("\nDataset Information:")
    print(f"Shape: {df.shape}")
    print("\nClass distribution:")
    print(df['State'].value_counts())
    print(f"Percentage of fall cases: {df['State'].mean() * 100:.2f}%")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values:")
        print(missing_values[missing_values > 0])
        # Fill missing values or drop rows with missing values
        df = df.dropna()
        print(f"Shape after handling missing values: {df.shape}")
    else:
        print("\nNo missing values found.")
    
    # Group by Sequence to analyze time series
    sequence_groups = df.groupby('Sequence')
    sequence_lengths = sequence_groups.size()
    print(f"\nNumber of sequences: {len(sequence_lengths)}")
    print(f"Average sequence length: {sequence_lengths.mean():.2f}")
    
    # Drop the Source column before model training if you don't need it as a feature
    if 'Source' in df.columns:
        df = df.drop('Source', axis=1)
    
    return df

def create_features_and_labels(df):
    """
    Create features and labels for the model
    """
    # Features and target
    X = df.drop(['State', 'Sequence'], axis=1)
    y = df['State']
    
    # Split the data preserving sequence information
    sequences = df['Sequence'].unique()
    train_seq, temp_seq = train_test_split(sequences, test_size=0.3, random_state=42)
    val_seq, test_seq = train_test_split(temp_seq, test_size=0.5, random_state=42)
    
    # Create train, validation, and test sets
    train_idx = df[df['Sequence'].isin(train_seq)].index
    val_idx = df[df['Sequence'].isin(val_seq)].index
    test_idx = df[df['Sequence'].isin(test_seq)].index
    
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for future use
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, X_train.columns, test_idx, df

def balance_dataset(X_train, y_train):
    """
    Balance the dataset using SMOTE
    """
    print("\nBalancing dataset with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Original class distribution: {pd.Series(y_train).value_counts()}")
    print(f"Balanced class distribution: {pd.Series(y_train_balanced).value_counts()}")
    
    return X_train_balanced, y_train_balanced

def build_dense_model(input_shape):
    """
    Build a dense neural network model
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    
    return model

def build_lstm_model(X_train_seq):
    """
    Build an LSTM model for sequence data
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(128, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, model_type="dense"):
    """
    Train the neural network model
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        f'best_{model_type}_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test, model_type="dense"):
    """
    Evaluate the model on test data
    """
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Classification report
    print(f"\n{model_type.upper()} Model Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Fall', 'Fall'],
               yticklabels=['No Fall', 'Fall'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_type.upper()} Model')
    plt.savefig(f'{model_type}_confusion_matrix.png')
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_type.upper()} Model')
    plt.legend(loc="lower right")
    plt.savefig(f'{model_type}_roc_curve.png')
    
    return y_pred, y_pred_proba

def sequence_data_preparation(df):
    """
    Prepare sequence data for LSTM model
    """
    # Features and target
    features = df.drop(['State', 'Sequence'], axis=1).columns
    
    # Group by sequence
    sequences = []
    labels = []
    sequence_ids = []
    
    for seq_id, group in df.groupby('Sequence'):
        # Only include sequences with at least 3 timesteps
        if len(group) >= 3:
            # Get the features for this sequence
            seq_features = group[features].values
            
            # Get the label (most frequent state in this sequence)
            label = group['State'].mode()[0]
            
            sequences.append(seq_features)
            labels.append(label)
            sequence_ids.append(seq_id)
    
    # Pad sequences to the same length
    max_length = max(len(seq) for seq in sequences)
    
    X_padded = np.zeros((len(sequences), max_length, len(features)))
    for i, seq in enumerate(sequences):
        X_padded[i, :len(seq), :] = seq
    
    y_seq = np.array(labels)
    
    return X_padded, y_seq, sequence_ids

def make_prediction(model_path, scaler_path, input_data, is_sequence=False):
    """
    Make a prediction using the trained model
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Scale the input data
    if not is_sequence:
        # For dense model
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
    else:
        # For LSTM model
        # Assume input_data is already in the right shape for LSTM
        # We would need to scale each feature in the sequence
        input_reshaped = input_data.reshape(input_data.shape[0] * input_data.shape[1], input_data.shape[2])
        input_scaled = scaler.transform(input_reshaped)
        input_scaled = input_scaled.reshape(input_data.shape)
        prediction = model.predict(input_scaled)
    
    return prediction

def plot_training_history(history, model_type="dense"):
    """
    Plot the training history
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_type.upper()} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_type.upper()} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{model_type}_training_history.png')

def test_model_with_detailed_results(model_path, scaler_path, test_data, test_labels, feature_names, test_idx=None, original_df=None, is_sequence=False, model_type="dense"):
    """
    Test a trained model and display detailed results including predicted values
    """
    print(f"\nTesting {model_type.upper()} Model...")
    
    # Load the model
    model = load_model(model_path)
    
    # Make predictions
    if is_sequence:
        # For sequence data
        input_reshaped = test_data.reshape(test_data.shape[0] * test_data.shape[1], test_data.shape[2])
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        input_scaled = scaler.transform(input_reshaped)
        input_scaled = input_scaled.reshape(test_data.shape)
        y_pred_proba = model.predict(input_scaled)
        
        # Convert to binary predictions
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Create a DataFrame with results
        results_df = pd.DataFrame({
            'Sequence_ID': np.array(test_idx) if test_idx is not None else np.arange(len(test_labels)),
            'Actual': test_labels.flatten(),
            'Predicted': y_pred.flatten(),
            'Probability': y_pred_proba.flatten()
        })
        
    else:
        # For regular data
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        test_scaled = scaler.transform(test_data)
        y_pred_proba = model.predict(test_scaled)
        
        # Convert to binary predictions
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Create a DataFrame with results
        if original_df is not None and test_idx is not None:
            # Get original data for these test samples
            test_samples = original_df.iloc[test_idx].reset_index(drop=True)
            
            # Create results DataFrame with original features
            results_df = test_samples.copy()
            results_df['Actual'] = test_labels.values
            results_df['Predicted'] = y_pred
            results_df['Probability'] = y_pred_proba
        else:
            # Simple results DataFrame
            results_df = pd.DataFrame({
                'Sample_ID': np.arange(len(test_labels)),
                'Actual': test_labels,
                'Predicted': y_pred.flatten(),
                'Probability': y_pred_proba.flatten()
            })
    
    # Calculate metrics
    correct = (results_df['Actual'] == results_df['Predicted']).sum()
    total = len(results_df)
    accuracy = correct / total
    
    # Show results
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print(f"Correct Predictions: {correct} out of {total}")
    
    # Display incorrect predictions
    incorrect_df = results_df[results_df['Actual'] != results_df['Predicted']]
    print(f"\nIncorrect Predictions: {len(incorrect_df)} samples")
    
    if len(incorrect_df) > 0:
        print("\nSample of Incorrect Predictions:")
        print(incorrect_df.head(10))
    
    # Save detailed results to CSV
    results_filename = f"{model_type}_detailed_results.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"\nDetailed results saved to {results_filename}")
    
    return results_df

def main():
    """
    Main function to run the entire pipeline
    """
    # Define root folders to search for CSV files
    root_folders = ["datasets/02202025"]  # Add all folders here
    
    # Load all CSV files
    print("Loading data from multiple CSV files...")
    df = load_all_csv(root_folders)
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Create features and labels
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, test_idx, original_df = create_features_and_labels(df)
    
    # Balance the training dataset
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
    
    # ====== DENSE MODEL ======
    # Train Dense Neural Network
    print("\nTraining Dense Neural Network...")
    dense_model = build_dense_model(X_train.shape[1])
    dense_model, dense_history = train_model(dense_model, X_train_balanced, y_train_balanced, X_val, y_val)
    
    # Plot training history
    plot_training_history(dense_history, "dense")
    
    # Evaluate Dense Model
    y_pred_dense, y_proba_dense = evaluate_model(dense_model, X_test, y_test, "dense")
    
    # Save the model
    dense_model_path = 'fall_detection_dense_model.h5'
    dense_model.save(dense_model_path)
    
    # Test the Dense Model with detailed results
    test_model_with_detailed_results(
        dense_model_path,
        'scaler.pkl',
        X_test,
        y_test,
        feature_names,
        test_idx,
        original_df,
        is_sequence=False,
        model_type="dense"
    )
    
    # ====== LSTM MODEL ======
    # Prepare sequence data for LSTM
    print("\nPreparing sequence data for LSTM...")
    X_seq, y_seq, seq_ids = sequence_data_preparation(df)
    
    # Split sequence data
    X_train_seq, X_temp_seq, y_train_seq, y_temp_seq, train_seq_ids, temp_seq_ids = train_test_split(
        X_seq, y_seq, seq_ids, test_size=0.3, random_state=42
    )
    
    X_val_seq, X_test_seq, y_val_seq, y_test_seq, val_seq_ids, test_seq_ids = train_test_split(
        X_temp_seq, y_temp_seq, temp_seq_ids, test_size=0.5, random_state=42
    )
    
    # Balance sequence data
    # Convert to 2D for SMOTE then back to 3D
    shape_train = X_train_seq.shape
    X_train_seq_2d = X_train_seq.reshape(X_train_seq.shape[0], -1)
    
    smote = SMOTE(random_state=42)
    X_train_seq_2d_balanced, y_train_seq_balanced = smote.fit_resample(X_train_seq_2d, y_train_seq)
    
    X_train_seq_balanced = X_train_seq_2d_balanced.reshape(
        X_train_seq_2d_balanced.shape[0], 
        shape_train[1], 
        shape_train[2]
    )
    
    # Train LSTM Model
    print("\nTraining LSTM Model...")
    lstm_model = build_lstm_model(X_train_seq)
    lstm_model, lstm_history = train_model(lstm_model, X_train_seq_balanced, y_train_seq_balanced, 
                                           X_val_seq, y_val_seq, "lstm")
    
    # Plot training history
    plot_training_history(lstm_history, "lstm")
    
    # Evaluate LSTM Model
    y_pred_lstm, y_proba_lstm = evaluate_model(lstm_model, X_test_seq, y_test_seq, "lstm")
    
    # Save the LSTM model
    lstm_model_path = 'fall_detection_lstm_model.h5'
    lstm_model.save(lstm_model_path)
    
    # Test the LSTM Model with detailed results
    test_model_with_detailed_results(
        lstm_model_path,
        'scaler.pkl',
        X_test_seq,
        y_test_seq,
        feature_names,
        test_seq_ids,
        None,
        is_sequence=True,
        model_type="lstm"
    )
    
    print("\nModel training, evaluation and testing complete.")
    print(f"Models saved as '{dense_model_path}' and '{lstm_model_path}'")
    print("Scaler saved as 'scaler.pkl'")
    print("Detailed test results saved to CSV files.")

def test_existing_models():
    """
    Function to test existing trained models without retraining
    """
    # Define root folders to search for CSV files
    root_folders = ["datasets/02202025"]  # Add all folders here
    
    # Check if models exist
    dense_model_path = 'best_dense_model.keras'
    lstm_model_path = 'best_lstm_model.keras'
    scaler_path = 'scaler.pkl'
    
    if not os.path.exists(dense_model_path) or not os.path.exists(lstm_model_path) or not os.path.exists(scaler_path):
        print("Models or scaler not found. Please run the training process first.")
        return
        
    # Load all CSV files
    print("Loading data for testing...")
    df = load_all_csv(root_folders)
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Create features and labels
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, test_idx, original_df = create_features_and_labels(df)
    
    # Test Dense Model
    print("\nTesting Dense Model on test data...")
    dense_results = test_model_with_detailed_results(
        dense_model_path,
        scaler_path,
        X_test,
        y_test,
        feature_names,
        test_idx,
        original_df,
        is_sequence=False,
        model_type="dense"
    )
    
    # Prepare sequence data for LSTM testing
    print("\nPreparing sequence data for LSTM testing...")
    X_seq, y_seq, seq_ids = sequence_data_preparation(df)
    
    # Split sequence data (only need test set for testing)
    _, X_temp_seq, _, y_temp_seq, _, temp_seq_ids = train_test_split(X_seq, y_seq, seq_ids, test_size=0.3, random_state=42)
    _, X_test_seq, _, y_test_seq, _, test_seq_ids = train_test_split(X_temp_seq, y_temp_seq, temp_seq_ids, test_size=0.5, random_state=42)
    
    # Test LSTM Model
    print("\nTesting LSTM Model on test data...")
    lstm_results = test_model_with_detailed_results(
        lstm_model_path,
        scaler_path,
        X_test_seq,
        y_test_seq,
        feature_names,
        test_seq_ids,
        None,
        is_sequence=True,
        model_type="lstm"
    )
    
    print("\nModel testing complete. Detailed results saved to CSV files.")
    
    return dense_results, lstm_results

if __name__ == "__main__":
    # Uncomment the line below to run the full training pipeline
    main()
    
    # Or use this to test existing models without retraining
    # test_existing_models()


#Sample Output:

#     DENSE Model Evaluation:
#               precision    recall  f1-score   support

#            0       0.93      0.91      0.92        85
#            1       0.58      0.65      0.61        17

#     accuracy                           0.86       102
#    macro avg       0.75      0.78      0.76       102
# weighted avg       0.87      0.86      0.87       102

# WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 

# Testing DENSE Model...
# WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
# /Users/ebunadebesin/Desktop/NN-Final Project/evn_nnpr/lib/python3.11/site-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
#   warnings.warn(
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step

# Model Accuracy: 0.8333
# Correct Predictions: 85 out of 102

# Incorrect Predictions: 17 samples

# Sample of Incorrect Predictions:
#     Sequence     Time  Acceleration  AngularVelocity  State  ...  Impact2  Impact3  Actual  Predicted  Probability
# 7        104  10500.0       7638.90          4726.39      1  ...    27.11     0.00       1          0     0.001036
# 8        108  10900.0       9798.38         17349.84      1  ...    31.26     0.00       1          0     0.004212
# 9        120  12100.0      14140.23          4282.20      1  ...     0.00     0.00       1          0     0.000898
# 21       120  12100.0      30252.35          8875.12      1  ...    47.13     0.00       1          0     0.003898
# 33       104  10500.0      29658.95         25174.07      1  ...    28.57    19.29       1          0     0.000836
# 34       108  10900.0      25954.20          2968.24      1  ...   144.57     0.00       1          0     0.002024
# 48       148  14900.0      21757.88          7986.76      1  ...   194.14     7.57       1          0     0.000362
# 49       158  15900.0      17624.00          2905.40      1  ...    31.26     0.00       1          0     0.001948
# 60       104  10500.0       6354.82         10112.98      1  ...    19.29     0.00       1          0     0.001403
# 61       108  10900.0      10670.28         24643.91      1  ...    38.34     0.00       1          0     0.000341

# [10 rows x 12 columns]

# Detailed results saved to dense_detailed_results.csv