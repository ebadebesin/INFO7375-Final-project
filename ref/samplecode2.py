import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
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
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

# def balance_dataset(X_train, y_train):
#     """
#     Balance the dataset using SMOTE
#     """
#     print("\nBalancing dataset with SMOTE...")
#     smote = SMOTE(random_state=42)
#     X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
#     print(f"Original class distribution: {pd.Series(y_train).value_counts()}")
#     print(f"Balanced class distribution: {pd.Series(y_train_balanced).value_counts()}")
    
#     return X_train_balanced, y_train_balanced

def balance_dataset(X_train, y_train):
    """
    Balance the dataset using SMOTE and Tomek Links
    """
    print("\nBalancing dataset with SMOTE and Tomek Links...")
    smote_tomek = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
    
    print(f"Original class distribution: {pd.Series(y_train).value_counts()}")
    print(f"Balanced class distribution: {pd.Series(y_train_balanced).value_counts()}")
    
    return X_train_balanced, y_train_balanced

def augment_data(X, y, augmentation_factor=2):
    """
    Augment the dataset by adding noise
    """
    augmented_X, augmented_y = [], []
    for _ in range(augmentation_factor):
        noise = np.random.normal(0, 0.01, X.shape)  # Add Gaussian noise
        augmented_X.append(X + noise)
        augmented_y.append(y)
    
    augmented_X = np.vstack(augmented_X)
    augmented_y = np.hstack(augmented_y)
    
    return augmented_X, augmented_y

def weighted_binary_crossentropy(weights):
    """
    Weighted binary cross-entropy loss
    """
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='float32')
        weights_tensor = y_true * weights[1] + (1 - y_true) * weights[0]
        return K.mean(weights_tensor * K.binary_crossentropy(y_true, y_pred))
    return loss

#######################

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
        f'best_{model_type}_model.keras',
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
    # plt.savefig(f'{model_type}_confusion_matrix.png')
    
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
    # plt.savefig(f'{model_type}_roc_curve.png')
    
    # Return predictions and actual values
    return y_pred.flatten(), y_pred_proba.flatten(), y_test
    # return y_pred, y_pred_proba



def sequence_data_preparation(df):
    """
    Prepare sequence data for LSTM model
    """
    # Features and target
    features = df.drop(['State', 'Sequence'], axis=1).columns
    
    # Group by sequence
    sequences = []
    labels = []
    
    for seq_id, group in df.groupby('Sequence'):
        # Only include sequences with at least 3 timesteps
        if len(group) >= 3:
            # Get the features for this sequence
            seq_features = group[features].values
            
            # Get the label (most frequent state in this sequence)
            label = group['State'].mode()[0]
            
            sequences.append(seq_features)
            labels.append(label)
    
    # Pad sequences to the same length
    max_length = max(len(seq) for seq in sequences)
    
    X_padded = np.zeros((len(sequences), max_length, len(features)))
    for i, seq in enumerate(sequences):
        X_padded[i, :len(seq), :] = seq
    
    y_seq = np.array(labels)
    
    return X_padded, y_seq

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
    X_train, X_val, X_test, y_train, y_val, y_test = create_features_and_labels(df)
    
    ########
    class_weights = {0: 1.0, 1: 2.0}  # Example weights
    # loss_fn = weighted_binary_crossentropy([class_weights[0], class_weights[1]])
    ############


    # Balance the training dataset
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
    
    # Augment the training dataset
    X_train_augmented, y_train_augmented = augment_data(X_train_balanced, y_train_balanced)

     # Train Dense Neural Network
    print("\nTraining Dense Neural Network...")
    dense_model = build_dense_model(X_train_augmented.shape[1])
    loss_fn = weighted_binary_crossentropy([class_weights[0], class_weights[1]])
    dense_model.compile(optimizer=Adam(learning_rate=0.001), loss=loss_fn, metrics=['accuracy'])

    dense_model.fit(X_train_augmented, y_train_augmented, epochs=100, batch_size=32, validation_data=(X_val, y_val))


    # # Train Dense Neural Network
    # print("\nTraining Dense Neural Network...")
    # dense_model = build_dense_model(X_train.shape[1])
    # dense_model, dense_history = train_model(dense_model, X_train_balanced, y_train_balanced, X_val, y_val)
    
    # Plot training history
    # plot_training_history(dense_history, "dense")
    
    # # Evaluate Dense Model
    # y_pred_dense, y_proba_dense = evaluate_model(dense_model, X_test, y_test, "dense")
    
    # Evaluate Dense Model
    y_pred_dense, y_proba_dense, y_test_dense = evaluate_model(dense_model, X_test, y_test, "dense")


    # Print 10 sample rows with predicted and actual values
    print("\nSample Predictions:")
    sample_indices = np.random.choice(len(X_test), 10, replace=False)  # Randomly select 10 samples
    for i in sample_indices:
        print(f"Sample {i + 1}:")
        print(f"Predicted: {y_pred_dense[i]}, Probability: {y_proba_dense[i]:.4f}, Actual: {y_test_dense.iloc[i]}")
        print("-" * 50)

    # Save the model
    # dense_model.save('fall_detection_dense_model.h5')
    
    


    # # Prepare sequence data for LSTM
    # print("\nPreparing sequence data for LSTM...")
    # X_seq, y_seq = sequence_data_preparation(df)
    
    # # Split sequence data
    # X_train_seq, X_temp_seq, y_train_seq, y_temp_seq = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42)
    # X_val_seq, X_test_seq, y_val_seq, y_test_seq = train_test_split(X_temp_seq, y_temp_seq, test_size=0.5, random_state=42)
    
    # # Balance sequence data
    # # Convert to 2D for SMOTE then back to 3D
    # shape_train = X_train_seq.shape
    # X_train_seq_2d = X_train_seq.reshape(X_train_seq.shape[0], -1)
    
    # smote = SMOTE(random_state=42)
    # X_train_seq_2d_balanced, y_train_seq_balanced = smote.fit_resample(X_train_seq_2d, y_train_seq)
    
    # X_train_seq_balanced = X_train_seq_2d_balanced.reshape(
    #     X_train_seq_2d_balanced.shape[0], 
    #     shape_train[1], 
    #     shape_train[2]
    # )
    
    # # Train LSTM Model
    # print("\nTraining LSTM Model...")
    # lstm_model = build_lstm_model(X_train_seq)
    # lstm_model, lstm_history = train_model(lstm_model, X_train_seq_balanced, y_train_seq_balanced, 
    #                                        X_val_seq, y_val_seq, "lstm")
    
    # # Plot training history
    # plot_training_history(lstm_history, "lstm")
    
    # # Evaluate LSTM Model
    # y_pred_lstm, y_proba_lstm = evaluate_model(lstm_model, X_test_seq, y_test_seq, "lstm")
    
    # Save the LSTM model
    # lstm_model.save('fall_detection_lstm_model.h5')
    
    print("\nModel training and evaluation complete.")
    # print("Models saved as 'fall_detection_dense_model.h5' and 'fall_detection_lstm_model.h5'")
    # print("Scaler saved as 'scaler.pkl'")

if __name__ == "__main__":
    main()



# Sample Output:

# Dataset Information:
# Shape: (666, 10)

# Class distribution:
# State
# 0    556
# 1    110
# Name: count, dtype: int64
# Percentage of fall cases: 16.52%

# No missing values found.

# Number of sequences: 113
# Average sequence length: 5.89
# Training set size: 476
# Validation set size: 88
# Test set size: 102

# Balancing dataset with SMOTE and Tomek Links...
# Original class distribution: State
# 0    389
# 1     87
# Name: count, dtype: int64
# Balanced class distribution: State
# 0    384
# 1    384
# Name: count, dtype: int64

# Training Dense Neural Network...

# DENSE Model Evaluation:
#               precision    recall  f1-score   support

#            0       0.94      0.92      0.93        85
#            1       0.63      0.71      0.67        17

#     accuracy                           0.88       102
#    macro avg       0.79      0.81      0.80       102
# weighted avg       0.89      0.88      0.88       102


# Sample Predictions:
# Sample 33:
# Predicted: 0, Probability: 0.0028, Actual: 0
# --------------------------------------------------
# Sample 6:
# Predicted: 0, Probability: 0.0376, Actual: 0
# --------------------------------------------------
# Sample 59:
# Predicted: 0, Probability: 0.0043, Actual: 0
# --------------------------------------------------
# Sample 41:
# Predicted: 0, Probability: 0.0005, Actual: 0
# --------------------------------------------------
# Sample 1:
# Predicted: 0, Probability: 0.0000, Actual: 0
# --------------------------------------------------
# Sample 75:
# Predicted: 1, Probability: 0.9984, Actual: 1
# --------------------------------------------------
# Sample 57:
# Predicted: 0, Probability: 0.0030, Actual: 0
# --------------------------------------------------
# Sample 61:
# Predicted: 0, Probability: 0.0019, Actual: 1
# --------------------------------------------------
# Sample 20:
# Predicted: 1, Probability: 0.9935, Actual: 0
# --------------------------------------------------
# Sample 38:
# Predicted: 0, Probability: 0.0000, Actual: 0
# --------------------------------------------------

# Model training and evaluation complete.
