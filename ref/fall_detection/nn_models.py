import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

def weighted_binary_crossentropy(weights):
    """
    Weighted binary cross-entropy loss
    """
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='float32')
        weights_tensor = y_true * weights[1] + (1 - y_true) * weights[0]
        return K.mean(weights_tensor * K.binary_crossentropy(y_true, y_pred))
    return loss

def build_dense_model(input_shape):
    """
    Build a dense neural network model
    
    Args:
        input_shape: Number of input features
        
    Returns:
        Compiled Keras sequential model
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

def build_lstm_model(input_shape):
    """
    Build an LSTM model for sequence data
    
    Args:
        input_shape: Tuple of (timesteps, features)
        
    Returns:
        Compiled Keras LSTM model
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
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

def compile_model_with_weighted_loss(model, class_weights):
    """
    Compile a model with weighted binary cross-entropy loss
    
    Args:
        model: Keras model to compile
        class_weights: Dictionary of class weights {0: weight_0, 1: weight_1}
        
    Returns:
        Compiled model
    """
    loss_fn = weighted_binary_crossentropy([class_weights[0], class_weights[1]])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    
    return model



