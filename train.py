import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def load_preprocessed_data():
    """
    Load the preprocessed data from disk.
    """
    try:
        X_train = np.load('data/X_train.npy')
        y_train = np.load('data/y_train.npy')
        X_test = np.load('data/X_test.npy')
        y_test = np.load('data/y_test.npy')
        
        print(f"Loaded training data: {X_train.shape}")
        print(f"Loaded testing data: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    except FileNotFoundError:
        print("Preprocessed data not found. Please run preprocess.py first.")
        return None, None, None, None

def build_model(input_shape):
    """
    Build a deep neural network model for binary classification.
    """
    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer - binary classification
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    print(model.summary())
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train the model and save it to disk.
    """
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Save best model during training
    os.makedirs('models', exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        filepath='models/ids_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=7,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.close()
    
    return history

if __name__ == "__main__":
    print("Training intrusion detection model...")
    
    # Load preprocessed data
    X_train, y_train, X_test, y_test = load_preprocessed_data()
    
    if X_train is not None:
        # Build model
        model = build_model(X_train.shape[1])
        
        # Train model
        history = train_model(model, X_train, y_train, X_test, y_test)
        
        print("Model training completed and saved to 'models/ids_model.h5'.") 