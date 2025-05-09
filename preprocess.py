import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

def load_data():
    """
    Load NSL-KDD dataset. If the dataset is not present,
    provide instructions to download it.
    """
    try:
        train_path = 'data/KDDTrain+.txt'
        test_path = 'data/KDDTest+.txt'
        
        # Column names based on NSL-KDD dataset documentation
        column_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'class', 'difficulty'
        ]
        
        # Load data
        train_df = pd.read_csv(train_path, header=None, names=column_names)
        test_df = pd.read_csv(test_path, header=None, names=column_names)
        
        # Drop difficulty level column as it's not needed for classification
        train_df.drop('difficulty', axis=1, inplace=True)
        test_df.drop('difficulty', axis=1, inplace=True)
        
        print(f"Loaded training data: {train_df.shape}")
        print(f"Loaded testing data: {test_df.shape}")
        
        return train_df, test_df
    
    except FileNotFoundError:
        print("Dataset files not found. Please download the NSL-KDD dataset from: ")
        print("https://www.unb.ca/cic/datasets/nsl.html")
        print("Extract and place KDDTrain+.txt and KDDTest+.txt in the 'data' directory.")
        return None, None

def preprocess_data(train_df, test_df):
    """
    Preprocess the training and testing data.
    """
    if train_df is None or test_df is None:
        return None, None, None, None
    
    # Convert attack classes to binary labels
    # Normal = 0, Attack = 1
    train_df['binary_class'] = train_df['class'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['binary_class'] = test_df['class'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Convert categorical features to numerical using Label Encoding
    categorical_cols = ['protocol_type', 'service', 'flag']
    encoders = {}
    
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        train_df[col] = encoders[col].fit_transform(train_df[col])
        test_df[col] = encoders[col].transform(test_df[col])
    
    # Save the encoders for later use
    with open('models/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    # Separate features and target
    X_train = train_df.drop(['class', 'binary_class'], axis=1)
    y_train = train_df['binary_class']
    
    X_test = test_df.drop(['class', 'binary_class'], axis=1)
    y_test = test_df['binary_class']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save the scaler for later use
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Preprocessed training data: {X_train.shape}")
    print(f"Preprocessed testing data: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

def save_processed_data(X_train, y_train, X_test, y_test):
    """
    Save the preprocessed data to disk.
    """
    if X_train is None:
        return
    
    # Create numpy arrays
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)
    
    print("Preprocessed data saved successfully!")

if __name__ == "__main__":
    print("Loading and preprocessing the NSL-KDD dataset...")
    
    # Load data
    train_df, test_df = load_data()
    
    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)
    
    # Save processed data
    save_processed_data(X_train, y_train, X_test, y_test) 