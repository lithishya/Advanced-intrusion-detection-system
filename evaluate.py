import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from tensorflow.keras.models import load_model

def load_data():
    """
    Load the preprocessed test data.
    """
    try:
        X_test = np.load('data/X_test.npy')
        y_test = np.load('data/y_test.npy')
        
        print(f"Loaded test data: {X_test.shape}")
        return X_test, y_test
    
    except FileNotFoundError:
        print("Test data not found. Please run preprocess.py first.")
        return None, None

def load_trained_model():
    """
    Load the trained model.
    """
    try:
        model = load_model('models/ids_model.h5')
        print("Model loaded successfully.")
        return model
    
    except (ImportError, IOError):
        print("Model file not found. Please train the model first.")
        return None

def predict(model, X_test):
    """
    Make predictions on the test data.
    """
    # Get probability predictions
    y_pred_prob = model.predict(X_test)
    
    # Convert probabilities to binary predictions (0 or 1)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    return y_pred, y_pred_prob

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and generate metrics and plots.
    """
    # Get predictions
    y_pred, y_pred_prob = predict(model, X_test)
    
    # Calculate metrics
    print("\nModel Evaluation Metrics:")
    print("========================")
    
    # Classification report (precision, recall, f1-score)
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Malicious'])
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Malicious'],
                yticklabels=['Normal', 'Malicious'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('models/roc_curve.png')
    plt.close()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('models/precision_recall_curve.png')
    plt.close()
    
    print("\nEvaluation complete. Results saved in the 'models' directory.")
    
    return y_pred, y_pred_prob

def analyze_predictions(y_test, y_pred):
    """
    Analyze prediction results to provide insights.
    """
    # Calculate true positives, false positives, true negatives, false negatives
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    total = len(y_test)
    normal_count = np.sum(y_test == 0)
    attack_count = np.sum(y_test == 1)
    
    print("\nPrediction Analysis:")
    print("===================")
    print(f"Total test samples: {total}")
    print(f"Normal traffic: {normal_count} ({normal_count/total*100:.2f}%)")
    print(f"Malicious traffic: {attack_count} ({attack_count/total*100:.2f}%)")
    print("\nResults:")
    print(f"True Positives (correctly identified attacks): {tp}")
    print(f"False Positives (normal traffic misclassified as attacks): {fp}")
    print(f"True Negatives (correctly identified normal traffic): {tn}")
    print(f"False Negatives (attacks misclassified as normal): {fn}")
    
    # Calculate attack detection rate
    attack_detection_rate = tp / attack_count * 100
    false_alarm_rate = fp / normal_count * 100
    
    print(f"\nAttack Detection Rate: {attack_detection_rate:.2f}%")
    print(f"False Alarm Rate: {false_alarm_rate:.2f}%")

if __name__ == "__main__":
    print("Evaluating intrusion detection model...")
    
    # Load data
    X_test, y_test = load_data()
    
    if X_test is None or y_test is None:
        exit()
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        exit()
    
    # Evaluate model
    y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)
    
    # Analyze predictions
    analyze_predictions(y_test, y_pred) 