import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import shap
import joblib
from preprocess import load_data, preprocess_data

def load_model_and_scaler():
    """Load the trained model and scaler"""
    model = tf.keras.models.load_model('models/ids_model.h5')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

def get_feature_names():
    """Get feature names from the dataset"""
    _, test_data = load_data()
    # Get feature names (excluding the target)
    feature_names = test_data.columns.tolist()
    feature_names.remove('attack_cat')
    return feature_names

def explain_model_predictions(num_samples=100):
    """Generate SHAP explanations for model predictions"""
    model, scaler = load_model_and_scaler()
    feature_names = get_feature_names()
    
    # Load and preprocess a small sample of test data
    _, test_data = load_data()
    test_sample = test_data.sample(num_samples, random_state=42)
    
    X_test, _ = preprocess_data(test_sample, scaler=scaler, is_training=False)
    
    # Create a background dataset for SHAP
    background = X_test[:100]
    
    # Use SHAP's DeepExplainer for neural networks
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_test)
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values[0], X_test, feature_names=feature_names, 
                     show=False, plot_size=(12, 8))
    plt.tight_layout()
    plt.savefig('static/images/shap_summary.png')
    plt.close()
    
    # Create waterfall plot for a single instance
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap.Explanation(values=shap_values[0][0], 
                                        base_values=explainer.expected_value[0],
                                        data=X_test[0], 
                                        feature_names=feature_names))
    plt.tight_layout()
    plt.savefig('static/images/shap_waterfall.png')
    plt.close()
    
    return {
        'summary_plot': 'static/images/shap_summary.png',
        'waterfall_plot': 'static/images/shap_waterfall.png'
    }

def get_feature_importance():
    """Get feature importance based on SHAP values"""
    model, scaler = load_model_and_scaler()
    feature_names = get_feature_names()
    
    # Load and preprocess a sample of test data
    _, test_data = load_data()
    test_sample = test_data.sample(100, random_state=42)
    
    X_test, _ = preprocess_data(test_sample, scaler=scaler, is_training=False)
    
    # Use SHAP's DeepExplainer
    explainer = shap.DeepExplainer(model, X_test[:50])
    shap_values = explainer.shap_values(X_test)
    
    # Calculate average absolute SHAP value for each feature
    feature_importance = np.mean(np.abs(shap_values[0]), axis=0)
    
    # Create a DataFrame with feature names and importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    return importance_df

if __name__ == "__main__":
    # Make sure the images directory exists
    import os
    if not os.path.exists('static/images'):
        os.makedirs('static/images')
    
    # Generate explanations
    plot_paths = explain_model_predictions()
    print(f"Explanation plots saved to: {plot_paths}")
    
    # Get and display feature importance
    importance = get_feature_importance()
    print("\nFeature Importance:")
    print(importance.head(10)) 