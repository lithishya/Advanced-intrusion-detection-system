import os
import sys
import subprocess

def check_data_exists():
    """
    Check if the NSL-KDD dataset files exist.
    """
    train_path = 'data/KDDTrain+.txt'
    test_path = 'data/KDDTest+.txt'
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        return True
    else:
        print("NSL-KDD dataset files not found.")
        print("Please download the dataset from: https://www.unb.ca/cic/datasets/nsl.html")
        print(f"Extract and place KDDTrain+.txt and KDDTest+.txt in the 'data' directory.")
        return False

def run_step(script_name, step_desc):
    """
    Run a Python script and handle errors.
    """
    print(f"\n{'='*80}")
    print(f"STEP: {step_desc}")
    print(f"{'='*80}")
    
    try:
        subprocess.run([sys.executable, script_name], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {str(e)}")
        return False

def main():
    """
    Run the full intrusion detection pipeline.
    """
    print("Network Intrusion Detection System")
    print("Using Deep Learning")
    print("\nStarting pipeline...\n")
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Check if dataset exists
    if not check_data_exists():
        return
    
    # Run preprocessing
    if not run_step('preprocess.py', 'Data Preprocessing'):
        return
    
    # Run model training
    if not run_step('train.py', 'Model Training'):
        return
    
    # Run model evaluation
    if not run_step('evaluate.py', 'Model Evaluation'):
        return
    
    print("\n\nPipeline completed successfully!")
    print("\nTo start the API server, run:")
    print("python api.py")

if __name__ == "__main__":
    main() 