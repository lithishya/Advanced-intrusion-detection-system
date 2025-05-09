import json
import requests

def test_health():
    """
    Test the health endpoint of the API.
    """
    url = 'http://localhost:5000/health'
    response = requests.get(url)
    
    print("\nHealth Check:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def test_prediction():
    """
    Test the prediction endpoint with sample data.
    """
    url = 'http://localhost:5000/predict'
    
    # Load sample data
    with open('sample_request.json', 'r') as f:
        data = json.load(f)
    
    # Send POST request
    response = requests.post(url, json=data)
    
    print("\nSingle Prediction:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def test_batch_prediction():
    """
    Test the batch prediction endpoint with multiple samples.
    """
    url = 'http://localhost:5000/batch_predict'
    
    # Load sample data and create a batch
    with open('sample_request.json', 'r') as f:
        data = json.load(f)
    
    # Create a batch with the same sample repeated
    batch_data = [data, data]
    
    # Send POST request
    response = requests.post(url, json=batch_data)
    
    print("\nBatch Prediction:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    print("Testing Intrusion Detection System API")
    print("======================================")
    print("Make sure the API server is running (python api.py)")
    
    try:
        # Test health check
        test_health()
        
        # Test single prediction
        test_prediction()
        
        # Test batch prediction
        test_batch_prediction()
        
        print("\nAPI tests completed successfully!")
    
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the API server.")
        print("Make sure the server is running with 'python api.py'")
    except Exception as e:
        print(f"\nError occurred: {str(e)}") 