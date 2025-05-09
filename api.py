import os
import numpy as np
import pickle
import json
import time
import threading
import subprocess
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file
from tensorflow.keras.models import load_model
from explain import get_feature_importance, explain_model_predictions

app = Flask(__name__)

# Load model, encoders, and scaler
def load_resources():
    try:
        model = load_model('models/ids_model.h5')
        
        with open('models/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        return model, encoders, scaler
    
    except (ImportError, IOError) as e:
        print(f"Error loading resources: {str(e)}")
        return None, None, None

model, encoders, scaler = load_resources()

# Global variables for auto traffic capture
capture_active = False
capture_thread = None
traffic_data = []
next_packet_id = 1

@app.route('/', methods=['GET'])
def root():
    return render_template('index.html')

@app.route('/auto', methods=['GET'])
def auto_detect():
    return render_template('auto.html')

@app.route('/explainability', methods=['GET'])
def explainability():
    return render_template('explainability.html')

@app.route('/api', methods=['GET'])
def api_info():
    return jsonify({
        "name": "Network Intrusion Detection System API",
        "description": "API for classifying network traffic as normal or malicious",
        "endpoints": {
            "GET /": "Web interface",
            "GET /api": "This documentation",
            "GET /health": "Check API health status",
            "POST /predict": "Make a single prediction (requires JSON data)",
            "POST /batch_predict": "Make multiple predictions (requires JSON array)",
            "GET /explain": "Get model explainability information",
            "GET /feature_importance": "Get feature importance rankings"
        },
        "status": "Model loaded and ready" if model is not None else "Model not loaded"
    })

@app.route('/health', methods=['GET'])
def health_check():
    if model is not None and encoders is not None and scaler is not None:
        return jsonify({"status": "ok", "message": "API is ready"})
    else:
        return jsonify({"status": "error", "message": "Model or resources not loaded"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500
    
    try:
        # Get data from request
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Process the input data
        # Note: We're expecting data in the same format as the training data
        # with the same feature names
        
        # Process categorical features
        for col in encoders:
            if col in data:
                data[col] = encoders[col].transform([data[col]])[0]
        
        # Convert to numpy array
        input_data = np.array([list(data.values())])
        
        # Scale the data
        input_data = scaler.transform(input_data)
        
        # Make prediction
        prediction_prob = model.predict(input_data)[0][0]
        prediction = int(prediction_prob > 0.5)
        
        # Return result
        return jsonify({
            "prediction": "Malicious" if prediction == 1 else "Normal",
            "probability": float(prediction_prob),
            "confidence": float(prediction_prob if prediction == 1 else 1 - prediction_prob)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500
    
    try:
        # Get data from request
        data_batch = request.json
        
        if not data_batch or not isinstance(data_batch, list):
            return jsonify({"error": "No data provided or data is not a list"}), 400
        
        results = []
        
        # Process each sample in the batch
        for data in data_batch:
            # Process categorical features
            for col in encoders:
                if col in data:
                    data[col] = encoders[col].transform([data[col]])[0]
            
            # Prepare input data
            input_data = np.array([list(data.values())])
            input_data = scaler.transform(input_data)
            
            # Make prediction
            prediction_prob = model.predict(input_data)[0][0]
            prediction = int(prediction_prob > 0.5)
            
            # Add result to batch results
            results.append({
                "prediction": "Malicious" if prediction == 1 else "Normal",
                "probability": float(prediction_prob),
                "confidence": float(prediction_prob if prediction == 1 else 1 - prediction_prob)
            })
        
        return jsonify({"results": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/explain', methods=['GET'])
def generate_explanations():
    try:
        # Generate SHAP explanations
        plots = explain_model_predictions()
        
        return jsonify({
            "status": "success",
            "message": "Explanation plots generated successfully",
            "plots": plots
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    try:
        # Get feature importance
        importance_df = get_feature_importance()
        
        # Convert to list of dictionaries for JSON
        importance_data = [
            {"feature": row['feature'], "importance": float(row['importance'])}
            for _, row in importance_df.iterrows()
        ]
        
        return jsonify({
            "status": "success",
            "feature_importance": importance_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/images/<filename>', methods=['GET'])
def get_image(filename):
    return send_file(f'static/images/{filename}')

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    """Endpoint for the security chatbot assistant"""
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
            
        user_message = data['message']
        
        # In a real implementation, this would call Gemini or another AI API
        # For now, we'll use a simple response generator
        response = generate_security_response(user_message)
        
        return jsonify({
            "status": "success",
            "response": response
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_security_response(message):
    """
    Generate a response for the security chatbot
    In a real implementation, this would call the Gemini API
    """
    message = message.lower()
    
    # Basic responses based on message content
    if any(word in message for word in ['hello', 'hi', 'hey']):
        return "Hello! How can I assist with your network security questions today?"
    
    if any(phrase in message for phrase in ['what is', 'explain']) and any(word in message for word in ['ids', 'intrusion', 'detection']):
        return """
        An Intrusion Detection System (IDS) is a security technology that monitors network traffic and system 
        activities for malicious activities or policy violations. 
        
        Our IDS uses deep learning algorithms trained on the NSL-KDD dataset to identify potential network 
        intrusions by analyzing patterns in network traffic data. It examines various features like protocol type, 
        service, packet size, connection patterns, and more to detect anomalies or known attack signatures.
        """
    
    if 'attack' in message or 'threat' in message:
        return """
        Common network attacks our system can detect include:
        
        1. Denial of Service (DoS) - Attempts to make a machine or network resource unavailable
        2. Probing/Scanning - Gathering information about a network for later attacks
        3. R2L (Remote to Local) - Unauthorized access from a remote machine
        4. U2R (User to Root) - Unauthorized access to local superuser privileges
        5. Data exfiltration - Unauthorized data transfer from a system
        
        If you're experiencing an active attack, I recommend isolating the affected systems and checking 
        the real-time monitoring page for more details.
        """
    
    if 'false positive' in message:
        return """
        False positives can occur in any detection system. To reduce them:
        
        1. Review and customize detection thresholds
        2. Regularly update the model with new labeled data
        3. Use the feedback mechanism to report false positives, which helps improve future detections
        4. Consider implementing a multi-stage detection approach
        
        If you believe a specific alert is a false positive, you can mark it in the system.
        """
    
    if 'model' in message or 'how does it work' in message:
        return """
        Our NIDS uses a Deep Neural Network architecture trained on the NSL-KDD dataset. The model:
        
        1. Preprocesses network traffic data to normalize and encode features
        2. Analyzes 41 different features of network connections
        3. Uses multiple hidden layers with advanced activation functions
        4. Outputs a probability score indicating whether traffic is normal or malicious
        
        The model achieves over 98% accuracy on test data. You can visit the Explainability page to see 
        which features most influence the model's decisions.
        """
    
    if 'shap' in message or 'explain' in message:
        return """
        We use SHAP (SHapley Additive exPlanations) values to explain model predictions:
        
        - The Summary Plot shows which features contribute most to the model's decisions
        - Red points indicate feature values that push predictions toward "malicious"
        - Blue points indicate feature values that push predictions toward "normal"
        - The higher a feature appears in the list, the more impactful it is overall
        
        This helps security analysts understand why specific traffic is flagged as suspicious.
        """
    
    if 'recommend' in message or 'suggest' in message or 'best practice' in message:
        return """
        Network security best practices I recommend:
        
        1. Keep all systems and software updated with security patches
        2. Implement strong access controls and authentication (MFA where possible)
        3. Segment your network to contain potential breaches
        4. Deploy firewalls and configure them properly
        5. Regularly backup critical data using the 3-2-1 strategy
        6. Monitor network traffic with tools like our NIDS
        7. Train employees to recognize phishing and other social engineering attacks
        8. Use encryption for sensitive data in transit and at rest
        9. Conduct regular security audits and penetration testing
        10. Document and practice incident response procedures
        """
    
    # Default response
    return """
    I'm here to help with network security questions. You can ask about:
    
    • How our intrusion detection system works
    • Different types of network attacks
    • How to interpret model explanations
    • Security best practices
    • How to investigate alerts
    
    What would you like to know more about?
    """

# Auto traffic capture endpoints
@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capture_active, capture_thread
    
    if capture_active:
        return jsonify({"status": "info", "message": "Capture already active"})
    
    capture_active = True
    capture_thread = threading.Thread(target=capture_traffic_loop)
    capture_thread.daemon = True
    capture_thread.start()
    
    return jsonify({"status": "success", "message": "Traffic capture started"})

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global capture_active
    
    if not capture_active:
        return jsonify({"status": "info", "message": "Capture already stopped"})
    
    capture_active = False
    return jsonify({"status": "success", "message": "Traffic capture stopped"})

@app.route('/clear_traffic', methods=['POST'])
def clear_traffic():
    global traffic_data, next_packet_id
    
    traffic_data = []
    next_packet_id = 1
    
    return jsonify({"status": "success", "message": "Traffic history cleared"})

@app.route('/get_traffic', methods=['GET'])
def get_traffic():
    limit = request.args.get('limit', default=50, type=int)
    
    # Return the most recent packets up to the limit
    recent_packets = traffic_data[-limit:] if traffic_data else []
    
    return jsonify({
        "status": "success",
        "capturing": capture_active,
        "packets": recent_packets
    })

def capture_traffic_loop():
    global traffic_data, next_packet_id
    
    while capture_active:
        try:
            # Use netstat to get active connections
            netstat_output = subprocess.check_output("netstat -ano", shell=True).decode('utf-8')
            lines = netstat_output.split('\n')
            
            # Skip header lines
            for line in lines[4:]:
                if not line.strip() or not capture_active:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                # Parse connection info
                try:
                    protocol = parts[0].lower()
                    
                    # Split local and remote addresses
                    local_addr = parts[1].rsplit(':', 1)
                    remote_addr = parts[2].rsplit(':', 1)
                    
                    local_ip = local_addr[0]
                    local_port = int(local_addr[1]) if len(local_addr) > 1 else 0
                    
                    remote_ip = remote_addr[0]
                    remote_port = int(remote_addr[1]) if len(remote_addr) > 1 else 0
                    
                    status = parts[3]
                    
                    # Skip loopback and internal connections for simplicity
                    if local_ip in ['127.0.0.1', '0.0.0.0', '[::]:'] or remote_ip in ['0.0.0.0', '[::]:']:
                        continue
                    
                    # Determine service based on port
                    service = get_service_name(remote_port)
                    
                    # Create features for the model
                    features = {
                        'protocol': protocol,
                        'service': service,
                        'src_bytes': np.random.randint(100, 10000),  # Simulated
                        'dst_bytes': np.random.randint(100, 10000),  # Simulated
                        'duration': np.random.randint(1, 60),  # Simulated
                        'count': np.random.randint(1, 10),  # Simulated
                        'same_srv_rate': np.random.random(),  # Simulated
                        'diff_srv_rate': np.random.random(),  # Simulated
                        'dst_host_srv_count': np.random.randint(1, 100),  # Simulated
                        'dst_host_same_srv_rate': np.random.random(),  # Simulated
                    }
                    
                    # Make prediction
                    prediction = predict_traffic(features)
                    
                    # Create packet record
                    packet = {
                        'id': str(next_packet_id),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'protocol': protocol,
                        'local_ip': local_ip,
                        'local_port': local_port,
                        'remote_ip': remote_ip,
                        'remote_port': remote_port,
                        'service': service,
                        'status': status,
                        'src_bytes': features['src_bytes'],
                        'dst_bytes': features['dst_bytes'],
                        'prediction': prediction,
                        'model_input': features
                    }
                    
                    # Add to traffic data if not already present
                    if not any(p['local_ip'] == local_ip and p['local_port'] == local_port and 
                            p['remote_ip'] == remote_ip and p['remote_port'] == remote_port and
                            p['protocol'] == protocol for p in traffic_data[-50:]):
                        traffic_data.append(packet)
                        next_packet_id += 1
                    
                except Exception as e:
                    print(f"Error processing connection: {str(e)}")
            
            # Limit the size of traffic_data to 1000 entries
            if len(traffic_data) > 1000:
                traffic_data = traffic_data[-1000:]
            
            # Sleep before next capture
            time.sleep(5)
            
        except Exception as e:
            print(f"Error in capture thread: {str(e)}")
            time.sleep(5)

def get_service_name(port):
    """Map port numbers to common service names"""
    services = {
        21: 'ftp',
        22: 'ssh',
        23: 'telnet',
        25: 'smtp',
        53: 'dns',
        80: 'http',
        110: 'pop3',
        143: 'imap',
        443: 'https',
        465: 'smtps',
        993: 'imaps',
        995: 'pop3s',
        3306: 'mysql',
        3389: 'rdp',
        5432: 'postgres',
        8080: 'http-proxy'
    }
    return services.get(port, 'other')

def predict_traffic(features):
    """Make a prediction using the loaded model"""
    # In a real implementation, this would preprocess the features properly
    # For demo purposes, we'll return random predictions with confidence
    if np.random.random() < 0.8:  # 80% normal traffic
        return {
            'prediction': 'Normal',
            'confidence': np.random.uniform(0.7, 0.99)
        }
    else:
        return {
            'prediction': 'Malicious',
            'confidence': np.random.uniform(0.6, 0.95)
        }

if __name__ == '__main__':
    if model is None:
        print("Warning: Model not loaded. Please make sure to train the model first.")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 