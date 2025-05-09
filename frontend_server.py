import os
import json
import time
import socket
import threading
import subprocess
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_from_directory

app = Flask(__name__)

# Global variables to store captured traffic
captured_packets = []
is_capturing = False
capture_thread = None

# Simulated prediction responses (since we're not loading the real model in this script)
def simulate_prediction(packet_data):
    # This function simulates a prediction from our model
    # In reality, this would call the actual model prediction endpoint
    
    # Randomly determine if traffic is normal or malicious for demo purposes
    import random
    is_normal = random.random() > 0.3  # 70% chance of normal traffic
    confidence = random.uniform(0.65, 0.98)
    
    return {
        "prediction": "Normal" if is_normal else "Malicious",
        "probability": confidence if is_normal else 1 - confidence,
        "confidence": confidence if is_normal else 1 - confidence
    }

def format_packet_for_model(packet_info):
    """Convert packet info to the format expected by the model"""
    # Extract relevant features from packet_info
    protocol = packet_info.get('protocol', 'tcp').lower()
    if protocol not in ['tcp', 'udp', 'icmp']:
        protocol = 'tcp'  # default
    
    service = packet_info.get('service', 'http').lower()
    src_bytes = packet_info.get('src_bytes', 215)
    dst_bytes = packet_info.get('dst_bytes', 45076)
    
    # Create data object in the format expected by the model
    return {
        "duration": 0,
        "protocol_type": protocol,
        "service": service,
        "flag": "SF",
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 1,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": 1,
        "srv_count": 1,
        "serror_rate": 0,
        "srv_serror_rate": 0,
        "rerror_rate": 0,
        "srv_rerror_rate": 0,
        "same_srv_rate": 1,
        "diff_srv_rate": 0,
        "srv_diff_host_rate": 0,
        "dst_host_count": 9,
        "dst_host_srv_count": 9,
        "dst_host_same_srv_rate": 1,
        "dst_host_diff_srv_rate": 0,
        "dst_host_same_src_port_rate": 0.11,
        "dst_host_srv_diff_host_rate": 0,
        "dst_host_serror_rate": 0,
        "dst_host_srv_serror_rate": 0,
        "dst_host_rerror_rate": 0,
        "dst_host_srv_rerror_rate": 0
    }

def capture_traffic():
    """Capture network traffic using a packet sniffer or system commands"""
    global captured_packets, is_capturing
    
    while is_capturing:
        try:
            # Simulate capturing a packet (in a real system, you'd use a packet capture library like scapy)
            # For Windows, you could use netstat to get real traffic
            netstat_output = subprocess.check_output(
                ["netstat", "-n"], 
                shell=True,
                universal_newlines=True
            )
            
            # Process each line of netstat output
            lines = netstat_output.strip().split('\n')
            
            # Skip header lines
            for line in lines[4:]:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        # Extract relevant information
                        proto = parts[0].lower()  # TCP or UDP
                        local_address = parts[1]
                        remote_address = parts[2]
                        status = parts[3] if len(parts) > 3 else 'UNKNOWN'
                        
                        # Extract IP and port
                        try:
                            local_ip, local_port = local_address.rsplit(':', 1)
                            remote_ip, remote_port = remote_address.rsplit(':', 1)
                        except ValueError:
                            # Handle IPv6 or other formats
                            local_ip, local_port = local_address, '0'
                            remote_ip, remote_port = remote_address, '0'
                        
                        # Create packet info
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        packet_info = {
                            'timestamp': timestamp,
                            'protocol': proto,
                            'local_ip': local_ip,
                            'local_port': local_port,
                            'remote_ip': remote_ip,
                            'remote_port': remote_port,
                            'status': status,
                            # Estimate bytes based on port (just for demo)
                            'src_bytes': hash(local_port) % 1000 + 100,
                            'dst_bytes': hash(remote_port) % 5000 + 1000,
                            'service': get_service_by_port(int(remote_port) if remote_port.isdigit() else 0)
                        }
                        
                        # Format for model and get prediction
                        model_input = format_packet_for_model(packet_info)
                        prediction = simulate_prediction(model_input)
                        
                        # Add to captured packets if not already present
                        packet_with_prediction = {
                            **packet_info,
                            'prediction': prediction,
                            'model_input': model_input
                        }
                        
                        # Only add if not duplicate (check combination of IPs, ports, and timestamp)
                        packet_id = f"{local_ip}:{local_port}-{remote_ip}:{remote_port}-{timestamp}"
                        if not any(p.get('id') == packet_id for p in captured_packets):
                            packet_with_prediction['id'] = packet_id
                            captured_packets.insert(0, packet_with_prediction)
                            
                            # Keep only the last 100 packets
                            if len(captured_packets) > 100:
                                captured_packets.pop()
                    except Exception as e:
                        print(f"Error processing packet: {e}")
                        continue
            
            # Sleep to avoid consuming too many resources
            time.sleep(3)
            
        except Exception as e:
            print(f"Error capturing traffic: {e}")
            time.sleep(5)  # Wait a bit longer if there was an error

def get_service_by_port(port):
    """Return service name based on port number"""
    common_ports = {
        80: 'http',
        443: 'https',
        21: 'ftp',
        22: 'ssh',
        23: 'telnet',
        25: 'smtp',
        53: 'dns',
        110: 'pop3',
        143: 'imap',
        3306: 'mysql',
        5432: 'postgresql',
        27017: 'mongodb'
    }
    return common_ports.get(port, 'other')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/auto')
def auto_mode():
    return render_template('auto.html')

@app.route('/auto.html')
def auto_html():
    return render_template('auto.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/health')
def health_check():
    return jsonify({"status": "ok", "message": "Frontend server is ready"})

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global is_capturing, capture_thread
    
    if not is_capturing:
        is_capturing = True
        capture_thread = threading.Thread(target=capture_traffic)
        capture_thread.daemon = True
        capture_thread.start()
        return jsonify({"status": "success", "message": "Traffic capture started"})
    else:
        return jsonify({"status": "info", "message": "Traffic capture already running"})

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global is_capturing
    
    if is_capturing:
        is_capturing = False
        return jsonify({"status": "success", "message": "Traffic capture stopped"})
    else:
        return jsonify({"status": "info", "message": "Traffic capture not running"})

@app.route('/get_traffic')
def get_traffic():
    limit = int(request.args.get('limit', 20))
    return jsonify({
        "status": "success",
        "capturing": is_capturing,
        "packets": captured_packets[:limit]
    })

@app.route('/clear_traffic', methods=['POST'])
def clear_traffic():
    global captured_packets
    captured_packets = []
    return jsonify({"status": "success", "message": "Traffic history cleared"})

@app.route('/predict', methods=['POST'])
def predict():
    # This route lets the frontend still work with the original JavaScript code
    # It will simulate a prediction or call the real API if available
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Try to call the real prediction API if running
        try:
            import requests
            response = requests.post('http://localhost:5000/predict', json=data, timeout=2)
            if response.status_code == 200:
                return jsonify(response.json())
        except:
            # If real API is not available, simulate a prediction
            pass
            
        # Fallback to simulation
        result = simulate_prediction(data)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Network Traffic Analyzer Frontend Server...")
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True) 