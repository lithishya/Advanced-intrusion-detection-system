# Network Intrusion Detection System (NIDS)

## Project Overview
This Network Intrusion Detection System is an advanced security solution that uses deep learning to detect and analyze potential network intrusions. The system provides real-time monitoring, alert generation, and detailed explanations of detected threats through an interactive chatbot interface.

## Features
- Real-time network traffic monitoring
- Deep learning-based intrusion detection
- Interactive security assistant chatbot
- Detailed threat explanations using SHAP values
- User-friendly web interface
- Alert management system
- Model explainability visualization

## Technical Stack

### Frontend
- HTML5
- CSS3
- JavaScript (ES6+)
- Font Awesome (for icons)
- Modern responsive design

### Backend
- Python
- Flask (Web Framework)
- Deep Learning Model (trained on NSL-KDD dataset)
- SHAP (SHapley Additive exPlanations) for model interpretability

### Key Components

#### 1. Security Chatbot
- Interactive AI-powered assistant
- Topic-based quick access buttons
- Real-time response generation
- Message history tracking
- Typing indicators
- Suggestion chips for common queries

#### 2. Intrusion Detection Model
- Deep Neural Network architecture
- Trained on NSL-KDD dataset
- Real-time traffic analysis
- Feature extraction and processing
- Anomaly detection capabilities

#### 3. Alert System
- Real-time alert generation
- Alert severity classification
- Detailed alert information
- False positive handling
- Alert history tracking

## Project Structure
```
project/
├── static/
│   ├── chatbot.js      # Chatbot implementation
│   ├── styles.css      # Main styles
│   └── ...            # Other static assets
├── templates/
│   ├── index.html     # Main dashboard
│   └── ...           # Other templates
├── api.py            # Flask backend
├── model/           # Deep learning model
└── README.md        # Project documentation
```

## Requirements

### System Requirements
- Python 3.8 or higher
- Modern web browser
- Network access for monitoring
- Sufficient RAM for model inference

### Python Dependencies
```
flask>=2.0.0
numpy>=1.19.0
pandas>=1.2.0
tensorflow>=2.4.0
shap>=0.39.0
scikit-learn>=0.24.0
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd nids-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python api.py
```

## How It Works

### 1. Network Traffic Monitoring
- The system continuously monitors network traffic
- Captures and processes network packets
- Extracts relevant features for analysis

### 2. Intrusion Detection
- Deep learning model analyzes traffic patterns
- Compares against known attack signatures
- Identifies anomalies and potential threats
- Generates alerts for suspicious activities

### 3. Alert Processing
- Alerts are categorized by severity
- Detailed information is extracted
- Notifications are sent to administrators
- Alerts are stored for historical analysis

### 4. Security Assistant
- Interactive chatbot interface
- Quick access to common security topics
- Real-time threat explanations
- User-friendly query handling

### 5. Model Explainability
- SHAP values explain model decisions
- Visual representations of feature importance
- Detailed breakdown of detection logic
- Helps in understanding false positives

## Usage

### Dashboard
- Monitor real-time network traffic
- View active alerts
- Access historical data
- Configure system settings

### Chatbot Interface
- Click the robot icon to open the chatbot
- Use topic buttons for quick access
- Type queries about security
- Get instant responses and explanations

### Alert Management
- Review and acknowledge alerts
- Mark false positives
- Access detailed alert information
- Track alert history

## Security Features

### 1. Attack Detection
- DDoS attacks
- Port scanning
- Brute force attempts
- Malware communication
- Suspicious traffic patterns

### 2. Alert Categories
- Critical
- High
- Medium
- Low
- Informational

### 3. Response Actions
- Automatic blocking
- Alert notifications
- Logging and tracking
- Incident response guidance

## Best Practices

### System Maintenance
- Regular model updates
- Database maintenance
- Log rotation
- Performance monitoring

### Security Recommendations
- Regular system updates
- Strong password policies
- Network segmentation
- Access control implementation
- Regular security audits

## Contributing
Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Support
For support, please open an issue in the repository or contact the development team.

## Acknowledgments
- NSL-KDD dataset providers
- Open-source community
- Contributing developers 