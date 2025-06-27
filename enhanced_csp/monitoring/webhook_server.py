#!/usr/bin/env python3
"""
Simple webhook server to receive Alertmanager notifications
"""
import json
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    print(f"\n🚨 ALERT RECEIVED at {datetime.now()}")
    print("=" * 50)
    
    if 'alerts' in data:
        for alert in data['alerts']:
            status = alert.get('status', 'unknown')
            labels = alert.get('labels', {})
            annotations = alert.get('annotations', {})
            
            print(f"Status: {status}")
            print(f"Alert: {labels.get('alertname', 'Unknown')}")
            print(f"Severity: {labels.get('severity', 'Unknown')}")
            print(f"Summary: {annotations.get('summary', 'No summary')}")
            print(f"Description: {annotations.get('description', 'No description')}")
            print("-" * 30)
    
    print("=" * 50)
    return jsonify({"status": "received"})

@app.route('/webhook/critical', methods=['POST'])
def webhook_critical():
    data = request.get_json()
    print(f"\n🔥 CRITICAL ALERT at {datetime.now()}")
    print("=" * 50)
    
    if 'alerts' in data:
        for alert in data['alerts']:
            labels = alert.get('labels', {})
            annotations = alert.get('annotations', {})
            
            print(f"🚨 CRITICAL: {labels.get('alertname', 'Unknown')}")
            print(f"Summary: {annotations.get('summary', 'No summary')}")
            print(f"Description: {annotations.get('description', 'No description')}")
    
    print("=" * 50)
    return jsonify({"status": "critical_received"})

if __name__ == '__main__':
    print("🎯 Starting webhook server on http://localhost:5001")
    print("This will receive and display alerts from Alertmanager")
    app.run(host='0.0.0.0', port=5001, debug=True)
