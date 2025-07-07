#!/bin/bash
# Deploy monitoring enhancements

set -euo pipefail

echo "Deploying monitoring enhancements..."

# Start tracing
echo "Starting distributed tracing..."
cd monitoring/tracing
docker-compose -f docker-compose.tracing.yml up -d
cd ../..

# Start anomaly detection
echo "Starting anomaly detection..."
cd monitoring/anomaly_detection
docker build -t csp-anomaly-detector -f Dockerfile.anomaly .
docker-compose -f docker-compose.anomaly.yml up -d
cd ../..

# Update Prometheus with new rules
echo "Updating Prometheus configuration..."
cp monitoring/security/rules/security_alerts.yml monitoring/prometheus/rules/
docker-compose -f monitoring/docker-compose.monitoring.yml restart prometheus

# Show status
echo ""
echo "Monitoring enhancements deployed!"
echo ""
echo "New endpoints available:"
echo "- Jaeger UI: http://localhost:16686"
echo "- Security Status: http://localhost:8000/api/security/status"
echo "- Anomalies: http://localhost:8000/api/anomalies/current"
echo ""
echo "Integration steps:"
echo "1. Update backend/main.py to import monitoring_integration"
echo "2. Call integrate_enhanced_monitoring(app) after creating FastAPI app"
echo "3. Restart the backend service"
