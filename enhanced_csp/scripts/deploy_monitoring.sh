#!/bin/bash
# Deploy complete monitoring stack

set -euo pipefail

echo "Deploying Enhanced CSP Monitoring..."

# 1. Update configuration
echo "Updating configuration..."
cp monitoring/prometheus/prometheus-final.yml monitoring/prometheus/prometheus.yml
cp monitoring/prometheus/rules/alerts-final.yml monitoring/prometheus/rules/alerts.yml

# 2. Deploy database exporters
echo "Deploying exporters..."
docker-compose -f monitoring/docker-compose.exporters.yml up -d

# 3. Restart monitoring stack
echo "Restarting monitoring services..."
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

# 4. Wait for services
echo "Waiting for services to start..."
sleep 30

# 5. Test endpoints
echo "Testing endpoints..."
python tests/test_monitoring_coverage.py

echo "Monitoring deployment complete!"
echo ""
echo "Access points:"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3000 (admin/admin)"
echo "- Alertmanager: http://localhost:9093"
echo "- API Metrics: http://localhost:8000/metrics"
echo ""
echo "Next steps:"
echo "1. Configure alert notification channels in Alertmanager"
echo "2. Set up Grafana notification channels"
echo "3. Customize dashboards for your use case"
echo "4. Set up long-term metrics storage if needed"
