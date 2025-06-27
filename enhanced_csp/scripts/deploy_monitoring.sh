#!/bin/bash
set -euo pipefail

echo "🚀 Deploying Enhanced CSP Monitoring Stack..."

# Create monitoring namespace
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

# Deploy Prometheus
echo "📊 Deploying Prometheus..."
kubectl apply -f monitoring/prometheus/ -n monitoring

# Deploy Alertmanager
echo "🚨 Deploying Alertmanager..."
kubectl apply -f monitoring/alertmanager/ -n monitoring

# Deploy Grafana
echo "📈 Deploying Grafana..."
kubectl apply -f monitoring/grafana/ -n monitoring

# Deploy Loki
echo "📝 Deploying Loki..."
kubectl apply -f monitoring/loki/ -n monitoring

# Deploy Promtail
echo "📊 Deploying Promtail..."
kubectl apply -f monitoring/promtail/ -n monitoring

# Wait for deployments
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n monitoring
kubectl wait --for=condition=available --timeout=300s deployment/grafana -n monitoring
kubectl wait --for=condition=available --timeout=300s deployment/loki -n monitoring

echo "✅ Monitoring stack deployed successfully!"
echo ""
echo "Access URLs:"
echo "📊 Prometheus: http://localhost:9090"
echo "📈 Grafana: http://localhost:3000 (admin/admin)"
echo "🚨 Alertmanager: http://localhost:9093"
echo "📝 Loki: http://localhost:3100"