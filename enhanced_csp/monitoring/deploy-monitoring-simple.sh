#!/bin/bash
# Simplified Monitoring Deployment
# ================================

set -euo pipefail

echo "Deploying monitoring enhancements (simplified)..."
echo ""

# 1. Ensure network exists
echo "→ Creating Docker network..."
docker network create scripts_csp-network 2>/dev/null || echo "  Network already exists"

# 2. Start Tracing (Jaeger + OTEL)
echo ""
echo "→ Starting Distributed Tracing..."
cd monitoring/tracing
docker-compose -f docker-compose.tracing.yml up -d
cd ../..
echo "  ✓ Jaeger UI will be available at http://localhost:16686"

# 3. Build and start Anomaly Detection
echo ""
echo "→ Starting Anomaly Detection..."
cd monitoring/anomaly_detection

# Ensure directories exist
mkdir -p models config

# Build image
echo "  Building anomaly detector image..."
docker build -t csp-anomaly-detector -f Dockerfile.anomaly . || {
    echo "  ✗ Failed to build anomaly detector"
    echo "  Continuing with deployment..."
}

# Start container only if build succeeded
if docker images | grep -q csp-anomaly-detector; then
    docker-compose -f docker-compose.anomaly.yml up -d
    echo "  ✓ Anomaly detector started"
else
    echo "  ⚠ Skipping anomaly detector (build failed)"
fi

cd ../..

# 4. Verify running services
echo ""
echo "→ Verifying services..."
echo ""

# Check Jaeger
if docker ps | grep -q csp_jaeger; then
    echo "  ✓ Jaeger is running"
else
    echo "  ✗ Jaeger is not running"
fi

# Check OTEL Collector
if docker ps | grep -q csp_otel_collector; then
    echo "  ✓ OTEL Collector is running"
else
    echo "  ✗ OTEL Collector is not running"
fi

# Check Anomaly Detector
if docker ps | grep -q csp_anomaly_detector; then
    echo "  ✓ Anomaly Detector is running"
else
    echo "  ⚠ Anomaly Detector is not running (optional)"
fi

# 5. Test endpoints
echo ""
echo "→ Testing endpoints..."
echo ""

# Test Jaeger
if curl -s -f http://localhost:16686 > /dev/null 2>&1; then
    echo "  ✓ Jaeger UI is accessible at http://localhost:16686"
else
    echo "  ⚠ Jaeger UI is not yet accessible (may still be starting)"
fi

# Test OTEL metrics
if curl -s -f http://localhost:8888/metrics > /dev/null 2>&1; then
    echo "  ✓ OTEL Collector metrics available at http://localhost:8888/metrics"
else
    echo "  ⚠ OTEL Collector metrics not yet available"
fi

echo ""
echo "========================================="
echo "Deployment Summary"
echo "========================================="
echo ""
echo "Essential services (Tracing) have been deployed."
echo ""
echo "Access points:"
echo "• Jaeger UI: http://localhost:16686"
echo "• OTEL Metrics: http://localhost:8888/metrics"
echo ""
echo "Next steps:"
echo "1. Wait ~30 seconds for services to fully start"
echo "2. Verify Jaeger UI at http://localhost:16686"
echo "3. Integrate with your backend using the monitoring_integration module"
echo ""
echo "To check logs:"
echo "• Jaeger: docker logs csp_jaeger"
echo "• OTEL: docker logs csp_otel_collector"
echo "• Anomaly: docker logs csp_anomaly_detector"
