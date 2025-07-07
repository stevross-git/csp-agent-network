#!/bin/bash
# Fix OTEL Collector Configuration
# =================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
# STOP THE FAILING OTEL COLLECTOR
# ============================================================================

log_info "Stopping OTEL collector..."
docker stop csp_otel_collector 2>/dev/null || true
docker rm csp_otel_collector 2>/dev/null || true

# ============================================================================
# FIX OTEL COLLECTOR CONFIGURATION
# ============================================================================

log_info "Fixing OTEL Collector configuration..."

# Create the correct configuration that uses OTLP exporter instead of jaeger
cat > monitoring/tracing/config/otel-collector.yaml << 'EOF'
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024
  
  memory_limiter:
    check_interval: 1s
    limit_mib: 512
    spike_limit_mib: 128

exporters:
  # Use OTLP exporter to send to Jaeger
  otlp:
    endpoint: jaeger:4317
    tls:
      insecure: true
  
  # Prometheus exporter for metrics
  prometheus:
    endpoint: "0.0.0.0:8888"
    namespace: traces
    const_labels:
      environment: production
  
  # Debug exporter (optional, for troubleshooting)
  debug:
    verbosity: detailed

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [otlp]  # Send traces to Jaeger via OTLP
    
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [prometheus]
  
  # Enable telemetry for the collector itself
  telemetry:
    logs:
      level: info
    metrics:
      level: detailed
      address: 0.0.0.0:8888
EOF

log_success "Fixed OTEL Collector configuration"

# ============================================================================
# ALTERNATIVE: SIMPLER SETUP WITHOUT OTEL COLLECTOR
# ============================================================================

log_info "Creating alternative simpler setup..."

cat > monitoring/tracing/docker-compose.tracing-simple.yml << 'EOF'
version: '3.8'

services:
  # Just Jaeger with OTLP support
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: csp_jaeger_simple
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
      - COLLECTOR_OTLP_ENABLED=true
      - SPAN_STORAGE_TYPE=memory
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # Collector HTTP
      - "14250:14250"  # Collector gRPC
      - "4317:4317"    # OTLP gRPC receiver (direct)
      - "4318:4318"    # OTLP HTTP receiver (direct)
    networks:
      - scripts_csp-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:14269/"]
      interval: 5s
      timeout: 5s
      retries: 3

networks:
  scripts_csp-network:
    external: true
EOF

# ============================================================================
# RESTART WITH FIXED CONFIGURATION
# ============================================================================

log_info "Restarting services with fixed configuration..."

cd monitoring/tracing

# Try the full setup first
log_info "Starting Jaeger and OTEL Collector..."
docker-compose -f docker-compose.tracing.yml up -d

# Wait a bit
sleep 5

# Check if OTEL collector started successfully
if docker ps | grep -q csp_otel_collector; then
    log_success "OTEL Collector started successfully"
else
    log_warning "OTEL Collector failed to start, trying simpler setup..."
    
    # Stop everything
    docker-compose -f docker-compose.tracing.yml down
    
    # Start the simpler setup
    docker-compose -f docker-compose.tracing-simple.yml up -d
    
    log_info "Started simpler setup (Jaeger only with direct OTLP support)"
fi

cd ../..

# ============================================================================
# VERIFY SERVICES
# ============================================================================

log_info "Waiting for services to fully start..."
sleep 10

echo ""
echo "Service Status:"
echo "==============="

# Check Jaeger
if docker ps | grep -q jaeger; then
    JAEGER_CONTAINER=$(docker ps --format "{{.Names}}" | grep jaeger | head -1)
    echo "✓ Jaeger: Running (Container: $JAEGER_CONTAINER)"
    
    # Test Jaeger UI
    if curl -s -f http://localhost:16686 > /dev/null 2>&1; then
        echo "✓ Jaeger UI: Accessible at http://localhost:16686"
    else
        echo "⚠ Jaeger UI: Starting up..."
    fi
else
    echo "✗ Jaeger: Not running"
fi

echo ""

# Check OTEL Collector if it exists
if docker ps | grep -q csp_otel_collector; then
    echo "✓ OTEL Collector: Running"
    
    # Test OTEL metrics endpoint
    if curl -s -f http://localhost:8888/metrics > /dev/null 2>&1; then
        echo "✓ OTEL Metrics: Accessible at http://localhost:8888/metrics"
    else
        echo "⚠ OTEL Metrics: Starting up..."
    fi
else
    echo "ℹ OTEL Collector: Not running (using direct OTLP to Jaeger)"
fi

# ============================================================================
# PROVIDE INTEGRATION EXAMPLE
# ============================================================================

log_info "Creating integration example..."

cat > monitoring/tracing/test-trace.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to send a trace to Jaeger
"""
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
import time

# Configure the tracer
resource = Resource.create({"service.name": "test-service"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

# Configure OTLP exporter
# If using OTEL Collector: localhost:4317
# If using Jaeger directly: localhost:4317
otlp_exporter = OTLPSpanExporter(
    endpoint="localhost:4317",
    insecure=True
)

# Add the exporter to the tracer
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Create a test trace
with tracer.start_as_current_span("test-operation") as span:
    span.set_attribute("test.type", "manual")
    span.set_attribute("test.value", 42)
    
    # Simulate some work
    time.sleep(0.1)
    
    # Create a child span
    with tracer.start_as_current_span("child-operation") as child:
        child.set_attribute("child.data", "test-data")
        time.sleep(0.05)

print("Test trace sent! Check Jaeger UI at http://localhost:16686")
EOF

chmod +x monitoring/tracing/test-trace.py

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "====================================================="
echo "OTEL COLLECTOR CONFIGURATION FIXED"
echo "====================================================="
echo ""
echo "✅ Fixed OTEL Collector config (using OTLP exporter)"
echo "✅ Created simpler alternative (Jaeger-only setup)"
echo "✅ Services restarted"
echo ""
echo "CURRENT SETUP:"

if docker ps | grep -q csp_otel_collector; then
    echo "- Using full setup: Jaeger + OTEL Collector"
    echo "- Send traces to: localhost:4317 (OTEL Collector)"
    echo "- Metrics available at: http://localhost:8888/metrics"
else
    echo "- Using simple setup: Jaeger with direct OTLP"
    echo "- Send traces to: localhost:4317 (Jaeger directly)"
fi

echo ""
echo "ACCESS POINTS:"
echo "- Jaeger UI: http://localhost:16686"
echo ""
echo "TEST THE SETUP:"
echo "1. Install test dependencies:"
echo "   pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
echo ""
echo "2. Run the test script:"
echo "   python3 monitoring/tracing/test-trace.py"
echo ""
echo "3. View traces in Jaeger:"
echo "   http://localhost:16686"
echo ""
echo "INTEGRATION WITH YOUR APP:"
echo "- OTLP endpoint: localhost:4317"
echo "- Protocol: gRPC"
echo "- No authentication required (insecure mode)"
echo ""
echo "CHECK LOGS IF NEEDED:"
echo "- docker logs $(docker ps --format '{{.Names}}' | grep jaeger | head -1)"
echo "- docker logs csp_otel_collector (if running)"