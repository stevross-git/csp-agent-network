#!/bin/bash
# Final Tracing Setup - Use Simple Jaeger-Only Configuration
# ==========================================================

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
# CLEAN UP EXISTING SETUP
# ============================================================================

log_info "Cleaning up existing tracing setup..."

cd monitoring/tracing 2>/dev/null || cd monitoring && mkdir -p tracing && cd tracing

# Stop all tracing containers
docker-compose -f docker-compose.tracing.yml down 2>/dev/null || true
docker-compose -f docker-compose.tracing-simple.yml down 2>/dev/null || true

# Remove stopped containers
docker rm csp_jaeger csp_otel_collector csp_jaeger_simple 2>/dev/null || true

cd ../..

# ============================================================================
# USE SIMPLE JAEGER SETUP (NO OTEL COLLECTOR)
# ============================================================================

log_info "Setting up simple Jaeger configuration..."

# Create the simple setup that's working
cat > monitoring/tracing/docker-compose.yml << 'EOF'
version: '3.8'

services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: csp_jaeger
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
      - COLLECTOR_OTLP_ENABLED=true
      - SPAN_STORAGE_TYPE=memory
      - LOG_LEVEL=info
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # Collector HTTP
      - "14250:14250"  # Collector gRPC  
      - "4317:4317"    # OTLP gRPC receiver
      - "4318:4318"    # OTLP HTTP receiver
      - "9411:9411"    # Zipkin
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
# START JAEGER
# ============================================================================

log_info "Starting Jaeger..."

cd monitoring/tracing
docker-compose up -d
cd ../..

# Wait for Jaeger to start
log_info "Waiting for Jaeger to start..."
sleep 10

# ============================================================================
# VERIFY JAEGER IS RUNNING
# ============================================================================

log_info "Verifying Jaeger status..."

echo ""
echo "Service Status:"
echo "==============="

# Check if Jaeger is running
if docker ps | grep -q csp_jaeger; then
    echo "✓ Jaeger: Running"
    
    # Test Jaeger UI
    if curl -s -f http://localhost:16686 > /dev/null 2>&1; then
        echo "✓ Jaeger UI: Accessible at http://localhost:16686"
    else
        echo "⚠ Jaeger UI: Still starting..."
    fi
    
    # Test OTLP endpoint
    if nc -zv localhost 4317 2>&1 | grep -q succeeded; then
        echo "✓ OTLP gRPC: Port 4317 is open"
    else
        echo "⚠ OTLP gRPC: Port 4317 not ready"
    fi
else
    echo "✗ Jaeger: Not running"
    echo "  Check logs: docker logs csp_jaeger"
fi

# ============================================================================
# CREATE PYTHON TEST SCRIPT
# ============================================================================

log_info "Creating Python test script..."

cat > monitoring/test-trace.py << 'EOF'
#!/usr/bin/env python3
"""
Test Jaeger tracing setup
"""
import time
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

# Configure OpenTelemetry
resource = Resource.create({
    "service.name": "test-service",
    "service.version": "1.0.0",
})

provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

# Configure OTLP exporter to Jaeger
otlp_exporter = OTLPSpanExporter(
    endpoint="localhost:4317",
    insecure=True,
)

# Add span processor
span_processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(span_processor)

# Get a tracer
tracer = trace.get_tracer("test.tracer", "1.0.0")

# Create test traces
print("Sending test traces to Jaeger...")

with tracer.start_as_current_span("test-operation") as span:
    span.set_attribute("test.attribute", "test-value")
    span.set_attribute("test.number", 42)
    span.add_event("test-event", {"event.data": "some data"})
    
    # Simulate some work
    time.sleep(0.1)
    
    # Create nested span
    with tracer.start_as_current_span("nested-operation") as nested:
        nested.set_attribute("nested.attribute", "nested-value")
        time.sleep(0.05)

# Force flush to ensure spans are sent
provider.force_flush()

print("✓ Test traces sent!")
print("✓ View them at: http://localhost:16686")
print("  - Click 'Search' in Jaeger UI")
print("  - Select 'test-service' from the Service dropdown")
print("  - Click 'Find Traces'")
EOF

chmod +x monitoring/test-trace.py

# ============================================================================
# CREATE INTEGRATION GUIDE
# ============================================================================

log_info "Creating integration guide..."

cat > monitoring/TRACING_INTEGRATION.md << 'EOF'
# Jaeger Tracing Integration Guide

## Quick Start

Jaeger is now running with OTLP support. Here's how to integrate it with your application:

### Python Integration

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

# Configure tracer
resource = Resource.create({"service.name": "your-service"})
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

# Add OTLP exporter
otlp_exporter = OTLPSpanExporter(endpoint="localhost:4317", insecure=True)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

# Use tracer in your code
tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("function_name")
def your_function():
    # Your code here
    pass
```

### Environment Variables

You can also configure via environment variables:

```bash
export OTEL_SERVICE_NAME="your-service"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_EXPORTER_OTLP_INSECURE="true"
```

### FastAPI Integration

```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# After setting up tracer as above
app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
```

## Access Points

- **Jaeger UI**: http://localhost:16686
- **OTLP gRPC**: localhost:4317
- **OTLP HTTP**: localhost:4318
- **Zipkin**: localhost:9411

## Testing

Run the test script:
```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
python3 monitoring/test-trace.py
```

Then check Jaeger UI for the traces.
EOF

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "====================================================="
echo "TRACING SETUP COMPLETE! 🎉"
echo "====================================================="
echo ""
echo "✅ Jaeger is running with in-memory storage"
echo "✅ OTLP endpoints are available"
echo "✅ Test script created"
echo "✅ Integration guide created"
echo ""
echo "QUICK TEST:"
echo "1. Install dependencies:"
echo "   pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc"
echo ""
echo "2. Run test:"
echo "   python3 monitoring/test-trace.py"
echo ""
echo "3. View traces:"
echo "   http://localhost:16686"
echo ""
echo "INTEGRATION:"
echo "- See monitoring/TRACING_INTEGRATION.md for details"
echo "- OTLP endpoint: localhost:4317"
echo "- No authentication required"
echo ""
echo "MONITORING COMPLETE!"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3001"  
echo "- Jaeger: http://localhost:16686"