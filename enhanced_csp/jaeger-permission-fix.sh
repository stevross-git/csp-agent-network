#!/bin/bash
# Fix Jaeger Storage Permission Issues
# =====================================

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
# STOP FAILING CONTAINERS
# ============================================================================

log_info "Stopping failing containers..."

# Stop Jaeger if it's running/restarting
docker stop csp_jaeger 2>/dev/null || true
docker rm csp_jaeger 2>/dev/null || true

# Stop OTEL collector too
docker stop csp_otel_collector 2>/dev/null || true
docker rm csp_otel_collector 2>/dev/null || true

log_success "Stopped containers"

# ============================================================================
# FIX JAEGER DOCKER COMPOSE
# ============================================================================

log_info "Fixing Jaeger configuration..."

# Update the docker-compose file to use in-memory storage instead of badger
cat > monitoring/tracing/docker-compose.tracing.yml << 'EOF'
version: '3.8'

services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: csp_jaeger
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
      - COLLECTOR_OTLP_ENABLED=true
      # Use in-memory storage to avoid permission issues
      - SPAN_STORAGE_TYPE=memory
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # Collector HTTP
      - "14250:14250"  # Collector gRPC
      - "4317:4317"    # OTLP gRPC receiver
      - "4318:4318"    # OTLP HTTP receiver
    networks:
      - scripts_csp-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:14269/"]
      interval: 5s
      timeout: 5s
      retries: 3

  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    container_name: csp_otel_collector
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./config/otel-collector.yaml:/etc/otel-collector-config.yaml
    ports:
      - "4319:4317"    # OTLP gRPC
      - "4320:4318"    # OTLP HTTP
      - "8888:8888"    # Prometheus metrics
    networks:
      - scripts_csp-network
    depends_on:
      - jaeger
    restart: unless-stopped

networks:
  scripts_csp-network:
    external: true
EOF

log_success "Updated Jaeger configuration to use in-memory storage"

# ============================================================================
# ENSURE OTEL CONFIG EXISTS
# ============================================================================

log_info "Ensuring OTEL collector config exists..."

# Make sure the config directory exists
mkdir -p monitoring/tracing/config

# Check if otel-collector.yaml exists
if [ ! -f "monitoring/tracing/config/otel-collector.yaml" ]; then
    log_warning "OTEL collector config not found, creating it..."
    
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
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true
  
  prometheus:
    endpoint: "0.0.0.0:8888"
    namespace: traces
    const_labels:
      environment: production

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [jaeger]
    
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [prometheus]
EOF
    log_success "Created OTEL collector configuration"
fi

# ============================================================================
# RESTART SERVICES
# ============================================================================

log_info "Starting services with fixed configuration..."

cd monitoring/tracing
docker-compose -f docker-compose.tracing.yml up -d
cd ../..

# Wait for services to start
log_info "Waiting for services to start..."
sleep 10

# ============================================================================
# VERIFY SERVICES
# ============================================================================

log_info "Verifying services..."

echo ""
echo "Service Status:"
echo "==============="

# Check if Jaeger is running
if docker ps | grep -q csp_jaeger; then
    echo "✓ Jaeger: Running"
else
    echo "✗ Jaeger: Not running"
    echo "  Check logs: docker logs csp_jaeger"
fi

# Check if OTEL collector is running
if docker ps | grep -q csp_otel_collector; then
    echo "✓ OTEL Collector: Running"
else
    echo "✗ OTEL Collector: Not running"
    echo "  Check logs: docker logs csp_otel_collector"
fi

echo ""

# Test endpoints
echo "Endpoint Tests:"
echo "==============="

# Test Jaeger UI
if curl -s -f http://localhost:16686 > /dev/null 2>&1; then
    echo "✓ Jaeger UI: http://localhost:16686"
else
    echo "✗ Jaeger UI: Not accessible yet (may still be starting)"
fi

# Test OTEL metrics
if curl -s -f http://localhost:8888/metrics > /dev/null 2>&1; then
    echo "✓ OTEL Metrics: http://localhost:8888/metrics"
else
    echo "✗ OTEL Metrics: Not accessible yet"
fi

# ============================================================================
# ALTERNATIVE: JAEGER WITH ELASTICSEARCH (OPTIONAL)
# ============================================================================

log_info "Creating alternative setup with Elasticsearch (for production)..."

cat > monitoring/tracing/docker-compose.tracing-prod.yml << 'EOF'
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.17.10
    container_name: csp_elasticsearch
    environment:
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    networks:
      - scripts_csp-network
    restart: unless-stopped

  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: csp_jaeger_prod
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
      - COLLECTOR_OTLP_ENABLED=true
      - SPAN_STORAGE_TYPE=elasticsearch
      - ES_SERVER_URLS=http://elasticsearch:9200
    ports:
      - "16686:16686"
      - "14268:14268"
      - "14250:14250"
      - "4317:4317"
      - "4318:4318"
    networks:
      - scripts_csp-network
    depends_on:
      - elasticsearch
    restart: unless-stopped

  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    container_name: csp_otel_collector_prod
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./config/otel-collector.yaml:/etc/otel-collector-config.yaml
    ports:
      - "4319:4317"
      - "4320:4318"
      - "8888:8888"
    networks:
      - scripts_csp-network
    depends_on:
      - jaeger
    restart: unless-stopped

networks:
  scripts_csp-network:
    external: true

volumes:
  elasticsearch_data:
    driver: local
EOF

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "====================================================="
echo "JAEGER PERMISSION ISSUES FIXED"
echo "====================================================="
echo ""
echo "✅ Stopped failing containers"
echo "✅ Updated Jaeger to use in-memory storage"
echo "✅ Created/verified OTEL collector config"
echo "✅ Restarted services"
echo ""
echo "IMPORTANT NOTES:"
echo ""
echo "1. Jaeger is now using IN-MEMORY storage"
echo "   - Traces will be lost on restart"
echo "   - Good for development/testing"
echo ""
echo "2. For production with persistent storage:"
echo "   cd monitoring/tracing"
echo "   docker-compose -f docker-compose.tracing-prod.yml up -d"
echo "   (This uses Elasticsearch for storage)"
echo ""
echo "3. Access points:"
echo "   - Jaeger UI: http://localhost:16686"
echo "   - OTEL Metrics: http://localhost:8888/metrics"
echo ""
echo "4. To verify everything is working:"
echo "   curl http://localhost:16686"
echo "   curl http://localhost:8888/metrics"
echo ""
echo "5. Check logs if needed:"
echo "   docker logs csp_jaeger"
echo "   docker logs csp_otel_collector"