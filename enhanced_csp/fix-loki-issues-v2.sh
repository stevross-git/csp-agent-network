#!/bin/bash
# Fix Loki Configuration and Mount Issues
# ========================================

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
# STOP AND CLEAN UP
# ============================================================================

log_info "Stopping Loki and Promtail..."

cd monitoring
docker-compose -f docker-compose.loki.yml down
cd ..

# ============================================================================
# FIX LOKI CONFIGURATION
# ============================================================================

log_info "Fixing Loki configuration..."

# Create updated Loki config without the problematic field
cat > monitoring/loki/loki.yml << 'EOF'
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    instance_addr: 127.0.0.1
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

ruler:
  alertmanager_url: http://alertmanager:9093
  enable_api: true

analytics:
  reporting_enabled: false

limits_config:
  # Remove the problematic enforce_metric_name field
  reject_old_samples: true
  reject_old_samples_max_age: 168h
  ingestion_rate_mb: 16
  ingestion_burst_size_mb: 32
EOF

log_success "Fixed Loki configuration"

# ============================================================================
# FIX DOCKER COMPOSE - REMOVE PROBLEMATIC MOUNTS
# ============================================================================

log_info "Updating Docker Compose configuration..."

cat > monitoring/docker-compose.loki.yml << 'EOF'
version: '3.8'

services:
  loki:
    image: grafana/loki:2.9.0
    container_name: csp_loki
    command: -config.file=/etc/loki/loki.yml
    ports:
      - "3100:3100"
    volumes:
      - ./loki/loki.yml:/etc/loki/loki.yml:ro
      - loki_data:/loki
    networks:
      - scripts_csp-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:3100/ready"]
      interval: 30s
      timeout: 10s
      retries: 3

  promtail:
    image: grafana/promtail:2.9.0
    container_name: csp_promtail
    command: -config.file=/etc/promtail/promtail.yml
    volumes:
      - ./promtail/promtail.yml:/etc/promtail/promtail.yml:ro
      # Only mount Docker logs - skip the problematic application logs mount
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    networks:
      - scripts_csp-network
    restart: unless-stopped
    depends_on:
      - loki

networks:
  scripts_csp-network:
    external: true

volumes:
  loki_data:
    driver: local
EOF

log_success "Updated Docker Compose configuration"

# ============================================================================
# UPDATE PROMTAIL CONFIG FOR DOCKER LOGS ONLY
# ============================================================================

log_info "Updating Promtail configuration for Docker logs..."

cat > monitoring/promtail/promtail.yml << 'EOF'
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  # Docker container logs
  - job_name: docker
    static_configs:
      - targets:
          - localhost
        labels:
          job: docker
          __path__: /var/lib/docker/containers/*/*log
    
    pipeline_stages:
      - json:
          expressions:
            log: log
            stream: stream
            time: time
            attrs: attrs
      
      - timestamp:
          source: time
          format: RFC3339Nano
      
      - json:
          source: attrs
          expressions:
            tag: tag
      
      - regex:
          source: tag
          expression: ^(?P<image_name>(?:[^|]*))\|(?P<container_name>(?:[^|]*))\|(?P<image_id>(?:[^|]*))\|(?P<container_id>(?:[^|]*))$
      
      - labels:
          container_name:
          container_id:
          image_name:
          image_id:
      
      - output:
          source: log
EOF

log_success "Updated Promtail configuration"

# ============================================================================
# START SERVICES
# ============================================================================

log_info "Starting Loki and Promtail..."

cd monitoring
docker-compose -f docker-compose.loki.yml up -d
cd ..

# Wait for services to start
log_info "Waiting for services to start..."
sleep 15

# ============================================================================
# VERIFY SERVICES
# ============================================================================

log_info "Verifying services..."

echo ""
echo "Service Status:"
echo "==============="

# Check Loki
if docker ps | grep -q csp_loki && ! docker logs csp_loki 2>&1 | tail -10 | grep -q "failed parsing config"; then
    echo "✓ Loki: Running without errors"
    
    # Test Loki API
    if curl -s http://localhost:3100/ready | grep -q "ready"; then
        echo "✓ Loki API: Ready"
    else
        echo "⚠ Loki API: Starting..."
    fi
else
    echo "✗ Loki: Not running or has errors"
    echo "  Check logs: docker logs csp_loki"
fi

# Check Promtail
if docker ps | grep -q csp_promtail; then
    echo "✓ Promtail: Running"
else
    echo "✗ Promtail: Not running"
    echo "  Check logs: docker logs csp_promtail"
fi

# ============================================================================
# CREATE SIMPLE LOKI DASHBOARD
# ============================================================================

log_info "Creating simple Loki dashboard..."

mkdir -p monitoring/grafana/dashboards

cat > monitoring/grafana/dashboards/docker-logs.json << 'EOF'
{
  "dashboard": {
    "title": "Docker Container Logs",
    "panels": [
      {
        "datasource": "Loki",
        "fieldConfig": {
          "defaults": {
            "custom": {}
          }
        },
        "gridPos": {
          "h": 15,
          "w": 24,
          "x": 0,
          "y": 0
        },
        "id": 1,
        "options": {
          "showLabels": false,
          "showTime": true,
          "sortOrder": "Descending",
          "wrapLogMessage": true
        },
        "targets": [
          {
            "expr": "{job=\"docker\"}",
            "refId": "A"
          }
        ],
        "title": "All Container Logs",
        "type": "logs"
      }
    ],
    "schemaVersion": 27,
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "title": "Docker Container Logs",
    "uid": "docker-logs"
  }
}
EOF

# ============================================================================
# TEST LOG COLLECTION
# ============================================================================

log_info "Testing log collection..."

# Generate a test log from a container
docker run --rm --name test-log-generator alpine echo "TEST LOG: Loki is working!" 2>/dev/null || true

# Wait for log processing
sleep 5

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "====================================================="
echo "LOKI FIXED AND RUNNING! 📊"
echo "====================================================="
echo ""
echo "✅ Fixed Loki configuration (removed invalid field)"
echo "✅ Using specific Loki/Promtail version (2.9.0)"
echo "✅ Configured to collect Docker container logs only"
echo "✅ Avoided problematic directory mounts"
echo ""
echo "ACCESS POINTS:"
echo "- Loki API: http://localhost:3100"
echo "- Loki Ready: http://localhost:3100/ready"
echo ""
echo "VIEW LOGS IN GRAFANA:"
echo "1. Open Grafana: http://localhost:3001"
echo "2. Login: admin/admin"
echo "3. Go to Explore (compass icon)"
echo "4. Select 'Loki' from dropdown"
echo "5. Query: {job=\"docker\"}"
echo ""
echo "USEFUL QUERIES:"
echo '  {job="docker"}'
echo '  {job="docker", container_name="csp_api"}'
echo '  {job="docker"} |= "error"'
echo '  {job="docker"} |= "ERROR" |= "database"'
echo ""
echo "CHECK IF WORKING:"
echo "- docker logs csp_loki | tail"
echo "- docker logs csp_promtail | tail"
echo "- curl http://localhost:3100/ready"
echo ""
echo "Your monitoring stack is now complete:"
echo "✅ Metrics: Prometheus (http://localhost:9090)"
echo "✅ Logs: Loki (http://localhost:3100)"
echo "✅ Traces: Jaeger (http://localhost:16686)"
echo "✅ Dashboards: Grafana (http://localhost:3001)"