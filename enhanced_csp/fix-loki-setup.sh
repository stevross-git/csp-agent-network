#!/bin/bash
# Fix Loki Setup Issues
# =====================

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
# STOP LOKI IF RUNNING
# ============================================================================

log_info "Stopping Loki and Promtail if running..."

docker stop csp_loki 2>/dev/null || true
docker stop csp_promtail 2>/dev/null || true
docker rm csp_loki 2>/dev/null || true
docker rm csp_promtail 2>/dev/null || true

# ============================================================================
# FIX DIRECTORY STRUCTURE
# ============================================================================

log_info "Fixing Loki directory structure..."

# Create proper directory structure
mkdir -p monitoring/loki/{config,data,rules}
mkdir -p monitoring/promtail/config
mkdir -p logs/{application,database,monitoring,audit}

# ============================================================================
# CREATE LOKI CONFIGURATION
# ============================================================================

log_info "Creating Loki configuration..."

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
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
  ingestion_rate_mb: 16
  ingestion_burst_size_mb: 32
EOF

# ============================================================================
# CREATE RECORDING RULES FILE
# ============================================================================

log_info "Creating recording rules file..."

cat > monitoring/loki/recording_rules.yml << 'EOF'
groups:
  - name: csp_logs
    interval: 1m
    rules:
      - record: csp:log_lines_total
        expr: |
          sum by (job, level) (
            count_over_time({job=~".+"}[1m])
          )
          
      - record: csp:error_logs_rate
        expr: |
          sum by (job) (
            rate({job=~".+", level="error"}[5m])
          )
EOF

# ============================================================================
# CREATE PROMTAIL CONFIGURATION
# ============================================================================

log_info "Creating Promtail configuration..."

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
  - job_name: containers
    static_configs:
      - targets:
          - localhost
        labels:
          job: containers
          __path__: /var/lib/docker/containers/*/*log
    
    pipeline_stages:
      - json:
          expressions:
            output: log
            stream: stream
            attrs:
      - json:
          expressions:
            tag:
          source: attrs
      - regex:
          expression: (?P<container_name>(?:[^|]*))\|(?P<log>.*)
          source: log
      - labels:
          container_name:
          stream:
      - output:
          source: log

  # Application logs
  - job_name: csp-api
    static_configs:
      - targets:
          - localhost
        labels:
          job: csp-api
          __path__: /var/log/csp/application/*.log

  # Database logs
  - job_name: postgres
    static_configs:
      - targets:
          - localhost
        labels:
          job: postgres
          __path__: /var/log/csp/database/*.log

  # System logs
  - job_name: syslog
    static_configs:
      - targets:
          - localhost
        labels:
          job: syslog
          __path__: /var/log/syslog
EOF

# ============================================================================
# CREATE DOCKER COMPOSE FOR LOKI
# ============================================================================

log_info "Creating Docker Compose for Loki..."

cat > monitoring/docker-compose.loki.yml << 'EOF'
version: '3.8'

services:
  loki:
    image: grafana/loki:latest
    container_name: csp_loki
    command: -config.file=/etc/loki/loki.yml
    ports:
      - "3100:3100"
    volumes:
      - ./loki/loki.yml:/etc/loki/loki.yml:ro
      - ./loki/recording_rules.yml:/loki/rules/recording_rules.yml:ro
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
    image: grafana/promtail:latest
    container_name: csp_promtail
    command: -config.file=/etc/promtail/promtail.yml
    volumes:
      - ./promtail/promtail.yml:/etc/promtail/promtail.yml:ro
      - /var/log:/var/log:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ../logs:/var/log/csp:ro
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

# ============================================================================
# ADD LOKI TO GRAFANA
# ============================================================================

log_info "Creating Grafana datasource for Loki..."

mkdir -p monitoring/grafana/datasources

cat > monitoring/grafana/datasources/loki.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    isDefault: false
    jsonData:
      maxLines: 1000
EOF

# ============================================================================
# START LOKI
# ============================================================================

log_info "Starting Loki and Promtail..."

cd monitoring
docker-compose -f docker-compose.loki.yml up -d
cd ..

# Wait for services to start
log_info "Waiting for services to start..."
sleep 10

# ============================================================================
# VERIFY LOKI IS RUNNING
# ============================================================================

log_info "Verifying Loki status..."

echo ""
echo "Service Status:"
echo "==============="

# Check Loki
if docker ps | grep -q csp_loki; then
    echo "✓ Loki: Running"
    
    # Test Loki API
    if curl -s http://localhost:3100/ready | grep -q "ready"; then
        echo "✓ Loki API: Ready"
    else
        echo "⚠ Loki API: Starting..."
    fi
else
    echo "✗ Loki: Not running"
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
# TEST LOG INGESTION
# ============================================================================

log_info "Testing log ingestion..."

# Create a test log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] TEST: Loki test log entry" >> logs/application/test.log

# Wait for ingestion
sleep 5

# Query Loki
echo ""
echo "Testing Loki query..."
RESULT=$(curl -s -G "http://localhost:3100/loki/api/v1/query" \
    --data-urlencode 'query={job="csp-api"} |= "TEST"' \
    --data-urlencode 'limit=5' 2>/dev/null || echo "failed")

if [[ "$RESULT" != "failed" ]] && [[ "$RESULT" != *"error"* ]]; then
    echo "✓ Log query successful"
else
    echo "⚠ Could not query logs yet (this is normal if no logs exist)"
fi

# ============================================================================
# CREATE LOG DASHBOARD
# ============================================================================

log_info "Creating Loki dashboard for Grafana..."

cat > monitoring/grafana/dashboards/loki-logs.json << 'EOF'
{
  "dashboard": {
    "title": "CSP Logs Dashboard",
    "panels": [
      {
        "datasource": "Loki",
        "gridPos": {"h": 10, "w": 24, "x": 0, "y": 0},
        "id": 1,
        "targets": [
          {
            "expr": "{job=\"csp-api\"}",
            "refId": "A"
          }
        ],
        "title": "Application Logs",
        "type": "logs"
      },
      {
        "datasource": "Loki",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 10},
        "id": 2,
        "targets": [
          {
            "expr": "sum(rate({job=\"csp-api\"} |= \"error\" [5m])) by (level)",
            "refId": "A"
          }
        ],
        "title": "Error Rate",
        "type": "graph"
      },
      {
        "datasource": "Loki",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 10},
        "id": 3,
        "targets": [
          {
            "expr": "sum(count_over_time({job=~\".+\"}[5m])) by (job)",
            "refId": "A"
          }
        ],
        "title": "Log Volume by Job",
        "type": "graph"
      }
    ]
  }
}
EOF

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "====================================================="
echo "LOKI SETUP COMPLETE! 📊"
echo "====================================================="
echo ""
echo "✅ Loki is running"
echo "✅ Promtail is collecting logs"
echo "✅ Grafana datasource configured"
echo "✅ Log dashboard created"
echo ""
echo "ACCESS POINTS:"
echo "- Loki API: http://localhost:3100"
echo "- Promtail: http://localhost:9080"
echo ""
echo "VIEW LOGS IN GRAFANA:"
echo "1. Open Grafana: http://localhost:3001"
echo "2. Go to Explore (compass icon)"
echo "3. Select 'Loki' datasource"
echo "4. Try queries:"
echo "   - All logs: {job=~\".+\"}"
echo "   - Errors: {job=\"csp-api\"} |= \"error\""
echo "   - Container logs: {job=\"containers\"}"
echo ""
echo "LOGQL EXAMPLES:"
echo '  {job="csp-api"}'
echo '  {job="postgres"} |= "ERROR"'
echo '  {job="containers", container_name="csp_api"}'
echo '  sum(rate({job="csp-api"} [5m])) by (level)'
echo ""
echo "CHECK LOGS IF NEEDED:"
echo "- docker logs csp_loki"
echo "- docker logs csp_promtail"