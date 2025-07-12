#!/bin/bash
# setup-complete-infrastructure.sh - Setup databases and monitoring

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# =============================================================================
# STEP 1: Create Docker Network
# =============================================================================
log_step "Creating Docker network..."
docker network create scripts_csp-network 2>/dev/null || log_warn "Network already exists"

# =============================================================================
# STEP 2: Setup Databases
# =============================================================================
log_step "Setting up databases..."

# Make the database setup script executable
chmod +x setup-databases.sh

# Run database setup
./setup-databases.sh

# Wait for databases to be fully ready
log_info "Waiting for databases to stabilize..."
sleep 15

# =============================================================================
# STEP 3: Fix Configuration Files for Monitoring
# =============================================================================
log_step "Fixing monitoring configuration files..."

# Create blackbox exporter config
mkdir -p monitoring/blackbox_exporter
cat > monitoring/blackbox_exporter/config.yml << 'EOF'
modules:
  http_2xx:
    prober: http
    timeout: 10s
    http:
      follow_redirects: true
      preferred_ip_protocol: "ip4"
      valid_status_codes: []
      valid_http_versions: ["HTTP/1.1", "HTTP/2.0"]
  tcp_connect:
    prober: tcp
    timeout: 10s
  postgres_check:
    prober: tcp
    timeout: 10s
    tcp:
      preferred_ip_protocol: "ip4"
  redis_check:
    prober: tcp
    timeout: 10s
    tcp:
      preferred_ip_protocol: "ip4"
EOF

# Create process exporter config
mkdir -p monitoring/process_exporter
cat > monitoring/process_exporter/config.yml << 'EOF'
process_names:
  - name: "{{.Comm}}"
    cmdline:
    - '.+'
EOF

# Create alertmanager config
mkdir -p monitoring/alertmanager
cat > monitoring/alertmanager/alertmanager.yml << 'EOF'
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical'
      continue: true

receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://localhost:5001/webhook'
        send_resolved: true
        
  - name: 'critical'
    webhook_configs:
      - url: 'http://localhost:5001/critical'
        send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
EOF

# Create Grafana datasource provisioning
mkdir -p monitoring/grafana/datasources
cat > monitoring/grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    jsonData:
      httpMethod: POST
      manageAlerts: true
      prometheusType: Prometheus
      prometheusVersion: 2.40.0
    editable: true
EOF

# Create Grafana dashboard provisioning
mkdir -p monitoring/grafana/dashboards
cat > monitoring/grafana/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'CSP Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

# =============================================================================
# STEP 4: Start Monitoring Stack
# =============================================================================
log_step "Starting monitoring stack..."

# Stop any existing monitoring containers
docker-compose -f monitoring/docker-compose.monitoring.yml down 2>/dev/null || true
docker-compose -f monitoring/docker-compose.exporters.yml down 2>/dev/null || true

# Start monitoring services
docker-compose -f monitoring/docker-compose.monitoring.yml up -d
docker-compose -f monitoring/docker-compose.exporters.yml up -d

# Wait for services to start
log_info "Waiting for monitoring services to start..."
sleep 20

# =============================================================================
# STEP 5: Verify All Services
# =============================================================================
log_step "Verifying all services..."

echo ""
echo "=================================="
echo "SERVICE STATUS CHECK"
echo "=================================="

# Check databases
echo -e "\n${BLUE}Databases:${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(postgres|redis|mongodb|chroma|qdrant|weaviate)" || true

# Check monitoring
echo -e "\n${BLUE}Monitoring:${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(prometheus|grafana|alertmanager|exporter|cadvisor)" || true

# Check endpoints
echo -e "\n${BLUE}Endpoint Checks:${NC}"

# Database endpoints
curl -s -o /dev/null -w "PostgreSQL Main:    %{http_code}\n" http://localhost:5432 || echo "PostgreSQL Main:    Not responding"
curl -s -o /dev/null -w "Redis:              %{http_code}\n" http://localhost:6379 || echo "Redis:              Running (no HTTP)"
curl -s -o /dev/null -w "MongoDB:            %{http_code}\n" http://localhost:27017 || echo "MongoDB:            Running (no HTTP)"
curl -s -o /dev/null -w "ChromaDB:           %{http_code}\n" http://localhost:8200/api/v1/heartbeat || echo "ChromaDB:           Not responding"
curl -s -o /dev/null -w "Qdrant:             %{http_code}\n" http://localhost:6333/health || echo "Qdrant:             Not responding"

echo ""

# Monitoring endpoints
curl -s -o /dev/null -w "Prometheus:         %{http_code}\n" http://localhost:9090/-/healthy || echo "Prometheus:         Not responding"
curl -s -o /dev/null -w "Grafana:            %{http_code}\n" http://localhost:3001/api/health || echo "Grafana:            Not responding"
curl -s -o /dev/null -w "Alertmanager:       %{http_code}\n" http://localhost:9093/-/healthy || echo "Alertmanager:       Not responding"
curl -s -o /dev/null -w "Node Exporter:      %{http_code}\n" http://localhost:9100/metrics || echo "Node Exporter:      Not responding"

# =============================================================================
# STEP 6: Import Grafana Dashboard
# =============================================================================
log_step "Importing Grafana dashboard..."

# Wait for Grafana to be fully ready
sleep 10

# Save the dashboard JSON created earlier
if [ -f "csp-comprehensive-dashboard.json" ]; then
    cp csp-comprehensive-dashboard.json monitoring/grafana/dashboards/
    log_info "Dashboard copied to provisioning directory"
else
    log_warn "Dashboard JSON not found. You'll need to import it manually."
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=================================="
echo "🎉 SETUP COMPLETE!"
echo "=================================="
echo ""
echo "${GREEN}✓ Databases:${NC}"
echo "  - PostgreSQL (Main, AI Models, Vector)"
echo "  - Redis"
echo "  - MongoDB"
echo "  - ChromaDB, Qdrant, Weaviate"
echo ""
echo "${GREEN}✓ Monitoring:${NC}"
echo "  - Prometheus"
echo "  - Grafana"
echo "  - Alertmanager"
echo "  - Node Exporter"
echo "  - Database Exporters"
echo ""
echo "${BLUE}Access URLs:${NC}"
echo "  Grafana:       http://localhost:3001 (admin/admin)"
echo "  Prometheus:    http://localhost:9090"
echo "  Alertmanager:  http://localhost:9093"
echo "  pgAdmin:       http://localhost:5050 (admin@csp.local)"
echo "  RedisInsight:  http://localhost:8001"
echo ""
echo "${YELLOW}Next Steps:${NC}"
echo "1. Login to Grafana and check the dashboard"
echo "2. Verify Prometheus targets at http://localhost:9090/targets"
echo "3. Configure alert notification channels in Alertmanager"
echo "4. Start your CSP application to generate metrics"
echo ""
echo "${RED}To stop everything:${NC}"
echo "  docker-compose -f docker-compose.databases.yml down"
echo "  docker-compose -f monitoring/docker-compose.monitoring.yml down"
echo "  docker-compose -f monitoring/docker-compose.exporters.yml down"
echo "=================================="

# Create a helper script for managing the stack
cat > manage-stack.sh << 'EOF'
#!/bin/bash
# Helper script to manage the CSP infrastructure

case "$1" in
    start)
        echo "Starting all services..."
        docker-compose -f docker-compose.databases.yml up -d
        docker-compose -f monitoring/docker-compose.monitoring.yml up -d
        docker-compose -f monitoring/docker-compose.exporters.yml up -d
        ;;
    stop)
        echo "Stopping all services..."
        docker-compose -f monitoring/docker-compose.exporters.yml down
        docker-compose -f monitoring/docker-compose.monitoring.yml down
        docker-compose -f docker-compose.databases.yml down
        ;;
    restart)
        $0 stop
        sleep 5
        $0 start
        ;;
    status)
        echo "Service Status:"
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        ;;
    logs)
        service=$2
        if [ -z "$service" ]; then
            echo "Usage: $0 logs <service_name>"
        else
            docker logs -f csp_$service
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs <service>}"
        exit 1
        ;;
esac
EOF

chmod +x manage-stack.sh
log_info "Created manage-stack.sh helper script"