#!/bin/bash
set -euo pipefail

# Docker-based Enhanced CSP Monitoring Setup
# ===========================================

echo "🚀 Setting up Enhanced CSP Monitoring Stack with Docker..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are available
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi

    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi

    log_success "Docker and Docker Compose are available"
}

# Create monitoring directory structure
create_monitoring_structure() {
    log_info "Creating monitoring directory structure..."
    
    mkdir -p monitoring/{prometheus,grafana,alertmanager,loki,promtail}
    mkdir -p monitoring/grafana/{dashboards,datasources,plugins}
    mkdir -p monitoring/prometheus/{rules,data}
    mkdir -p monitoring/alertmanager/{data,config}
    mkdir -p monitoring/loki/{data,config}
    mkdir -p monitoring/promtail/config
    mkdir -p logs/{application,database,monitoring,audit}
    
    log_success "Monitoring directory structure created"
}

# Create Prometheus configuration
create_prometheus_config() {
    log_info "Creating Prometheus configuration..."
    
    cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'csp-monitor'
    environment: 'docker'

rule_files:
  - "/etc/prometheus/rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # CSP Services (from your running containers)
  - job_name: 'csp-chroma'
    static_configs:
      - targets: ['csp_chroma:8200']
    metrics_path: '/api/v1/heartbeat'
    scrape_interval: 30s

  - job_name: 'csp-qdrant'
    static_configs:
      - targets: ['csp_qdrant:6333']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'csp-weaviate'
    static_configs:
      - targets: ['csp_weaviate:8080']
    metrics_path: '/v1/meta'
    scrape_interval: 30s

  - job_name: 'csp-redis'
    static_configs:
      - targets: ['csp_redis:6379']
    scrape_interval: 30s

  - job_name: 'csp-postgres'
    static_configs:
      - targets: ['csp_postgres:5432']
    scrape_interval: 30s

  - job_name: 'csp-postgres-vector'
    static_configs:
      - targets: ['csp_postgres_vector:5434']
    scrape_interval: 30s

  - job_name: 'csp-ai-models-db'
    static_configs:
      - targets: ['csp_ai_models_db:5433']
    scrape_interval: 30s

  # Docker container metrics
  - job_name: 'docker-containers'
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: [__meta_docker_container_name]
        target_label: container_name
      - source_labels: [__meta_docker_container_id]
        target_label: container_id
EOF

    log_success "Prometheus configuration created"
}

# Create alert rules
create_alert_rules() {
    log_info "Creating Prometheus alert rules..."
    
    cat > monitoring/prometheus/rules/alert_rules.yml << 'EOF'
groups:
  - name: docker_container_alerts
    rules:
      - alert: ContainerDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Container {{ $labels.job }} is down"
          description: "Container {{ $labels.job }} has been down for more than 1 minute"

      - alert: HighCPUUsage
        expr: rate(container_cpu_usage_seconds_total[5m]) * 100 > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage on {{ $labels.container_name }}"
          description: "CPU usage is above 80% for 5 minutes"

      - alert: HighMemoryUsage
        expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) * 100 > 90
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage on {{ $labels.container_name }}"
          description: "Memory usage is above 90% for 5 minutes"

  - name: csp_service_alerts
    rules:
      - alert: ChromaServiceDown
        expr: up{job="csp-chroma"} == 0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Chroma vector database is down"
          description: "Chroma service has been unreachable for 2 minutes"

      - alert: QdrantServiceDown
        expr: up{job="csp-qdrant"} == 0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Qdrant vector database is down"
          description: "Qdrant service has been unreachable for 2 minutes"

      - alert: WeaviateServiceDown
        expr: up{job="csp-weaviate"} == 0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Weaviate vector database is down"
          description: "Weaviate service has been unreachable for 2 minutes"

      - alert: RedisServiceDown
        expr: up{job="csp-redis"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis cache is down"
          description: "Redis service has been unreachable for 1 minute"

      - alert: PostgresServiceDown
        expr: up{job=~"csp-postgres.*"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL database is down"
          description: "PostgreSQL service {{ $labels.job }} has been unreachable for 1 minute"
EOF

    log_success "Alert rules created"
}

# Create Alertmanager configuration
create_alertmanager_config() {
    log_info "Creating Alertmanager configuration..."
    
    cat > monitoring/alertmanager/alertmanager.yml << 'EOF'
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@csp-system.local'

route:
  group_by: ['alertname', 'container_name']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
    repeat_interval: 5m

receivers:
- name: 'default'
  webhook_configs:
  - url: 'http://localhost:5001/webhook'
    send_resolved: true

- name: 'critical-alerts'
  webhook_configs:
  - url: 'http://localhost:5001/webhook/critical'
    send_resolved: true
  # Add email or Slack configuration here
  # email_configs:
  # - to: 'admin@yourdomain.com'
  #   subject: 'CRITICAL: CSP System Alert'

inhibit_rules:
- source_match:
    severity: 'critical'
  target_match:
    severity: 'warning'
  equal: ['alertname', 'container_name']
EOF

    log_success "Alertmanager configuration created"
}

# Create Grafana datasource
create_grafana_datasource() {
    log_info "Creating Grafana datasource configuration..."
    
    cat > monitoring/grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    log_success "Grafana datasource configuration created"
}

# Create Docker Compose monitoring services
create_monitoring_compose() {
    log_info "Creating monitoring Docker Compose file..."
    
    cat > monitoring/docker-compose.monitoring.yml << 'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: csp_prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/prometheus/rules:/etc/prometheus/rules
      - prometheus_data:/prometheus
      - /var/run/docker.sock:/var/run/docker.sock:ro
    networks:
      - scripts_csp-network
    restart: unless-stopped
    depends_on:
      - alertmanager

  alertmanager:
    image: prom/alertmanager:latest
    container_name: csp_alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    networks:
      - scripts_csp-network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: csp_grafana_monitoring
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
    ports:
      - "3001:3000"  # Using port 3001 to avoid conflict with existing Grafana
    volumes:
      - grafana_monitoring_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - scripts_csp-network
    restart: unless-stopped
    depends_on:
      - prometheus

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: csp_cadvisor
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    privileged: true
    devices:
      - /dev/kmsg
    networks:
      - scripts_csp-network
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:latest
    container_name: csp_node_exporter
    command:
      - '--path.rootfs=/host'
    ports:
      - "9100:9100"
    volumes:
      - '/:/host:ro,rslave'
    pid: host
    networks:
      - scripts_csp-network
    restart: unless-stopped

networks:
  scripts_csp-network:
    external: true

volumes:
  prometheus_data:
  alertmanager_data:
  grafana_monitoring_data:
EOF

    log_success "Monitoring Docker Compose file created"
}

# Create simple webhook server for testing alerts
create_webhook_server() {
    log_info "Creating simple webhook server for alert testing..."
    
    cat > monitoring/webhook_server.py << 'EOF'
#!/usr/bin/env python3
"""
Simple webhook server to receive Alertmanager notifications
"""
import json
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    print(f"\n🚨 ALERT RECEIVED at {datetime.now()}")
    print("=" * 50)
    
    if 'alerts' in data:
        for alert in data['alerts']:
            status = alert.get('status', 'unknown')
            labels = alert.get('labels', {})
            annotations = alert.get('annotations', {})
            
            print(f"Status: {status}")
            print(f"Alert: {labels.get('alertname', 'Unknown')}")
            print(f"Severity: {labels.get('severity', 'Unknown')}")
            print(f"Summary: {annotations.get('summary', 'No summary')}")
            print(f"Description: {annotations.get('description', 'No description')}")
            print("-" * 30)
    
    print("=" * 50)
    return jsonify({"status": "received"})

@app.route('/webhook/critical', methods=['POST'])
def webhook_critical():
    data = request.get_json()
    print(f"\n🔥 CRITICAL ALERT at {datetime.now()}")
    print("=" * 50)
    
    if 'alerts' in data:
        for alert in data['alerts']:
            labels = alert.get('labels', {})
            annotations = alert.get('annotations', {})
            
            print(f"🚨 CRITICAL: {labels.get('alertname', 'Unknown')}")
            print(f"Summary: {annotations.get('summary', 'No summary')}")
            print(f"Description: {annotations.get('description', 'No description')}")
    
    print("=" * 50)
    return jsonify({"status": "critical_received"})

if __name__ == '__main__':
    print("🎯 Starting webhook server on http://localhost:5001")
    print("This will receive and display alerts from Alertmanager")
    app.run(host='0.0.0.0', port=5001, debug=True)
EOF

    chmod +x monitoring/webhook_server.py
    log_success "Webhook server created at monitoring/webhook_server.py"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    cd monitoring
    $COMPOSE_CMD -f docker-compose.monitoring.yml up -d
    cd ..
    
    log_success "Monitoring stack deployed!"
}

# Show access URLs
show_access_info() {
    echo ""
    log_success "🎉 Enhanced CSP Monitoring Stack is now running!"
    echo ""
    echo "📊 Access URLs:"
    echo "  • Prometheus:     http://localhost:9090"
    echo "  • Grafana:        http://localhost:3001 (admin/admin)"
    echo "  • Alertmanager:   http://localhost:9093"
    echo "  • cAdvisor:       http://localhost:8080"
    echo "  • Node Exporter:  http://localhost:9100"
    echo ""
    echo "🔧 Management Commands:"
    echo "  • View logs:      docker logs csp_prometheus"
    echo "  • Stop services:  cd monitoring && docker compose -f docker-compose.monitoring.yml down"
    echo "  • Start webhook:  python3 monitoring/webhook_server.py"
    echo ""
    echo "🚨 To test alerts:"
    echo "  1. Start webhook server: python3 monitoring/webhook_server.py"
    echo "  2. Stop a service: docker stop csp_redis"
    echo "  3. Check alerts in Alertmanager and webhook output"
    echo ""
}

# Main execution
main() {
    check_docker
    create_monitoring_structure
    create_prometheus_config
    create_alert_rules
    create_alertmanager_config
    create_grafana_datasource
    create_monitoring_compose
    create_webhook_server
    deploy_monitoring
    show_access_info
}

# Run the script
main "$@"