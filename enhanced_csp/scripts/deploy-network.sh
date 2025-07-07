#!/bin/bash
# deploy-network.sh - Deploy the network module with monitoring

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

# Configuration
COMPOSE_FILE="docker-compose.network.yml"
NETWORK_NAME="csp_network"

# Check prerequisites
log_step "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    log_error "Docker Compose is not installed"
    exit 1
fi

# Create required directories
log_step "Creating required directories..."
mkdir -p logs/network
mkdir -p database/network_init
mkdir -p monitoring/network_exporter

# Copy database schema if not exists
if [ ! -f "database/network_init/01-init-network.sql" ]; then
    log_warn "Network database schema not found. Please run setup-network-database.sh first"
    exit 1
fi

# Check if we should use existing database
log_step "Database configuration..."
echo "Do you want to:"
echo "1) Create a new PostgreSQL instance for network data"
echo "2) Use your existing enhanced_csp database"
read -p "Choose (1 or 2): " DB_CHOICE

if [ "$DB_CHOICE" = "2" ]; then
    log_info "Configuring to use existing database..."
    
    # Create a modified docker-compose that uses existing DB
    cat > docker-compose.network-temp.yml << 'EOF'
version: '3.8'

services:
  # Network Metrics Exporter only
  network_metrics_exporter:
    build:
      context: .
      dockerfile: Dockerfile.network-exporter
    container_name: network_metrics_exporter
    environment:
      DATABASE_URL: "postgresql+asyncpg://csp_user:csp_password@csp_postgres:5432/enhanced_csp"
      PROMETHEUS_PORT: "9200"
      PROMETHEUS_PUSHGATEWAY: "http://csp_pushgateway:9091"
      NODE_NAME: "network-master"
      LOG_LEVEL: "INFO"
    volumes:
      - ./network:/app/network:ro
      - ./monitoring/network_exporter:/app/monitoring/network_exporter:ro
      - ./logs/network:/var/log/csp/network
    ports:
      - "9200:9200"
    networks:
      - scripts_csp-network
    restart: unless-stopped

  # Network API Service
  network_api:
    build:
      context: .
      dockerfile: Dockerfile.network-api
    container_name: network_api
    environment:
      DATABASE_URL: "postgresql+asyncpg://csp_user:csp_password@csp_postgres:5432/enhanced_csp"
      REDIS_URL: "redis://csp_redis:6379/3"
      API_PORT: "8090"
    volumes:
      - ./network:/app/network:ro
      - ./logs/network:/var/log/csp/network
    ports:
      - "8090:8090"
    networks:
      - scripts_csp-network
    restart: unless-stopped

networks:
  scripts_csp-network:
    external: true
EOF
    
    COMPOSE_FILE="docker-compose.network-temp.yml"
    
    # Run database setup against existing database
    log_step "Setting up network schema in existing database..."
    PGPASSWORD=csp_password psql -h localhost -p 5432 -U csp_user -d enhanced_csp -f database/network_init/01-init-network.sql
fi

# Build images
log_step "Building Docker images..."
docker compose -f $COMPOSE_FILE build

# Start services
log_step "Starting network services..."
docker compose -f $COMPOSE_FILE up -d

# Wait for services to be healthy
log_step "Waiting for services to be ready..."
sleep 10

# Check service health
log_step "Checking service health..."

# Check metrics exporter
if curl -s http://localhost:9200/metrics > /dev/null; then
    log_info "✓ Network metrics exporter is running"
else
    log_warn "✗ Network metrics exporter is not responding"
fi

# Check API
if curl -s http://localhost:8090/health > /dev/null; then
    log_info "✓ Network API is running"
else
    log_warn "✗ Network API is not responding"
fi

# Display access information
echo ""
log_info "Network module deployed successfully!"
echo ""
echo "Access points:"
echo "  - Network API: http://localhost:8090"
echo "  - API Docs: http://localhost:8090/docs"
echo "  - Metrics: http://localhost:9200/metrics"
echo ""
echo "Next steps:"
echo "1. Update Prometheus configuration to scrape network_metrics_exporter:9200"
echo "2. Import network dashboards into Grafana"
echo "3. Configure alerting rules as needed"
echo ""
echo "To view logs:"
echo "  docker logs network_metrics_exporter"
echo "  docker logs network_api"
echo ""
echo "To stop services:"
echo "  docker compose -f $COMPOSE_FILE down"