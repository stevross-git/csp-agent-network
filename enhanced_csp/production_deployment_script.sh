#!/bin/bash

# Enhanced CSP System - Complete Production Deployment Script
# ===========================================================
# This script deploys the complete Enhanced CSP System to production
# with all components, monitoring, security, and automation.

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

log_step() {
    echo -e "\n${PURPLE}===================================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}===================================================${NC}\n"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_TYPE="${1:-production}"
DEPLOYMENT_TARGET="${2:-local}"
SKIP_TESTS="${3:-false}"

# Environment variables
export CSP_ENVIRONMENT="$DEPLOYMENT_TYPE"
export CSP_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET"
export CSP_VERSION="1.0.0"
export CSP_BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export CSP_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"

# Deployment configuration
DOCKER_REGISTRY="enhanced-csp"
KUBERNETES_NAMESPACE="enhanced-csp"
HELM_RELEASE_NAME="csp-system"

# ===========================================================================
# BANNER AND INITIALIZATION
# ===========================================================================

print_banner() {
    echo -e "${CYAN}"
    cat << 'EOF'
███████╗███╗   ██╗██╗  ██╗ █████╗ ███╗   ██╗ ██████╗███████╗██████╗ 
██╔════╝████╗  ██║██║  ██║██╔══██╗████╗  ██║██╔════╝██╔════╝██╔══██╗
█████╗  ██╔██╗ ██║███████║███████║██╔██╗ ██║██║     █████╗  ██║  ██║
██╔══╝  ██║╚██╗██║██╔══██║██╔══██║██║╚██╗██║██║     ██╔══╝  ██║  ██║
███████╗██║ ╚████║██║  ██║██║  ██║██║ ╚████║╚██████╗███████╗██████╔╝
╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝╚═════╝ 
                                                                      
 ██████╗███████╗██████╗     ███████╗██╗   ██╗███████╗████████╗███████╗███╗   ███╗
██╔════╝██╔════╝██╔══██╗    ██╔════╝╚██╗ ██╔╝██╔════╝╚══██╔══╝██╔════╝████╗ ████║
██║     ███████╗██████╔╝    ███████╗ ╚████╔╝ ███████╗   ██║   █████╗  ██╔████╔██║
██║     ╚════██║██╔═══╝     ╚════██║  ╚██╔╝  ╚════██║   ██║   ██╔══╝  ██║╚██╔╝██║
╚██████╗███████║██║         ███████║   ██║   ███████║   ██║   ███████╗██║ ╚═╝ ██║
 ╚═════╝╚══════╝╚═╝         ╚══════╝   ╚═╝   ╚══════╝   ╚═╝   ╚══════╝╚═╝     ╚═╝
EOF
    echo -e "${NC}"
    echo -e "${CYAN}Revolutionary AI-to-AI Communication Platform${NC}"
    echo -e "${CYAN}Version: $CSP_VERSION | Build: $CSP_BUILD_DATE${NC}"
    echo -e "${CYAN}Deployment: $DEPLOYMENT_TYPE -> $DEPLOYMENT_TARGET${NC}"
    echo ""
}

check_prerequisites() {
    log_step "CHECKING PREREQUISITES"
    
    local required_tools=("docker" "kubectl" "helm" "git" "curl" "jq")
    local missing_tools=()
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        else
            log_info "$tool is installed"
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_info "All prerequisites satisfied"
}

# ===========================================================================
# ENVIRONMENT SETUP
# ===========================================================================

setup_environment() {
    log_step "SETTING UP ENVIRONMENT"
    
    # Create necessary directories
    mkdir -p "$PROJECT_ROOT"/{logs,data,config,secrets,backups}
    
    # Generate environment configuration
    cat > "$PROJECT_ROOT/.env" << EOF
# Enhanced CSP System Environment Configuration
# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)

# System Configuration
CSP_ENVIRONMENT=$DEPLOYMENT_TYPE
CSP_VERSION=$CSP_VERSION
CSP_BUILD_DATE=$CSP_BUILD_DATE
CSP_GIT_COMMIT=$CSP_GIT_COMMIT

# Network Configuration
CSP_HOST=0.0.0.0
CSP_PORT=8000
CSP_ENABLE_HTTPS=true
CSP_MAX_CONNECTIONS=1000

# Database Configuration
CSP_DATABASE_URL=${CSP_DATABASE_URL:-postgresql+asyncpg://csp_user:csp_pass@postgres:5432/csp_db}
CSP_DATABASE_POOL_SIZE=20
CSP_DATABASE_MAX_OVERFLOW=30

# Redis Configuration
CSP_REDIS_URL=${CSP_REDIS_URL:-redis://redis:6379/0}
CSP_REDIS_MAX_CONNECTIONS=20

# AI Services Configuration
CSP_ENABLE_AI_INTEGRATION=true
OPENAI_API_KEY=${OPENAI_API_KEY:-}
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-}

# Quantum Computing Configuration
CSP_ENABLE_QUANTUM=true
QUANTUM_BACKEND=qiskit_simulator
QUANTUM_TOKEN=${QUANTUM_TOKEN:-}

# Blockchain Configuration
CSP_ENABLE_BLOCKCHAIN=true
BLOCKCHAIN_NETWORK=ethereum_testnet
WEB3_PROVIDER_URL=${WEB3_PROVIDER_URL:-}

# Security Configuration
CSP_ENABLE_ZERO_TRUST=true
CSP_ENABLE_MFA=true
CSP_SECURITY_KEY=${CSP_SECURITY_KEY:-$(openssl rand -base64 32)}
CSP_ENCRYPTION_KEY=${CSP_ENCRYPTION_KEY:-$(openssl rand -base64 32)}

# Performance Configuration
CSP_ENABLE_PERFORMANCE_OPTIMIZATION=true
CSP_ENABLE_AUTO_SCALING=true
CSP_PERFORMANCE_MONITORING_INTERVAL=10

# Monitoring Configuration
CSP_ENABLE_PROMETHEUS=true
CSP_ENABLE_GRAFANA=true
CSP_METRICS_RETENTION_DAYS=30

# Feature Flags
CSP_ENABLE_WEB_DASHBOARD=true
CSP_ENABLE_REAL_TIME_MONITORING=true
CSP_ENABLE_AUTONOMOUS_MANAGEMENT=true

# Debug Configuration
CSP_DEBUG=${CSP_DEBUG:-false}
CSP_LOG_LEVEL=${CSP_LOG_LEVEL:-INFO}
EOF
    
    log_info "Environment configuration created"
    
    # Set proper permissions
    chmod 600 "$PROJECT_ROOT/.env"
    
    # Create log directory structure
    mkdir -p "$PROJECT_ROOT/logs"/{application,system,security,audit}
    
    log_info "Environment setup completed"
}

# ===========================================================================
# DEPENDENCY INSTALLATION
# ===========================================================================

install_dependencies() {
    log_step "INSTALLING DEPENDENCIES"
    
    # Update system packages
    if command -v apt-get &> /dev/null; then
        log_info "Updating system packages (Ubuntu/Debian)"
        sudo apt-get update -y
        sudo apt-get install -y \
            python3.11 \
            python3.11-dev \
            python3-pip \
            python3-venv \
            postgresql-client \
            redis-tools \
            curl \
            wget \
            git \
            build-essential \
            libssl-dev \
            libffi-dev \
            nginx \
            certbot \
            python3-certbot-nginx
    elif command -v yum &> /dev/null; then
        log_info "Updating system packages (CentOS/RHEL)"
        sudo yum update -y
        sudo yum install -y \
            python3.11 \
            python3.11-devel \
            python3-pip \
            postgresql \
            redis \
            curl \
            wget \
            git \
            gcc \
            gcc-c++ \
            make \
            openssl-devel \
            libffi-devel \
            nginx \
            certbot \
            python3-certbot-nginx
    fi
    
    # Create Python virtual environment
    log_info "Creating Python virtual environment"
    python3.11 -m venv "$PROJECT_ROOT/venv"
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Upgrade pip and install wheel
    pip install --upgrade pip setuptools wheel
    
    # Install Python dependencies
    log_info "Installing Python dependencies"
    pip install -r "$PROJECT_ROOT/requirements.txt"
    
    # Install development dependencies if needed
    if [ "$DEPLOYMENT_TYPE" = "development" ]; then
        pip install -r "$PROJECT_ROOT/requirements-dev.txt"
    fi
    
    log_info "Dependencies installation completed"
}

# ===========================================================================
# DATABASE SETUP
# ===========================================================================

setup_database() {
    log_step "SETTING UP DATABASE"
    
    if [ "$DEPLOYMENT_TARGET" = "local" ]; then
        # Local PostgreSQL setup
        log_info "Setting up local PostgreSQL"
        
        # Install PostgreSQL if not present
        if ! command -v psql &> /dev/null; then
            if command -v apt-get &> /dev/null; then
                sudo apt-get install -y postgresql postgresql-contrib
            elif command -v yum &> /dev/null; then
                sudo yum install -y postgresql-server postgresql-contrib
                sudo postgresql-setup initdb
            fi
        fi
        
        # Start PostgreSQL service
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
        
        # Create database and user
        sudo -u postgres psql << EOF
CREATE USER csp_user WITH PASSWORD 'csp_pass';
CREATE DATABASE csp_db OWNER csp_user;
GRANT ALL PRIVILEGES ON DATABASE csp_db TO csp_user;
\q
EOF
        
        log_info "Local PostgreSQL setup completed"
    fi
    
    # Run database migrations
    log_info "Running database migrations"
    source "$PROJECT_ROOT/venv/bin/activate"
    cd "$PROJECT_ROOT"
    
    # Create Alembic configuration if it doesn't exist
    if [ ! -f "alembic.ini" ]; then
        log_info "Initializing Alembic"
        alembic init alembic
    fi
    
    # Run migrations
    alembic upgrade head
    
    log_info "Database setup completed"
}

# ===========================================================================
# DOCKER BUILD AND DEPLOYMENT
# ===========================================================================

build_docker_images() {
    log_step "BUILDING DOCKER IMAGES"
    
    cd "$PROJECT_ROOT"
    
    # Build core application image
    log_info "Building core application image"
    docker build \
        --tag "$DOCKER_REGISTRY/core:$CSP_VERSION" \
        --tag "$DOCKER_REGISTRY/core:latest" \
        --build-arg CSP_VERSION="$CSP_VERSION" \
        --build-arg CSP_BUILD_DATE="$CSP_BUILD_DATE" \
        --build-arg CSP_GIT_COMMIT="$CSP_GIT_COMMIT" \
        -f docker/Dockerfile.core .
    
    # Build component images
    local components=("quantum" "blockchain" "neural" "ai-hub" "security" "visualizer" "controller")
    
    for component in "${components[@]}"; do
        log_info "Building $component image"
        docker build \
            --tag "$DOCKER_REGISTRY/$component:$CSP_VERSION" \
            --tag "$DOCKER_REGISTRY/$component:latest" \
            --build-arg CSP_VERSION="$CSP_VERSION" \
            -f "docker/Dockerfile.$component" .
    done
    
    # Build web UI image
    log_info "Building web UI image"
    docker build \
        --tag "$DOCKER_REGISTRY/web-ui:$CSP_VERSION" \
        --tag "$DOCKER_REGISTRY/web-ui:latest" \
        -f docker/Dockerfile.web-ui .
    
    log_info "Docker images built successfully"
}

deploy_with_docker_compose() {
    log_step "DEPLOYING WITH DOCKER COMPOSE"
    
    cd "$PROJECT_ROOT"
    
    # Generate docker-compose.yml for the deployment
    cat > docker-compose.yml << EOF
version: '3.8'

services:
  # Core CSP Engine
  csp-core:
    image: $DOCKER_REGISTRY/core:$CSP_VERSION
    container_name: enhanced-csp-core
    ports:
      - "8000:8000"
    environment:
      - CSP_ENVIRONMENT=$DEPLOYMENT_TYPE
      - CSP_DATABASE_URL=postgresql://csp_user:csp_pass@postgres:5432/csp_db
      - CSP_REDIS_URL=redis://redis:6379/0
    env_file:
      - .env
    depends_on:
      - postgres
      - redis
    networks:
      - csp-network
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Quantum Computing Service
  csp-quantum:
    image: $DOCKER_REGISTRY/quantum:$CSP_VERSION
    container_name: enhanced-csp-quantum
    ports:
      - "8001:8001"
    environment:
      - CSP_CORE_URL=http://csp-core:8000
    env_file:
      - .env
    depends_on:
      - csp-core
    networks:
      - csp-network
    restart: unless-stopped

  # AI Hub Service
  csp-ai-hub:
    image: $DOCKER_REGISTRY/ai-hub:$CSP_VERSION
    container_name: enhanced-csp-ai-hub
    ports:
      - "8004:8004"
    environment:
      - CSP_CORE_URL=http://csp-core:8000
    env_file:
      - .env
    depends_on:
      - csp-core
    networks:
      - csp-network
    restart: unless-stopped

  # Security Engine
  csp-security:
    image: $DOCKER_REGISTRY/security:$CSP_VERSION
    container_name: enhanced-csp-security
    ports:
      - "8005:8005"
    environment:
      - CSP_CORE_URL=http://csp-core:8000
    env_file:
      - .env
    depends_on:
      - csp-core
    networks:
      - csp-network
    restart: unless-stopped

  # Web Dashboard
  csp-web-ui:
    image: $DOCKER_REGISTRY/web-ui:$CSP_VERSION
    container_name: enhanced-csp-web-ui
    ports:
      - "3000:3000"
    environment:
      - CSP_API_URL=http://csp-core:8000
    depends_on:
      - csp-core
    networks:
      - csp-network
    restart: unless-stopped

  # Database
  postgres:
    image: postgres:15
    container_name: enhanced-csp-postgres
    environment:
      POSTGRES_DB: csp_db
      POSTGRES_USER: csp_user
      POSTGRES_PASSWORD: csp_pass
    ports:
      - "5432:5432"
    networks:
      - csp-network
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  # Redis
  redis:
    image: redis:7-alpine
    container_name: enhanced-csp-redis
    ports:
      - "6379:6379"
    networks:
      - csp-network
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  # Monitoring - Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: enhanced-csp-prometheus
    ports:
      - "9090:9090"
    networks:
      - csp-network
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  # Monitoring - Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: enhanced-csp-grafana
    ports:
      - "3001:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: csp_admin
    networks:
      - csp-network
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    restart: unless-stopped

  # Load Balancer - Nginx
  nginx:
    image: nginx:alpine
    container_name: enhanced-csp-nginx
    ports:
      - "80:80"
      - "443:443"
    networks:
      - csp-network
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - csp-core
    restart: unless-stopped

networks:
  csp-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
EOF
    
    # Start services
    log_info "Starting services with Docker Compose"
    docker-compose up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Verify deployment
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "✅ Core service is healthy"
    else
        log_error "❌ Core service health check failed"
        return 1
    fi
    
    log_info "Docker Compose deployment completed"
}

deploy_to_kubernetes() {
    log_step "DEPLOYING TO KUBERNETES"
    
    # Create namespace
    kubectl create namespace "$KUBERNETES_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets
    kubectl create secret generic csp-secrets \
        --from-literal=database-url="$CSP_DATABASE_URL" \
        --from-literal=redis-url="$CSP_REDIS_URL" \
        --from-literal=openai-api-key="$OPENAI_API_KEY" \
        --from-literal=security-key="$CSP_SECURITY_KEY" \
        --namespace="$KUBERNETES_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy using Helm
    if [ ! -d "helm/enhanced-csp" ]; then
        log_error "Helm chart not found"
        return 1
    fi
    
    helm upgrade --install "$HELM_RELEASE_NAME" ./helm/enhanced-csp \
        --namespace "$KUBERNETES_NAMESPACE" \
        --set image.tag="$CSP_VERSION" \
        --set environment="$DEPLOYMENT_TYPE" \
        --values "helm/values-$DEPLOYMENT_TYPE.yaml" \
        --wait --timeout=10m
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/csp-core --namespace="$KUBERNETES_NAMESPACE" --timeout=300s
    
    # Get service URLs
    local service_url
    if kubectl get service csp-ingress --namespace="$KUBERNETES_NAMESPACE" > /dev/null 2>&1; then
        service_url=$(kubectl get service csp-ingress --namespace="$KUBERNETES_NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        log_info "Service URL: http://$service_url"
    fi
    
    log_info "Kubernetes deployment completed"
}

# ===========================================================================
# TESTING AND VALIDATION
# ===========================================================================

run_tests() {
    log_step "RUNNING TESTS"
    
    if [ "$SKIP_TESTS" = "true" ]; then
        log_warn "Skipping tests as requested"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    source "venv/bin/activate"
    
    # Run unit tests
    log_info "Running unit tests"
    python -m pytest tests/unit/ -v --tb=short
    
    # Run integration tests
    log_info "Running integration tests"
    python -m pytest tests/integration/ -v --tb=short
    
    # Run security tests
    log_info "Running security tests"
    python -m pytest tests/security/ -v --tb=short
    
    # Run performance tests
    log_info "Running performance tests"
    python -m pytest tests/performance/ -v --tb=short --benchmark-only
    
    # Run system health check
    log_info "Running system health check"
    python -c "
import asyncio
import sys
sys.path.append('.')
from master_system_orchestrator import EnhancedCSPSystemOrchestrator, EnhancedCSPConfig

async def health_check():
    config = EnhancedCSPConfig()
    system = EnhancedCSPSystemOrchestrator(config)
    try:
        await system.initialize()
        status = await system.get_system_status()
        print(f'System status: {status[\"system_info\"][\"running\"]}')
        return 0 if status['system_info']['running'] else 1
    except Exception as e:
        print(f'Health check failed: {e}')
        return 1
    finally:
        await system.shutdown()

exit_code = asyncio.run(health_check())
exit(exit_code)
"
    
    log_info "All tests completed successfully"
}

# ===========================================================================
# MONITORING SETUP
# ===========================================================================

setup_monitoring() {
    log_step "SETTING UP MONITORING"
    
    # Create Prometheus configuration
    mkdir -p "$PROJECT_ROOT/monitoring"
    
    cat > "$PROJECT_ROOT/monitoring/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'enhanced-csp'
    static_configs:
      - targets: ['csp-core:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
      
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
EOF
    
    # Create alert rules
    cat > "$PROJECT_ROOT/monitoring/alert_rules.yml" << EOF
groups:
  - name: enhanced-csp-alerts
    rules:
      - alert: HighCPUUsage
        expr: cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for more than 5 minutes"
          
      - alert: HighMemoryUsage
        expr: memory_usage_percent > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 85% for more than 5 minutes"
          
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "Service {{ \$labels.instance }} is down"
EOF
    
    # Create Grafana dashboard
    mkdir -p "$PROJECT_ROOT/monitoring/grafana/dashboards"
    
    # Copy dashboard files (assuming they exist)
    if [ -f "monitoring/dashboards/enhanced-csp-dashboard.json" ]; then
        cp monitoring/dashboards/*.json "$PROJECT_ROOT/monitoring/grafana/dashboards/"
    fi
    
    log_info "Monitoring setup completed"
}

# ===========================================================================
# SSL/TLS SETUP
# ===========================================================================

setup_ssl() {
    log_step "SETTING UP SSL/TLS"
    
    if [ "$DEPLOYMENT_TARGET" = "local" ]; then
        log_info "Generating self-signed certificates for local deployment"
        
        mkdir -p "$PROJECT_ROOT/nginx/ssl"
        
        # Generate self-signed certificate
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$PROJECT_ROOT/nginx/ssl/nginx.key" \
            -out "$PROJECT_ROOT/nginx/ssl/nginx.crt" \
            -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=localhost"
        
        log_info "Self-signed certificates generated"
    else
        log_info "Setting up Let's Encrypt certificates"
        
        # Install certbot and obtain certificates
        sudo certbot --nginx -d "$CSP_DOMAIN" --non-interactive --agree-tos --email "$CSP_EMAIL"
        
        # Setup automatic renewal
        echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
        
        log_info "Let's Encrypt certificates configured"
    fi
}

# ===========================================================================
# BACKUP CONFIGURATION
# ===========================================================================

setup_backup() {
    log_step "SETTING UP BACKUP SYSTEM"
    
    mkdir -p "$PROJECT_ROOT/scripts/backup"
    
    # Create backup script
    cat > "$PROJECT_ROOT/scripts/backup/backup.sh" << 'EOF'
#!/bin/bash

# Enhanced CSP System Backup Script
BACKUP_DIR="/opt/enhanced-csp/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Database backup
echo "Backing up database..."
pg_dump -h localhost -U csp_user csp_db | gzip > "$BACKUP_DIR/database_backup_$DATE.sql.gz"

# Configuration backup
echo "Backing up configuration..."
tar -czf "$BACKUP_DIR/config_backup_$DATE.tar.gz" -C /opt/enhanced-csp config/ .env

# Application data backup
echo "Backing up application data..."
tar -czf "$BACKUP_DIR/data_backup_$DATE.tar.gz" -C /opt/enhanced-csp data/

# Clean old backups
echo "Cleaning old backups..."
find "$BACKUP_DIR" -name "*.gz" -mtime +$RETENTION_DAYS -delete

# Upload to cloud storage (if configured)
if [ -n "$AWS_S3_BUCKET" ]; then
    echo "Uploading to S3..."
    aws s3 cp "$BACKUP_DIR/database_backup_$DATE.sql.gz" "s3://$AWS_S3_BUCKET/backups/database/"
    aws s3 cp "$BACKUP_DIR/config_backup_$DATE.tar.gz" "s3://$AWS_S3_BUCKET/backups/config/"
    aws s3 cp "$BACKUP_DIR/data_backup_$DATE.tar.gz" "s3://$AWS_S3_BUCKET/backups/data/"
fi

echo "Backup completed: $DATE"
EOF
    
    chmod +x "$PROJECT_ROOT/scripts/backup/backup.sh"
    
    # Setup daily backup cron job
    (crontab -l 2>/dev/null; echo "0 2 * * * $PROJECT_ROOT/scripts/backup/backup.sh") | crontab -
    
    log_info "Backup system configured"
}

# ===========================================================================
# SYSTEM VALIDATION
# ===========================================================================

validate_deployment() {
    log_step "VALIDATING DEPLOYMENT"
    
    local validation_errors=0
    
    # Check core service
    log_info "Checking core service..."
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "✅ Core service is healthy"
    else
        log_error "❌ Core service is not responding"
        ((validation_errors++))
    fi
    
    # Check API endpoints
    log_info "Checking API endpoints..."
    local endpoints=("/api/status" "/api/metrics" "/api/processes")
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f "http://localhost:8000$endpoint" > /dev/null 2>&1; then
            log_info "✅ API endpoint $endpoint is accessible"
        else
            log_warn "⚠️  API endpoint $endpoint is not accessible"
        fi
    done
    
    # Check web dashboard
    log_info "Checking web dashboard..."
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        log_info "✅ Web dashboard is accessible"
    else
        log_warn "⚠️  Web dashboard is not accessible"
    fi
    
    # Check monitoring
    log_info "Checking monitoring services..."
    if curl -f http://localhost:9090 > /dev/null 2>&1; then
        log_info "✅ Prometheus is accessible"
    else
        log_warn "⚠️  Prometheus is not accessible"
    fi
    
    if curl -f http://localhost:3001 > /dev/null 2>&1; then
        log_info "✅ Grafana is accessible"
    else
        log_warn "⚠️  Grafana is not accessible"
    fi
    
    # Check database connectivity
    log_info "Checking database connectivity..."
    if PGPASSWORD=csp_pass psql -h localhost -U csp_user -d csp_db -c "SELECT 1;" > /dev/null 2>&1; then
        log_info "✅ Database is accessible"
    else
        log_error "❌ Database is not accessible"
        ((validation_errors++))
    fi
    
    # Check Redis connectivity
    log_info "Checking Redis connectivity..."
    if redis-cli ping > /dev/null 2>&1; then
        log_info "✅ Redis is accessible"
    else
        log_error "❌ Redis is not accessible"
        ((validation_errors++))
    fi
    
    # Summary
    if [ $validation_errors -eq 0 ]; then
        log_info "🎉 All validation checks passed!"
        return 0
    else
        log_error "❌ $validation_errors validation errors detected"
        return 1
    fi
}

# ===========================================================================
# MAIN DEPLOYMENT LOGIC
# ===========================================================================

deploy_enhanced_csp_system() {
    local start_time=$(date +%s)
    
    print_banner
    
    # Check prerequisites
    check_prerequisites
    
    # Setup environment
    setup_environment
    
    # Install dependencies
    install_dependencies
    
    # Setup database
    setup_database
    
    # Build Docker images
    build_docker_images
    
    # Deploy based on target
    case "$DEPLOYMENT_TARGET" in
        "local")
            deploy_with_docker_compose
            ;;
        "kubernetes")
            deploy_to_kubernetes
            ;;
        *)
            log_error "Unknown deployment target: $DEPLOYMENT_TARGET"
            exit 1
            ;;
    esac
    
    # Setup monitoring
    setup_monitoring
    
    # Setup SSL/TLS
    setup_ssl
    
    # Setup backup system
    setup_backup
    
    # Run tests
    run_tests
    
    # Validate deployment
    validate_deployment
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_step "DEPLOYMENT COMPLETED"
    log_info "Deployment completed in ${duration}s"
    log_info "Environment: $DEPLOYMENT_TYPE"
    log_info "Target: $DEPLOYMENT_TARGET"
    log_info "Version: $CSP_VERSION"
    echo ""
    log_info "🌐 Access URLs:"
    log_info "   Main Application: http://localhost:8000"
    log_info "   Web Dashboard: http://localhost:3000"
    log_info "   Prometheus: http://localhost:9090"
    log_info "   Grafana: http://localhost:3001 (admin/csp_admin)"
    echo ""
    log_info "📚 Documentation: https://docs.enhanced-csp.com"
    log_info "🆘 Support: support@enhanced-csp.com"
    echo ""
    log_info "🎉 Enhanced CSP System is now running!"
}

# ===========================================================================
# SCRIPT EXECUTION
# ===========================================================================

# Main execution
case "${1:-help}" in
    "production"|"staging"|"development")
        deploy_enhanced_csp_system
        ;;
    "help"|"--help"|"-h")
        echo "Enhanced CSP System Deployment Script"
        echo ""
        echo "Usage: $0 [DEPLOYMENT_TYPE] [DEPLOYMENT_TARGET] [SKIP_TESTS]"
        echo ""
        echo "DEPLOYMENT_TYPE:"
        echo "  production  - Production deployment"
        echo "  staging     - Staging deployment"
        echo "  development - Development deployment"
        echo ""
        echo "DEPLOYMENT_TARGET:"
        echo "  local       - Local deployment with Docker Compose"
        echo "  kubernetes  - Kubernetes deployment"
        echo ""
        echo "SKIP_TESTS:"
        echo "  true        - Skip running tests"
        echo "  false       - Run all tests (default)"
        echo ""
        echo "Examples:"
        echo "  $0 production kubernetes"
        echo "  $0 development local"
        echo "  $0 staging kubernetes true"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
