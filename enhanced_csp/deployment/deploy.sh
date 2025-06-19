#!/bin/bash
# File: deploy/deploy.sh
# CSP Visual Designer Backend - Production Deployment Script
# =========================================================

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="csp-visual-designer"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Functions
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
    exit 1
}

check_requirements() {
    log_info "Checking deployment requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
    fi
    
    # Check kubectl (for Kubernetes deployment)
    if ! command -v kubectl &> /dev/null; then
        log_warning "kubectl is not installed - Kubernetes deployment will not be available"
    fi
    
    log_success "Requirements check completed"
}

build_docker_images() {
    log_info "Building Docker images..."
    
    # Build main application image
    docker build -t ${PROJECT_NAME}:${VERSION} \
        --target production \
        --build-arg VERSION=${VERSION} \
        .
    
    # Tag for registry
    docker tag ${PROJECT_NAME}:${VERSION} ${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}
    
    log_success "Docker images built successfully"
}

push_docker_images() {
    log_info "Pushing Docker images to registry..."
    
    # Login to registry (if needed)
    if [[ "${DOCKER_REGISTRY}" != "localhost:5000" ]]; then
        docker login ${DOCKER_REGISTRY}
    fi
    
    # Push images
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}:${VERSION}
    
    log_success "Docker images pushed successfully"
}

setup_environment() {
    log_info "Setting up environment configuration..."
    
    # Create environment-specific configuration
    if [[ ! -f ".env.${ENVIRONMENT}" ]]; then
        log_warning "Environment file .env.${ENVIRONMENT} not found, creating from template..."
        cp .env.example .env.${ENVIRONMENT}
        
        log_warning "Please edit .env.${ENVIRONMENT} with your configuration"
        read -p "Press enter to continue after editing the file..."
    fi
    
    # Copy environment file
    cp .env.${ENVIRONMENT} .env
    
    log_success "Environment configuration completed"
}

deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    # Stop existing containers
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml down
    
    # Pull latest images
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml pull
    
    # Start services
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Run database migrations
    docker-compose exec api python cli/manage.py db migrate --force
    
    # Create admin user (if needed)
    log_info "Creating initial admin user (if needed)..."
    docker-compose exec api python cli/manage.py users create-admin \
        --username admin \
        --password "${ADMIN_PASSWORD:-admin123}" \
        --email "${ADMIN_EMAIL:-admin@example.com}"
    
    log_success "Docker Compose deployment completed"
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is required for Kubernetes deployment"
    fi
    
    # Apply namespace
    kubectl apply -f k8s/namespace.yaml
    
    # Apply configmaps and secrets
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secrets.yaml
    
    # Apply persistent volumes
    kubectl apply -f k8s/persistent-volumes.yaml
    
    # Deploy databases
    kubectl apply -f k8s/postgres.yaml
    kubectl apply -f k8s/redis.yaml
    
    # Wait for databases to be ready
    log_info "Waiting for databases to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres --timeout=300s -n csp-system
    kubectl wait --for=condition=ready pod -l app=redis --timeout=300s -n csp-system
    
    # Deploy application
    kubectl apply -f k8s/api-deployment.yaml
    kubectl apply -f k8s/api-service.yaml
    kubectl apply -f k8s/ingress.yaml
    
    # Deploy monitoring
    kubectl apply -f k8s/prometheus.yaml
    kubectl apply -f k8s/grafana.yaml
    
    # Wait for application to be ready
    kubectl wait --for=condition=ready pod -l app=csp-api --timeout=300s -n csp-system
    
    # Run database migrations
    kubectl exec -n csp-system deployment/csp-api -- python cli/manage.py db migrate --force
    
    log_success "Kubernetes deployment completed"
}

health_check() {
    log_info "Performing health check..."
    
    # Determine health check URL
    if [[ "${DEPLOYMENT_TYPE}" == "kubernetes" ]]; then
        # Get ingress URL or use port-forward
        API_URL="http://localhost:8000"  # Adjust based on your ingress configuration
    else
        API_URL="http://localhost:8000"
    fi
    
    # Wait for API to be ready
    for i in {1..30}; do
        if curl -f ${API_URL}/health > /dev/null 2>&1; then
            log_success "Health check passed - API is responding"
            break
        fi
        
        if [[ $i -eq 30 ]]; then
            log_error "Health check failed - API is not responding after 5 minutes"
        fi
        
        log_info "Waiting for API to be ready... (attempt $i/30)"
        sleep 10
    done
    
    # Additional health checks
    log_info "Running additional health checks..."
    
    # Check database connectivity
    response=$(curl -s ${API_URL}/health | jq -r '.components.database.status')
    if [[ "${response}" != "healthy" ]]; then
        log_error "Database health check failed"
    fi
    
    # Check Redis connectivity
    response=$(curl -s ${API_URL}/health | jq -r '.components.redis.status')
    if [[ "${response}" != "healthy" ]]; then
        log_error "Redis health check failed"
    fi
    
    log_success "All health checks passed"
}

show_deployment_info() {
    log_info "Deployment completed successfully!"
    
    echo ""
    echo "=== Deployment Information ==="
    echo "Project: ${PROJECT_NAME}"
    echo "Version: ${VERSION}"
    echo "Environment: ${ENVIRONMENT}"
    echo "Deployment Type: ${DEPLOYMENT_TYPE}"
    echo ""
    
    if [[ "${DEPLOYMENT_TYPE}" == "kubernetes" ]]; then
        echo "=== Kubernetes Resources ==="
        kubectl get all -n csp-system
        echo ""
        
        echo "=== Access Information ==="
        echo "API URL: Check your ingress configuration"
        echo "Monitoring: kubectl port-forward svc/prometheus 9090:9090"
        echo "Grafana: kubectl port-forward svc/grafana 3000:3000"
    else
        echo "=== Docker Compose Services ==="
        docker-compose ps
        echo ""
        
        echo "=== Access Information ==="
        echo "API URL: http://localhost:8000"
        echo "API Docs: http://localhost:8000/docs"
        echo "Prometheus: http://localhost:9090"
        echo "Grafana: http://localhost:3000 (admin/admin)"
        echo "pgAdmin: http://localhost:5050 (admin@example.com/admin)"
    fi
    
    echo ""
    echo "=== Next Steps ==="
    echo "1. Configure your domain and SSL certificates"
    echo "2. Set up monitoring alerts"
    echo "3. Configure backup procedures"
    echo "4. Review security settings"
    echo "5. Load test your deployment"
}

# Main deployment function
main() {
    echo "========================================"
    echo "  CSP Visual Designer Backend Deploy   "
    echo "========================================"
    echo ""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment|-e)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --version|-v)
                VERSION="$2"
                shift 2
                ;;
            --deployment-type|-t)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-push)
                SKIP_PUSH=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  -e, --environment     Deployment environment (default: production)"
                echo "  -v, --version         Version tag (default: latest)"
                echo "  -t, --deployment-type Deployment type (docker-compose|kubernetes)"
                echo "  --skip-build          Skip Docker image building"
                echo "  --skip-push           Skip Docker image pushing"
                echo "  -h, --help            Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                ;;
        esac
    done
    
    # Set default deployment type
    DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-docker-compose}"
    
    # Validate deployment type
    if [[ "${DEPLOYMENT_TYPE}" != "docker-compose" && "${DEPLOYMENT_TYPE}" != "kubernetes" ]]; then
        log_error "Invalid deployment type. Must be 'docker-compose' or 'kubernetes'"
    fi
    
    log_info "Starting deployment with the following configuration:"
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Version: ${VERSION}"
    log_info "Deployment Type: ${DEPLOYMENT_TYPE}"
    echo ""
    
    # Run deployment steps
    check_requirements
    setup_environment
    
    if [[ "${SKIP_BUILD}" != "true" ]]; then
        build_docker_images
    fi
    
    if [[ "${SKIP_PUSH}" != "true" && "${DOCKER_REGISTRY}" != "localhost:5000" ]]; then
        push_docker_images
    fi
    
    if [[ "${DEPLOYMENT_TYPE}" == "kubernetes" ]]; then
        deploy_kubernetes
    else
        deploy_docker_compose
    fi
    
    health_check
    show_deployment_info
    
    log_success "Deployment completed successfully! 🚀"
}

# Trap errors and cleanup
trap 'log_error "Deployment failed at line $LINENO"' ERR

# Run main function
main "$@"

# ============================================================================
# Additional deployment scripts
# ============================================================================

# File: deploy/rollback.sh
#!/bin/bash
# Rollback deployment to previous version

set -e

log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
    exit 1
}

rollback_docker_compose() {
    log_info "Rolling back Docker Compose deployment..."
    
    # Get previous version from backup
    if [[ -f ".env.backup" ]]; then
        cp .env.backup .env
        docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
        log_info "Rollback completed"
    else
        log_error "No backup configuration found"
    fi
}

rollback_kubernetes() {
    log_info "Rolling back Kubernetes deployment..."
    
    # Rollback deployment
    kubectl rollout undo deployment/csp-api -n csp-system
    
    # Wait for rollback to complete
    kubectl rollout status deployment/csp-api -n csp-system
    
    log_info "Kubernetes rollback completed"
}

# Main rollback function
if [[ "${DEPLOYMENT_TYPE:-docker-compose}" == "kubernetes" ]]; then
    rollback_kubernetes
else
    rollback_docker_compose
fi

# File: deploy/backup.sh
#!/bin/bash
# Backup script for CSP Visual Designer

set -e

BACKUP_DIR="${BACKUP_DIR:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

create_backup_dir() {
    mkdir -p ${BACKUP_DIR}
}

backup_database() {
    log_info "Backing up PostgreSQL database..."
    
    # Extract database connection details from environment
    source .env
    
    pg_dump -h ${DB_HOST:-localhost} \
            -p ${DB_PORT:-5432} \
            -U ${DB_USER:-csp_user} \
            -d ${DB_NAME:-csp_visual_designer} \
            -f ${BACKUP_DIR}/database_${TIMESTAMP}.sql \
            --verbose
    
    # Compress backup
    gzip ${BACKUP_DIR}/database_${TIMESTAMP}.sql
    
    log_info "Database backup completed: ${BACKUP_DIR}/database_${TIMESTAMP}.sql.gz"
}

backup_redis() {
    log_info "Backing up Redis data..."
    
    # Copy Redis dump file
    docker-compose exec redis redis-cli BGSAVE
    sleep 5
    docker cp $(docker-compose ps -q redis):/data/dump.rdb ${BACKUP_DIR}/redis_${TIMESTAMP}.rdb
    
    log_info "Redis backup completed: ${BACKUP_DIR}/redis_${TIMESTAMP}.rdb"
}

backup_configuration() {
    log_info "Backing up configuration files..."
    
    tar -czf ${BACKUP_DIR}/config_${TIMESTAMP}.tar.gz \
        .env \
        docker-compose.yml \
        docker-compose.prod.yml \
        k8s/ \
        config/
    
    log_info "Configuration backup completed: ${BACKUP_DIR}/config_${TIMESTAMP}.tar.gz"
}

cleanup_old_backups() {
    log_info "Cleaning up old backups (keeping last 7 days)..."
    
    find ${BACKUP_DIR} -name "*.sql.gz" -mtime +7 -delete
    find ${BACKUP_DIR} -name "*.rdb" -mtime +7 -delete
    find ${BACKUP_DIR} -name "*.tar.gz" -mtime +7 -delete
    
    log_info "Cleanup completed"
}

# Main backup function
create_backup_dir
backup_database
backup_redis
backup_configuration
cleanup_old_backups

log_info "Backup process completed successfully!"

# File: deploy/monitoring.sh
#!/bin/bash
# Monitoring setup script

set -e

log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

setup_prometheus() {
    log_info "Setting up Prometheus configuration..."
    
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'csp-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF

    log_info "Prometheus configuration created"
}

setup_grafana_dashboards() {
    log_info "Setting up Grafana dashboards..."
    
    mkdir -p monitoring/grafana/dashboards
    
    # Create datasource configuration
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    # Create dashboard for CSP metrics
    cat > monitoring/grafana/dashboards/csp-dashboard.json << EOF
{
  "dashboard": {
    "title": "CSP Visual Designer",
    "panels": [
      {
        "title": "HTTP Requests",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(csp_http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(csp_http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
EOF

    log_info "Grafana dashboards created"
}

setup_alerting() {
    log_info "Setting up alerting rules..."
    
    cat > monitoring/alert_rules.yml << EOF
groups:
  - name: csp_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(csp_http_requests_total{status_code=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} for the last 5 minutes"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(csp_http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }}s"

      - alert: DatabaseDown
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database is down"
          description: "PostgreSQL database is not responding"
EOF

    log_info "Alerting rules created"
}

# Main monitoring setup
mkdir -p monitoring/grafana/datasources
setup_prometheus
setup_grafana_dashboards
setup_alerting

log_info "Monitoring setup completed!"

# File: deploy/ssl-setup.sh
#!/bin/bash
# SSL certificate setup using Let's Encrypt

set -e

DOMAIN="${DOMAIN:-api.example.com}"
EMAIL="${EMAIL:-admin@example.com}"

log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

install_certbot() {
    log_info "Installing Certbot..."
    
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y certbot python3-certbot-nginx
    elif command -v yum &> /dev/null; then
        sudo yum install -y certbot python3-certbot-nginx
    else
        log_error "Package manager not supported"
    fi
}

obtain_certificate() {
    log_info "Obtaining SSL certificate for ${DOMAIN}..."
    
    sudo certbot certonly \
        --nginx \
        --email ${EMAIL} \
        --agree-tos \
        --no-eff-email \
        -d ${DOMAIN}
    
    log_info "SSL certificate obtained successfully"
}

setup_auto_renewal() {
    log_info "Setting up automatic certificate renewal..."
    
    # Add cron job for automatic renewal
    (crontab -l 2>/dev/null; echo "0 12 * * * /usr/bin/certbot renew --quiet") | crontab -
    
    log_info "Auto-renewal configured"
}

# Main SSL setup
install_certbot
obtain_certificate
setup_auto_renewal

log_info "SSL setup completed!"

# Make all scripts executable
chmod +x deploy/*.sh