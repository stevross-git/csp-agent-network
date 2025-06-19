#!/bin/bash
# ==============================================================================
# ENHANCED CSP SYSTEM - COMPLETE INSTALLATION PACKAGE
# ==============================================================================

# setup.sh - Main Installation Script
#!/bin/bash
set -euo pipefail

# Enhanced CSP System Installation Script
# Version: 1.0.0
# Description: Complete installation of Enhanced CSP System

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/enhanced-csp-install.log"
VERSION="1.0.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${1}" | tee -a "${LOG_FILE}"
}

log_info() {
    log "${BLUE}[INFO]${NC} ${1}"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} ${1}"
}

log_warning() {
    log "${YELLOW}[WARNING]${NC} ${1}"
}

log_error() {
    log "${RED}[ERROR]${NC} ${1}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_success "Linux OS detected"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log_success "macOS detected"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log_success "Python ${PYTHON_VERSION} is compatible"
        else
            log_error "Python 3.8+ required, found ${PYTHON_VERSION}"
            exit 1
        fi
    else
        log_error "Python 3 not found"
        exit 1
    fi
    
    # Check Docker
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        log_success "Docker ${DOCKER_VERSION} found"
    else
        log_warning "Docker not found - will install"
        INSTALL_DOCKER=true
    fi
    
    # Check Kubernetes
    if command -v kubectl &> /dev/null; then
        KUBECTL_VERSION=$(kubectl version --client --output=yaml | grep gitVersion | cut -d' ' -f4)
        log_success "kubectl ${KUBECTL_VERSION} found"
    else
        log_warning "kubectl not found - will install"
        INSTALL_KUBECTL=true
    fi
    
    # Check Helm
    if command -v helm &> /dev/null; then
        HELM_VERSION=$(helm version --short)
        log_success "Helm ${HELM_VERSION} found"
    else
        log_warning "Helm not found - will install"
        INSTALL_HELM=true
    fi
}

# Install Docker
install_docker() {
    if [[ "${INSTALL_DOCKER:-false}" == "true" ]]; then
        log_info "Installing Docker..."
        
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Install Docker on Linux
            curl -fsSL https://get.docker.com -o get-docker.sh
            sh get-docker.sh
            sudo usermod -aG docker $USER
            rm get-docker.sh
            log_success "Docker installed successfully"
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            log_info "Please install Docker Desktop for Mac from https://docker.com/products/docker-desktop"
            exit 1
        fi
    fi
}

# Install kubectl
install_kubectl() {
    if [[ "${INSTALL_KUBECTL:-false}" == "true" ]]; then
        log_info "Installing kubectl..."
        
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
            sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
            rm kubectl
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/darwin/amd64/kubectl"
            chmod +x ./kubectl
            sudo mv ./kubectl /usr/local/bin/kubectl
        fi
        
        log_success "kubectl installed successfully"
    fi
}

# Install Helm
install_helm() {
    if [[ "${INSTALL_HELM:-false}" == "true" ]]; then
        log_info "Installing Helm..."
        
        curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
        
        log_success "Helm installed successfully"
    fi
}

# Create Python virtual environment
create_venv() {
    log_info "Creating Python virtual environment..."
    
    python3 -m venv venv
    source venv/bin/activate
    
    log_success "Virtual environment created"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install core dependencies
    pip install -r requirements.txt
    
    # Install optional dependencies
    pip install -r requirements-dev.txt
    
    # Install Enhanced CSP SDK
    pip install -e .
    
    log_success "Python dependencies installed"
}

# Setup databases
setup_databases() {
    log_info "Setting up databases..."
    
    # Start Redis container
    docker run -d \
        --name enhanced-csp-redis \
        -p 6379:6379 \
        redis:7-alpine \
        redis-server --requirepass enhanced_csp_redis_pass
    
    # Start PostgreSQL container
    docker run -d \
        --name enhanced-csp-postgres \
        -p 5432:5432 \
        -e POSTGRES_DB=enhanced_csp \
        -e POSTGRES_USER=csp_user \
        -e POSTGRES_PASSWORD=csp_password \
        postgres:15-alpine
    
    # Wait for databases to be ready
    sleep 10
    
    # Run database migrations
    source venv/bin/activate
    python -m enhanced_csp.database.migrate
    
    log_success "Databases setup complete"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring stack..."
    
    # Create monitoring namespace
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
    
    # Install Prometheus
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    helm install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --values config/prometheus-values.yaml
    
    # Install Grafana dashboards
    kubectl apply -f config/grafana-dashboards/
    
    log_success "Monitoring stack deployed"
}

# Validate installation
validate_installation() {
    log_info "Validating installation..."
    
    source venv/bin/activate
    
    # Run health checks
    python -m enhanced_csp.health_check
    
    # Run basic tests
    python -m pytest tests/integration/ -v
    
    log_success "Installation validation complete"
}

# Print post-installation instructions
print_instructions() {
    log_success "Enhanced CSP System installation complete!"
    
    cat << EOF

🎉 Installation Summary:
========================

✅ Enhanced CSP System v${VERSION} installed successfully
✅ All dependencies installed
✅ Databases configured
✅ Monitoring stack deployed

🚀 Quick Start:
===============

1. Activate the virtual environment:
   source venv/bin/activate

2. Start the Enhanced CSP system:
   python -m enhanced_csp.main

3. Access the web interface:
   http://localhost:8000

4. View monitoring dashboards:
   kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
   Open http://localhost:3000 (admin/prom-operator)

📚 Documentation:
=================

• API Documentation: http://localhost:8000/docs
• Tutorial: ./tutorials/getting-started/
• Examples: ./examples/
• Full docs: ./docs/

🔧 Configuration:
=================

• Main config: ./config/app.yaml
• Database config: ./config/database.yaml
• Monitoring config: ./config/monitoring.yaml

🆘 Support:
===========

• Issues: https://github.com/enhanced-csp/enhanced-csp/issues
• Documentation: https://docs.enhanced-csp.com
• Community: https://community.enhanced-csp.com

EOF
}

# Main installation function
main() {
    log_info "Starting Enhanced CSP System installation..."
    log_info "Installation log: ${LOG_FILE}"
    
    check_root
    check_requirements
    install_docker
    install_kubectl
    install_helm
    create_venv
    install_python_deps
    setup_databases
    setup_monitoring
    validate_installation
    print_instructions
    
    log_success "Installation completed successfully!"
}

# Run main function
main "$@"

# ==============================================================================
# requirements.txt - Python Dependencies
# ==============================================================================

# Core Dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1
psycopg2-binary==2.9.7
httpx==0.25.1
aiohttp==3.9.0
websockets==12.0

# Enhanced CSP Core
numpy==1.24.3
scipy==1.11.4
networkx==3.1
asyncio-mqtt==0.13.0

# Consciousness and AI
transformers==4.35.2
torch==2.1.0
sentence-transformers==2.2.2
scikit-learn==1.3.2
openai==1.3.5

# Quantum Computing
qiskit==0.44.2
cirq==1.2.0
pennylane==0.33.1
numpy-stl==3.0.1

# Neural Networks
tensorflow==2.13.0
keras==2.13.1
pytorch-lightning==2.1.2

# Monitoring and Observability
prometheus-client==0.19.0
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0
opentelemetry-instrumentation-fastapi==0.42b0
jaeger-client==4.8.0

# Security
cryptography==41.0.7
bcrypt==4.0.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# Configuration and Utilities
pyyaml==6.0.1
python-dotenv==1.0.0
click==8.1.7
rich==13.7.0
typer==0.9.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0
httpx==0.25.1
factory-boy==3.3.0

# ==============================================================================
# requirements-dev.txt - Development Dependencies
# ==============================================================================

# Development tools
black==23.10.1
isort==5.12.0
flake8==6.1.0
mypy==1.7.0
pre-commit==3.5.0

# Documentation
sphinx==7.2.6
sphinx-rtd-theme==1.3.0
mkdocs==1.5.3
mkdocs-material==9.4.8

# Testing and Coverage
pytest-cov==4.1.0
pytest-benchmark==4.0.0
pytest-xdist==3.5.0
locust==2.17.0

# Debugging and Profiling
ipdb==0.13.13
memory-profiler==0.61.0
line-profiler==4.1.1
py-spy==0.3.14

# Database tools
pgcli==4.0.1
redis-cli==3.5.3

# Kubernetes and Docker tools
kubernetes==28.1.0
docker==6.1.3

# API and Schema tools
openapi-generator-cli==7.1.0
swagger-codegen-cli==3.0.46

# ==============================================================================
# Makefile - Build and Development Commands
# ==============================================================================

# Makefile for Enhanced CSP System

.PHONY: help install test build deploy clean docs

# Default target
help:
	@echo "Enhanced CSP System - Make Commands"
	@echo "=================================="
	@echo ""
	@echo "Installation:"
	@echo "  install          Install the system and dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance benchmarks"
	@echo "  lint             Run code linting"
	@echo "  format           Format code with black and isort"
	@echo ""
	@echo "Building:"
	@echo "  build            Build Docker images"
	@echo "  build-all        Build all components"
	@echo "  push             Push images to registry"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-local     Deploy locally with docker-compose"
	@echo "  deploy-k8s       Deploy to Kubernetes cluster"
	@echo "  deploy-prod      Deploy to production"
	@echo ""
	@echo "Documentation:"
	@echo "  docs             Generate documentation"
	@echo "  docs-serve       Serve documentation locally"
	@echo ""
	@echo "Utilities:"
	@echo "  clean            Clean build artifacts"
	@echo "  logs             Show application logs"
	@echo "  shell            Open development shell"

# Installation targets
install:
	@echo "Installing Enhanced CSP System..."
	./scripts/setup.sh

install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements-dev.txt
	pre-commit install

# Testing targets
test:
	@echo "Running all tests..."
	python -m pytest tests/ -v --cov=enhanced_csp --cov-report=html

test-unit:
	@echo "Running unit tests..."
	python -m pytest tests/unit/ -v

test-integration:
	@echo "Running integration tests..."
	python -m pytest tests/integration/ -v

test-performance:
	@echo "Running performance benchmarks..."
	python -m pytest tests/performance/ --benchmark-only

# Code quality targets
lint:
	@echo "Running code linting..."
	flake8 enhanced_csp/
	mypy enhanced_csp/

format:
	@echo "Formatting code..."
	black enhanced_csp/ tests/
	isort enhanced_csp/ tests/

# Build targets
build:
	@echo "Building Docker images..."
	docker build -f docker/Dockerfile.core -t enhanced-csp/core:latest .
	docker build -f docker/Dockerfile.consciousness -t enhanced-csp/consciousness:latest .
	docker build -f docker/Dockerfile.quantum -t enhanced-csp/quantum:latest .
	docker build -f docker/Dockerfile.neural-mesh -t enhanced-csp/neural-mesh:latest .

build-all: build
	@echo "Building all components..."
	docker build -f docker/Dockerfile.api-gateway -t enhanced-csp/api-gateway:latest .
	docker build -f docker/Dockerfile.web-ui -t enhanced-csp/web-ui:latest .

push:
	@echo "Pushing images to registry..."
	docker push enhanced-csp/core:latest
	docker push enhanced-csp/consciousness:latest
	docker push enhanced-csp/quantum:latest
	docker push enhanced-csp/neural-mesh:latest

# Deployment targets
deploy-local:
	@echo "Deploying locally..."
	docker-compose up -d

deploy-k8s:
	@echo "Deploying to Kubernetes..."
	kubectl apply -f k8s/

deploy-prod:
	@echo "Deploying to production..."
	helm upgrade --install enhanced-csp ./helm/enhanced-csp \
		--namespace enhanced-csp \
		--create-namespace \
		--values helm/values-production.yaml

# Documentation targets
docs:
	@echo "Generating documentation..."
	cd docs && make html

docs-serve:
	@echo "Serving documentation..."
	cd docs && python -m http.server 8080

# Utility targets
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
	docker system prune -f

logs:
	@echo "Showing application logs..."
	kubectl logs -f deployment/enhanced-csp-core -n enhanced-csp

shell:
	@echo "Opening development shell..."
	docker run -it --rm \
		-v $(PWD):/app \
		-w /app \
		python:3.11-slim \
		bash

# Database management
db-migrate:
	@echo "Running database migrations..."
	python -m enhanced_csp.database.migrate

db-seed:
	@echo "Seeding database with test data..."
	python -m enhanced_csp.database.seed

db-reset:
	@echo "Resetting database..."
	python -m enhanced_csp.database.reset

# Development server
dev:
	@echo "Starting development server..."
	python -m uvicorn enhanced_csp.main:app --reload --host 0.0.0.0 --port 8000

# ==============================================================================
# docker-compose.yml - Local Development Environment
# ==============================================================================

version: '3.8'

services:
  # Enhanced CSP Core
  enhanced-csp-core:
    build:
      context: .
      dockerfile: docker/Dockerfile.core
    ports:
      - "8000:8000"
    environment:
      - CSP_MODE=development
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://csp_user:csp_password@postgres:5432/enhanced_csp
      - CONSCIOUSNESS_ENABLED=true
      - QUANTUM_ENABLED=true
      - NEURAL_MESH_ENABLED=true
    depends_on:
      - redis
      - postgres
      - consciousness-manager
      - quantum-manager
      - neural-mesh-manager
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    networks:
      - enhanced-csp-network

  # Consciousness Manager
  consciousness-manager:
    build:
      context: .
      dockerfile: docker/Dockerfile.consciousness
    ports:
      - "8001:8001"
    environment:
      - CONSCIOUSNESS_LEVEL=0.9
      - SYNC_INTERVAL=100ms
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    networks:
      - enhanced-csp-network

  # Quantum Manager
  quantum-manager:
    build:
      context: .
      dockerfile: docker/Dockerfile.quantum
    ports:
      - "8002:8002"
    environment:
      - QUANTUM_BACKEND=qiskit
      - ENTANGLEMENT_PAIRS_LIMIT=1000
      - COHERENCE_TIME=1000ms
    networks:
      - enhanced-csp-network

  # Neural Mesh Manager
  neural-mesh-manager:
    build:
      context: .
      dockerfile: docker/Dockerfile.neural-mesh
    ports:
      - "8003:8003"
    environment:
      - MESH_OPTIMIZATION_INTERVAL=5s
      - MAX_AGENTS_PER_MESH=100
    networks:
      - enhanced-csp-network

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --requirepass enhanced_csp_redis_pass
    volumes:
      - redis-data:/data
    networks:
      - enhanced-csp-network

  # PostgreSQL
  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=enhanced_csp
      - POSTGRES_USER=csp_user
      - POSTGRES_PASSWORD=csp_password
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - enhanced-csp-network

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    networks:
      - enhanced-csp-network

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana-data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./config/grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - enhanced-csp-network

  # Jaeger
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - enhanced-csp-network

volumes:
  redis-data:
  postgres-data:
  prometheus-data:
  grafana-data:

networks:
  enhanced-csp-network:
    driver: bridge

# ==============================================================================
# quick-start.sh - Quick Start Script
# ==============================================================================

#!/bin/bash

# Enhanced CSP System - Quick Start Script
# This script gets you up and running in minutes

set -euo pipefail

echo "🚀 Enhanced CSP System - Quick Start"
echo "===================================="
echo ""

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "✅ Docker is running"

# Clone repository if not already cloned
if [ ! -f "docker-compose.yml" ]; then
    echo "📥 Cloning Enhanced CSP repository..."
    git clone https://github.com/enhanced-csp/enhanced-csp.git
    cd enhanced-csp
fi

# Start the system
echo "🐳 Starting Enhanced CSP System with Docker Compose..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

services=("enhanced-csp-core:8000" "consciousness-manager:8001" "quantum-manager:8002" "neural-mesh-manager:8003")

for service in "${services[@]}"; do
    service_name=${service%:*}
    port=${service#*:}
    
    if curl -s http://localhost:$port/health >/dev/null; then
        echo "✅ $service_name is healthy"
    else
        echo "⚠️  $service_name is not responding"
    fi
done

echo ""
echo "🎉 Enhanced CSP System is running!"
echo ""
echo "📊 Access points:"
echo "• Web UI: http://localhost:8000"
echo "• API Documentation: http://localhost:8000/docs"
echo "• Prometheus: http://localhost:9090"
echo "• Grafana: http://localhost:3000 (admin/admin123)"
echo "• Jaeger: http://localhost:16686"
echo ""
echo "🎓 Next steps:"
echo "1. Open the web interface: http://localhost:8000"
echo "2. Try the interactive tutorial"
echo "3. Explore the API documentation"
echo "4. Check out the examples in ./examples/"
echo ""
echo "To stop the system: docker-compose down"
echo "To view logs: docker-compose logs -f"
echo ""
echo "🌟 Welcome to the future of AI communication!"

# ==============================================================================
# Environment Configuration Files
# ==============================================================================

# .env.example - Environment Variables Template
# Copy this to .env and customize for your environment

# ==============================================================================
# Enhanced CSP System Environment Configuration
# ==============================================================================

# Application Settings
CSP_MODE=production
CSP_VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security Settings
SECRET_KEY=your-super-secret-key-change-this-in-production
API_KEY_HEADER=X-API-Key
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=enhanced_csp
POSTGRES_USER=csp_user
POSTGRES_PASSWORD=csp_password
POSTGRES_URL=postgresql://csp_user:csp_password@localhost:5432/enhanced_csp

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=enhanced_csp_redis_pass
REDIS_DB=0
REDIS_URL=redis://:enhanced_csp_redis_pass@localhost:6379/0

# Enhanced CSP Features
CONSCIOUSNESS_ENABLED=true
CONSCIOUSNESS_LEVEL=0.9
CONSCIOUSNESS_SYNC_INTERVAL=100ms

QUANTUM_ENABLED=true
QUANTUM_BACKEND=qiskit
QUANTUM_ENTANGLEMENT_LIMIT=10000
QUANTUM_COHERENCE_TIME=1000ms
QUANTUM_FIDELITY_THRESHOLD=0.85

NEURAL_MESH_ENABLED=true
NEURAL_MESH_OPTIMIZATION_INTERVAL=5s
NEURAL_MESH_MAX_AGENTS=1000
NEURAL_MESH_LEARNING_RATE=0.01

# Performance Settings
MAX_CONCURRENT_PROCESSES=10000
EVENT_QUEUE_SIZE=100000
WORKER_THREADS=32
CONNECTION_POOL_SIZE=100

# Monitoring and Observability
PROMETHEUS_ENABLED=true
JAEGER_ENABLED=true
METRICS_PORT=9090
TRACING_ENDPOINT=http://localhost:14268/api/traces

# Cloud Provider Settings (for production)
CLOUD_PROVIDER=aws  # aws, gcp, azure
AWS_REGION=us-west-2
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key

# Kubernetes Settings
KUBERNETES_NAMESPACE=enhanced-csp
KUBERNETES_SERVICE_ACCOUNT=enhanced-csp-service-account

# TLS/SSL Settings
TLS_ENABLED=true
TLS_CERT_PATH=/certs/tls.crt
TLS_KEY_PATH=/certs/tls.key

# External Integrations
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Development Settings (dev/staging only)
ENABLE_CORS=true
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
ENABLE_SWAGGER_UI=true
ENABLE_REDOC=true