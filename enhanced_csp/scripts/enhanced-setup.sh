#!/bin/bash
# =============================================================================
# ENHANCED DOCKER MANAGEMENT SCRIPTS FOR CSP PROJECT
# =============================================================================

# scripts/enhanced-setup.sh
#!/bin/bash

echo "🚀 Enhanced CSP Docker Environment Setup"
echo "========================================"

# Detect OS for better compatibility
OS_TYPE=$(uname -s)
ARCH_TYPE=$(uname -m)

# Configuration
PROJECT_NAME="enhanced-csp"
COMPOSE_PROJECT_NAME="csp"
DEFAULT_ENVIRONMENT="development"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_debug() { echo -e "${PURPLE}[DEBUG]${NC} $1"; }

# Detect docker-compose file with priority order
detect_compose_file() {
    local possible_paths=(
        "docker-compose.yml"
        "deployment/docker/database/docker-compose.yml"
        "deployment/docker/docker-compose.yml"
        "docker/docker-compose.yml"
        "infrastructure/docker-compose.yml"
    )
    
    for path in "${possible_paths[@]}"; do
        if [ -f "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    
    return 1
}

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check available memory
    if [[ "$OS_TYPE" == "Linux" ]]; then
        TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$TOTAL_MEM" -lt 8 ]; then
            log_warning "System has ${TOTAL_MEM}GB RAM. Recommended: 8GB+ for full stack"
        fi
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        if ! docker compose version &> /dev/null; then
            log_error "Docker Compose is not available. Please install Docker Compose."
            exit 1
        else
            log_info "Using Docker Compose V2 (docker compose)"
            COMPOSE_CMD="docker compose"
        fi
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G.*//')
    if [ "$AVAILABLE_SPACE" -lt 10 ]; then
        log_warning "Available disk space: ${AVAILABLE_SPACE}GB. Recommended: 10GB+"
    fi
    
    log_success "System requirements check completed"
}

# Create enhanced directory structure
create_directory_structure() {
    log_info "Creating enhanced directory structure..."
    
    # Database directories
    mkdir -p database/{init,backups,scripts}
    mkdir -p database/{postgresql,redis,mongodb,elasticsearch,influxdb}
    mkdir -p database/ai_models/{init,models,checkpoints}
    mkdir -p database/vector/{chroma,qdrant,weaviate,milvus,pgvector}
    mkdir -p database/admin/{pgadmin,redis-insight,mongo-express}
    
    # Monitoring and logging
    mkdir -p monitoring/{prometheus,grafana,alertmanager,loki}
    mkdir -p monitoring/grafana/{dashboards,datasources,plugins}
    mkdir -p logs/{application,database,monitoring,audit}
    
    # Configuration directories
    mkdir -p config/{environments,secrets,certificates}
    mkdir -p config/environments/{development,staging,production}
    
    # Data and volumes
    mkdir -p data/{uploads,exports,temp,cache}
    mkdir -p volumes/{postgres,redis,mongodb,elasticsearch,influxdb}
    
    # Scripts and utilities
    mkdir -p scripts/{maintenance,migration,testing,deployment}
    
    log_success "Directory structure created"
}

# Create enhanced environment configuration
create_enhanced_env() {
    local env_file="${1:-.env.docker}"
    
    if [ -f "$env_file" ]; then
        log_info "Environment file $env_file already exists, creating backup"
        cp "$env_file" "${env_file}.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    log_info "Creating enhanced environment configuration..."
    
    cat > "$env_file" << 'EOL'
# =============================================================================
# ENHANCED CSP DOCKER ENVIRONMENT CONFIGURATION
# =============================================================================

# Project Configuration
COMPOSE_PROJECT_NAME=csp
CSP_ENVIRONMENT=development
CSP_VERSION=latest
CSP_DEBUG=true

# =============================================================================
# DATABASE CONFIGURATIONS
# =============================================================================

# PostgreSQL Main Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=csp_visual_designer
POSTGRES_USER=csp_user
POSTGRES_PASSWORD=csp_secure_pass_2024!
POSTGRES_MAX_CONNECTIONS=200
POSTGRES_SHARED_BUFFERS=256MB

# PostgreSQL AI Models Database
AI_MODELS_POSTGRES_HOST=postgres_ai_models
AI_MODELS_POSTGRES_PORT=5432
AI_MODELS_POSTGRES_DB=ai_models_db
AI_MODELS_POSTGRES_USER=ai_models_user
AI_MODELS_POSTGRES_PASSWORD=ai_models_secure_pass_2024!

# PostgreSQL Vector Database
VECTOR_POSTGRES_HOST=postgres_vector
VECTOR_POSTGRES_PORT=5432
VECTOR_POSTGRES_DB=vector_db
VECTOR_POSTGRES_USER=vector_user
VECTOR_POSTGRES_PASSWORD=vector_secure_pass_2024!

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=redis_secure_pass_2024!
REDIS_MAX_MEMORY=512mb
REDIS_MAX_MEMORY_POLICY=allkeys-lru

# MongoDB Configuration
MONGODB_HOST=mongodb
MONGODB_PORT=27017
MONGODB_USERNAME=csp_mongo_user
MONGODB_PASSWORD=mongo_secure_pass_2024!
MONGODB_DATABASE=csp_documents
MONGODB_AUTH_SOURCE=admin

# InfluxDB Configuration
INFLUXDB_HOST=influxdb
INFLUXDB_PORT=8086
INFLUXDB_ORG=csp_org
INFLUXDB_BUCKET=csp_metrics
INFLUXDB_USERNAME=csp_admin
INFLUXDB_PASSWORD=influx_secure_pass_2024!
INFLUXDB_TOKEN=csp_influx_token_change_in_production

# Elasticsearch Configuration
ELASTICSEARCH_HOST=elasticsearch
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=elastic_secure_pass_2024!

# =============================================================================
# VECTOR DATABASE CONFIGURATIONS
# =============================================================================

# Chroma Vector Database
CHROMA_HOST=chroma
CHROMA_PORT=8200
CHROMA_PERSIST_DIRECTORY=/chroma/chroma
CHROMA_ALLOW_RESET=true

# Qdrant Vector Database
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_API_KEY=qdrant_secure_key_2024!

# Weaviate Vector Database
WEAVIATE_HOST=weaviate
WEAVIATE_PORT=8080
WEAVIATE_AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
WEAVIATE_PERSISTENCE_DATA_PATH=/var/lib/weaviate

# Milvus Vector Database
MILVUS_HOST=milvus-standalone
MILVUS_PORT=19530
ETCD_ENDPOINTS=etcd:2379
MINIO_ADDRESS=minio:9000

# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_DEBUG=true
API_RELOAD=true

# Security Configuration
SECRET_KEY=your_super_secret_key_change_in_production_2024!
JWT_SECRET_KEY=your_jwt_secret_key_change_in_production_2024!
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000","http://127.0.0.1:3000"]
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=["*"]
CORS_ALLOW_HEADERS=["*"]

# =============================================================================
# EXTERNAL API CONFIGURATIONS
# =============================================================================

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORG_ID=your_openai_org_id_here
OPENAI_MODEL_DEFAULT=gpt-4
OPENAI_MAX_TOKENS=4000
OPENAI_TEMPERATURE=0.7

# Anthropic API
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL_DEFAULT=claude-3-sonnet-20240229

# Google API
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_PROJECT_ID=your_google_project_id

# Hugging Face
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
HUGGINGFACE_CACHE_DIR=/app/cache/huggingface

# =============================================================================
# MONITORING & OBSERVABILITY
# =============================================================================

# Prometheus
PROMETHEUS_PORT=9090
PROMETHEUS_RETENTION_TIME=30d
PROMETHEUS_STORAGE_TSDB_PATH=/prometheus

# Grafana
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=grafana_secure_pass_2024!
GRAFANA_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE_MAX_SIZE=100MB
LOG_FILE_BACKUP_COUNT=5

# =============================================================================
# ADMIN TOOL CONFIGURATIONS
# =============================================================================

# pgAdmin
PGADMIN_DEFAULT_EMAIL=admin@csp.local
PGADMIN_DEFAULT_PASSWORD=pgadmin_secure_pass_2024!
PGADMIN_PORT=5050

# Redis Insight
REDIS_INSIGHT_PORT=8001

# Mongo Express
MONGO_EXPRESS_PORT=8081
MONGO_EXPRESS_USERNAME=mongo_admin
MONGO_EXPRESS_PASSWORD=mongo_admin_secure_pass_2024!

# =============================================================================
# PERFORMANCE & SCALING
# =============================================================================

# Connection Pools
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600

# Cache Settings
CACHE_TTL=3600
CACHE_MAX_SIZE=1000
CACHE_BACKEND=redis

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BURST=10

# =============================================================================
# BACKUP & MAINTENANCE
# =============================================================================

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESSION=true
BACKUP_ENCRYPTION=false

# Maintenance
MAINTENANCE_MODE=false
HEALTH_CHECK_INTERVAL=30s
HEALTH_CHECK_TIMEOUT=10s
HEALTH_CHECK_RETRIES=3

# =============================================================================
# DEVELOPMENT SETTINGS
# =============================================================================

# Development Flags
DEV_MODE=true
DEV_RELOAD=true
DEV_DEBUG_SQL=false
DEV_MOCK_EXTERNAL_APIS=false

# Testing
TEST_DATABASE_URL=postgresql://csp_user:csp_secure_pass_2024!@postgres:5432/csp_test
TEST_REDIS_URL=redis://:redis_secure_pass_2024!@redis:6379/1

# =============================================================================
# FEATURE FLAGS
# =============================================================================

# Features
FEATURE_AI_INTEGRATION=true
FEATURE_REAL_TIME_COLLABORATION=true
FEATURE_VECTOR_SEARCH=true
FEATURE_ADVANCED_ANALYTICS=true
FEATURE_BLOCKCHAIN_INTEGRATION=false
FEATURE_QUANTUM_SIMULATION=false

# =============================================================================
# WARNING: CHANGE ALL PASSWORDS IN PRODUCTION!
# =============================================================================
EOL

    log_success "Enhanced environment configuration created: $env_file"
    log_warning "⚠️  IMPORTANT: Update all passwords and API keys before production use!"
}

# Create advanced monitoring configuration
create_monitoring_config() {
    log_info "Creating advanced monitoring configuration..."
    
    # Prometheus configuration
    cat > monitoring/prometheus/prometheus.yml << 'EOL'
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'csp-monitor'
    environment: '${CSP_ENVIRONMENT:-development}'

rule_files:
  - "alert_rules.yml"

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

  # CSP API
  - job_name: 'csp-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # PostgreSQL databases
  - job_name: 'postgres-main'
    static_configs:
      - targets: ['postgres:5432']
    metrics_path: '/metrics'

  - job_name: 'postgres-ai-models'
    static_configs:
      - targets: ['postgres_ai_models:5432']

  - job_name: 'postgres-vector'
    static_configs:
      - targets: ['postgres_vector:5432']

  # Redis
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  # MongoDB
  - job_name: 'mongodb'
    static_configs:
      - targets: ['mongodb:27017']

  # Vector Databases
  - job_name: 'chroma'
    static_configs:
      - targets: ['chroma:8200']
    metrics_path: '/api/v1/heartbeat'

  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant:6333']
    metrics_path: '/metrics'

  - job_name: 'weaviate'
    static_configs:
      - targets: ['weaviate:8080']
    metrics_path: '/v1/meta'

  # Node Exporter for system metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  # cAdvisor for container metrics
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
EOL

    # Grafana datasources
    cat > monitoring/grafana/datasources/prometheus.yml << 'EOL'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    
  - name: InfluxDB
    type: influxdb
    access: proxy
    url: http://influxdb:8086
    database: csp_metrics
    user: csp_admin
    secureJsonData:
      password: influx_secure_pass_2024!

  - name: Elasticsearch
    type: elasticsearch
    access: proxy
    url: http://elasticsearch:9200
    index: csp-logs-*
    timeField: "@timestamp"
EOL

    log_success "Monitoring configuration created"
}

# Enhanced service management functions
start_core_services() {
    local compose_file="$1"
    log_info "Starting core services..."
    
    $COMPOSE_CMD -f "$compose_file" up -d postgres redis
    wait_for_service "postgres" "pg_isready -U csp_user -d csp_visual_designer"
    wait_for_service "redis" "redis-cli ping"
}

start_database_services() {
    local compose_file="$1"
    log_info "Starting all database services..."
    
    $COMPOSE_CMD -f "$compose_file" up -d postgres postgres_ai_models postgres_vector mongodb influxdb elasticsearch
    
    # Wait for databases
    wait_for_service "postgres_ai_models" "pg_isready -U ai_models_user -d ai_models_db"
    wait_for_service "postgres_vector" "pg_isready -U vector_user -d vector_db"
}

start_vector_databases() {
    local compose_file="$1"
    log_info "Starting vector databases..."
    
    $COMPOSE_CMD -f "$compose_file" up -d chroma qdrant weaviate
    
    # Optional: Start Milvus if profile enabled
    $COMPOSE_CMD -f "$compose_file" --profile milvus up -d etcd minio milvus-standalone || log_info "Milvus profile not enabled"
}

wait_for_service() {
    local service_name="$1"
    local health_check="$2"
    local max_attempts=30
    local attempt=1
    
    log_info "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if $COMPOSE_CMD exec -T "$service_name" $health_check >/dev/null 2>&1; then
            log_success "$service_name is ready"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    log_warning "$service_name not ready after $max_attempts attempts"
    return 1
}

# Main setup function
main() {
    echo ""
    log_info "Starting Enhanced CSP Docker Environment Setup"
    echo "=============================================="
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment|-e)
                DEFAULT_ENVIRONMENT="$2"
                shift 2
                ;;
            --minimal)
                MINIMAL_SETUP=true
                shift
                ;;
            --no-monitoring)
                NO_MONITORING=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  -e, --environment    Environment (development/staging/production)"
                echo "  --minimal           Minimal setup (core services only)"
                echo "  --no-monitoring     Skip monitoring setup"
                echo "  -h, --help          Show this help"
                exit 0
                ;;
            *)
                log_warning "Unknown option: $1"
                shift
                ;;
        esac
    done
    
    # Run setup steps
    check_system_requirements
    
    # Detect Docker Compose file
    DOCKER_COMPOSE_FILE=$(detect_compose_file)
    if [ $? -ne 0 ]; then
        log_error "Could not find docker-compose.yml file in expected locations"
    fi
    
    log_info "Using Docker Compose file: $DOCKER_COMPOSE_FILE"
    echo "export DOCKER_COMPOSE_FILE='$DOCKER_COMPOSE_FILE'" > .docker-compose-path
    echo "export COMPOSE_CMD='$COMPOSE_CMD'" >> .docker-compose-path
    
    create_directory_structure
    create_enhanced_env ".env.docker"
    
    if [[ "$NO_MONITORING" != "true" ]]; then
        create_monitoring_config
    fi
    
    # Make scripts executable
    chmod +x scripts/*.sh 2>/dev/null || true
    
    echo ""
    log_success "Enhanced Docker environment setup completed!"
    echo ""
    echo "📍 Configuration Summary:"
    echo "  - Docker Compose: $DOCKER_COMPOSE_FILE"
    echo "  - Environment: $DEFAULT_ENVIRONMENT"
    echo "  - Compose Command: $COMPOSE_CMD"
    echo ""
    echo "🚀 Next Steps:"
    echo "  1. Review and update .env.docker with your API keys"
    echo "  2. Run: ./scripts/start-databases.sh"
    echo "  3. Run: ./scripts/start-all.sh"
    echo "  4. Access pgAdmin: http://localhost:5050"
    echo "  5. Access Grafana: http://localhost:3000"
    echo ""
    echo "🔧 Available Scripts:"
    echo "  - ./scripts/start-databases.sh"
    echo "  - ./scripts/start-all.sh"
    echo "  - ./scripts/stop-all.sh"
    echo "  - ./scripts/backup-all.sh"
    echo "  - ./scripts/status.sh"
    echo "  - ./scripts/logs.sh"
    echo "  - ./scripts/cleanup.sh"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi