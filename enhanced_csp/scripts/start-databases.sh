#!/bin/bash
# =============================================================================
# COMPLETE DOCKER MANAGEMENT SCRIPTS SUITE FOR ENHANCED CSP PROJECT
# =============================================================================

# scripts/start-databases.sh
#!/bin/bash

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh" 2>/dev/null || {
    # Fallback if common.sh doesn't exist
    log_info() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
    log_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $1"; }
    log_warning() { echo -e "\033[1;33m[WARNING]\033[0m $1"; }
    log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; exit 1; }
    
    load_compose_config() {
        if [[ -f .docker-compose-path ]]; then
            source .docker-compose-path
        else
            if [[ -f "docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="docker-compose.yml"
            elif [[ -f "deployment/docker/database/docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="deployment/docker/database/docker-compose.yml"
            elif [[ -f "deployment/docker/docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
            else
                log_error "Could not find docker-compose.yml file"
            fi
            
            if command -v docker-compose &> /dev/null; then
                export COMPOSE_CMD="docker-compose"
            else
                export COMPOSE_CMD="docker compose"
            fi
        fi
    }
}

echo "🗄️ Starting Enhanced CSP Databases"
echo "=================================="

# Parse command line arguments
PROFILE=""
MINIMAL=false
NO_VECTOR=false
NO_MONITORING=false
DETACHED=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --profile|-p)
            PROFILE="$2"
            shift 2
            ;;
        --minimal|-m)
            MINIMAL=true
            shift
            ;;
        --no-vector)
            NO_VECTOR=true
            shift
            ;;
        --no-monitoring)
            NO_MONITORING=true
            shift
            ;;
        --foreground|-f)
            DETACHED=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -p, --profile PROFILE   Use specific profile (milvus, monitoring, etc.)"
            echo "  -m, --minimal          Start only core databases (PostgreSQL, Redis)"
            echo "  --no-vector            Skip vector databases"
            echo "  --no-monitoring        Skip monitoring services"
            echo "  -f, --foreground       Run in foreground (don't detach)"
            echo "  -h, --help             Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                     # Start all databases"
            echo "  $0 --minimal           # Start only core databases"
            echo "  $0 --profile milvus    # Include Milvus vector database"
            exit 0
            ;;
        *)
            log_warning "Unknown option: $1"
            shift
            ;;
    esac
done

# Load configuration
load_compose_config

log_info "Using Docker Compose file: $DOCKER_COMPOSE_FILE"
log_info "Using Compose command: $COMPOSE_CMD"

# Load environment variables if available
if [[ -f .env.docker ]]; then
    log_info "Loading environment variables from .env.docker"
    export $(cat .env.docker | grep -v '^#' | grep -v '^$' | xargs)
fi

# Determine compose flags
COMPOSE_FLAGS="-f $DOCKER_COMPOSE_FILE"
if [[ "$DETACHED" == "true" ]]; then
    COMPOSE_FLAGS="$COMPOSE_FLAGS -d"
fi

# Add profile if specified
if [[ -n "$PROFILE" ]]; then
    COMPOSE_FLAGS="--profile $PROFILE $COMPOSE_FLAGS"
fi

# Function to wait for service health
wait_for_health() {
    local service="$1"
    local health_command="$2"
    local max_attempts="${3:-30}"
    local attempt=1
    
    log_info "Waiting for $service to be healthy..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" exec -T "$service" $health_command >/dev/null 2>&1; then
            log_success "$service is healthy"
            return 0
        fi
        
        if [[ $((attempt % 5)) -eq 0 ]]; then
            log_info "Still waiting for $service... (attempt $attempt/$max_attempts)"
        fi
        sleep 2
        ((attempt++))
    done
    
    log_warning "$service health check failed after $max_attempts attempts"
    return 1
}

# Start core databases first
start_core_databases() {
    log_info "🚀 Starting core databases (PostgreSQL, Redis)..."
    
    $COMPOSE_CMD $COMPOSE_FLAGS up postgres redis postgres_ai_models postgres_vector
    
    # Wait for core databases to be ready
    sleep 10
    
    wait_for_health "postgres" "pg_isready -U csp_user -d csp_visual_designer"
    wait_for_health "redis" "redis-cli ping"
    wait_for_health "postgres_ai_models" "pg_isready -U ai_models_user -d ai_models_db"
    wait_for_health "postgres_vector" "pg_isready -U vector_user -d vector_db"
}

# Start vector databases
start_vector_databases() {
    if [[ "$NO_VECTOR" == "true" ]]; then
        log_info "Skipping vector databases (--no-vector specified)"
        return
    fi
    
    log_info "🧠 Starting vector databases..."
    
    # Start main vector databases
    $COMPOSE_CMD $COMPOSE_FLAGS up chroma qdrant weaviate
    
    # Start Milvus if profile is enabled
    if [[ "$PROFILE" == "milvus" ]] || [[ "$PROFILE" == "full" ]]; then
        log_info "Starting Milvus ecosystem..."
        $COMPOSE_CMD $COMPOSE_FLAGS up etcd minio milvus-standalone
    fi
    
    # Health checks for vector databases
    sleep 15
    
    # Check Chroma
    local chroma_attempts=0
    while [[ $chroma_attempts -lt 15 ]]; do
        if curl -s http://localhost:8200/api/v1/heartbeat >/dev/null 2>&1; then
            log_success "Chroma is ready"
            break
        fi
        ((chroma_attempts++))
        sleep 2
    done
    
    # Check Qdrant
    local qdrant_attempts=0
    while [[ $qdrant_attempts -lt 15 ]]; do
        if curl -s http://localhost:6333/health >/dev/null 2>&1; then
            log_success "Qdrant is ready"
            break
        fi
        ((qdrant_attempts++))
        sleep 2
    done
    
    # Check Weaviate
    local weaviate_attempts=0
    while [[ $weaviate_attempts -lt 15 ]]; do
        if curl -s http://localhost:8080/v1/.well-known/ready >/dev/null 2>&1; then
            log_success "Weaviate is ready"
            break
        fi
        ((weaviate_attempts++))
        sleep 2
    done
}

# Start additional databases
start_additional_databases() {
    if [[ "$MINIMAL" == "true" ]]; then
        log_info "Skipping additional databases (--minimal specified)"
        return
    fi
    
    log_info "📊 Starting additional databases (MongoDB, Elasticsearch, InfluxDB)..."
    
    $COMPOSE_CMD $COMPOSE_FLAGS up mongodb elasticsearch influxdb
    
    # Wait for additional databases
    sleep 20
    
    # MongoDB health check
    wait_for_health "mongodb" "mongosh --quiet --eval 'db.adminCommand(\"ping\")'" 20
    
    # Elasticsearch health check (may take longer)
    log_info "Waiting for Elasticsearch (this may take a while)..."
    local es_attempts=0
    while [[ $es_attempts -lt 30 ]]; do
        if curl -s http://localhost:9200/_cluster/health >/dev/null 2>&1; then
            log_success "Elasticsearch is ready"
            break
        fi
        if [[ $((es_attempts % 10)) -eq 0 ]]; then
            log_info "Still waiting for Elasticsearch... (attempt $es_attempts/30)"
        fi
        ((es_attempts++))
        sleep 3
    done
    
    # InfluxDB health check
    local influx_attempts=0
    while [[ $influx_attempts -lt 15 ]]; do
        if curl -s http://localhost:8086/health >/dev/null 2>&1; then
            log_success "InfluxDB is ready"
            break
        fi
        ((influx_attempts++))
        sleep 2
    done
}

# Start admin tools
start_admin_tools() {
    if [[ "$MINIMAL" == "true" ]]; then
        log_info "Skipping admin tools (--minimal specified)"
        return
    fi
    
    log_info "🛠️ Starting admin tools..."
    
    $COMPOSE_CMD $COMPOSE_FLAGS up pgadmin redis-insight mongo-express
    
    # Give admin tools time to start
    sleep 10
}

# Start monitoring services
start_monitoring() {
    if [[ "$NO_MONITORING" == "true" ]]; then
        log_info "Skipping monitoring services (--no-monitoring specified)"
        return
    fi
    
    if [[ "$MINIMAL" == "true" ]]; then
        log_info "Skipping monitoring services (--minimal specified)"
        return
    fi
    
    log_info "📊 Starting monitoring services..."
    
    $COMPOSE_CMD $COMPOSE_FLAGS up prometheus grafana
    
    # Wait for monitoring services
    sleep 15
    
    # Check Prometheus
    local prom_attempts=0
    while [[ $prom_attempts -lt 10 ]]; do
        if curl -s http://localhost:9090/-/ready >/dev/null 2>&1; then
            log_success "Prometheus is ready"
            break
        fi
        ((prom_attempts++))
        sleep 2
    done
    
    # Check Grafana
    local grafana_attempts=0
    while [[ $grafana_attempts -lt 15 ]]; do
        if curl -s http://localhost:3000/api/health >/dev/null 2>&1; then
            log_success "Grafana is ready"
            break
        fi
        ((grafana_attempts++))
        sleep 2
    done
}

# Main execution
main() {
    log_info "Starting Enhanced CSP database services..."
    echo ""
    
    # Create necessary directories
    mkdir -p logs database/backups data
    
    # Start services in order
    start_core_databases
    start_vector_databases
    start_additional_databases
    start_admin_tools
    start_monitoring
    
    echo ""
    log_success "All database services started successfully!"
    echo ""
    
    # Display access information
    echo "📊 Database Access Points:"
    echo "=========================="
    echo "Core Databases:"
    echo "  • PostgreSQL (Main):      localhost:5432"
    echo "  • PostgreSQL (AI Models): localhost:5433" 
    echo "  • PostgreSQL (Vector):    localhost:5434"
    echo "  • Redis:                  localhost:6379"
    echo ""
    
    if [[ "$NO_VECTOR" != "true" ]]; then
        echo "🧠 Vector Databases:"
        echo "  • Chroma:    http://localhost:8200"
        echo "  • Qdrant:    http://localhost:6333"
        echo "  • Weaviate:  http://localhost:8080"
        if [[ "$PROFILE" == "milvus" ]] || [[ "$PROFILE" == "full" ]]; then
            echo "  • Milvus:    localhost:19530"
        fi
        echo ""
    fi
    
    if [[ "$MINIMAL" != "true" ]]; then
        echo "📊 Additional Databases:"
        echo "  • MongoDB:        localhost:27017"
        echo "  • Elasticsearch:  http://localhost:9200"
        echo "  • InfluxDB:       http://localhost:8086"
        echo ""
        
        echo "🛠️ Admin Tools:"
        echo "  • pgAdmin:        http://localhost:5050"
        echo "  • Redis Insight:  http://localhost:8001"
        echo "  • Mongo Express:  http://localhost:8081"
        echo ""
    fi
    
    if [[ "$NO_MONITORING" != "true" ]] && [[ "$MINIMAL" != "true" ]]; then
        echo "📈 Monitoring:"
        echo "  • Prometheus:  http://localhost:9090"
        echo "  • Grafana:     http://localhost:3000"
        echo ""
    fi
    
    echo "Next steps:"
    echo "  1. Check service status: ./scripts/status.sh"
    echo "  2. View logs: ./scripts/logs.sh"
    echo "  3. Start all services: ./scripts/start-all.sh"
}

# Execute main function
main

# =============================================================================
# scripts/start-all.sh
#!/bin/bash

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh" 2>/dev/null || {
    # Fallback functions
    log_info() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
    log_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $1"; }
    log_warning() { echo -e "\033[1;33m[WARNING]\033[0m $1"; }
    log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; exit 1; }
    
    load_compose_config() {
        if [[ -f .docker-compose-path ]]; then
            source .docker-compose-path
        else
            if [[ -f "docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="docker-compose.yml"
            elif [[ -f "deployment/docker/database/docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="deployment/docker/database/docker-compose.yml"
            elif [[ -f "deployment/docker/docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
            else
                log_error "Could not find docker-compose.yml file"
            fi
            
            if command -v docker-compose &> /dev/null; then
                export COMPOSE_CMD="docker-compose"
            else
                export COMPOSE_CMD="docker compose"
            fi
        fi
    }
}

echo "🚀 Starting Complete Enhanced CSP System"
echo "========================================"

# Parse command line arguments
PROFILE="full"
BUILD=false
PULL=false
RECREATE=false
DETACHED=true
TIMEOUT=300

while [[ $# -gt 0 ]]; do
    case $1 in
        --profile|-p)
            PROFILE="$2"
            shift 2
            ;;
        --build|-b)
            BUILD=true
            shift
            ;;
        --pull)
            PULL=true
            shift
            ;;
        --recreate|-r)
            RECREATE=true
            shift
            ;;
        --foreground|-f)
            DETACHED=false
            shift
            ;;
        --timeout|-t)
            TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -p, --profile PROFILE   Docker Compose profile (default: full)"
            echo "  -b, --build            Build images before starting"
            echo "  --pull                 Pull latest images before starting"
            echo "  -r, --recreate         Recreate containers"
            echo "  -f, --foreground       Run in foreground"
            echo "  -t, --timeout SECONDS  Startup timeout (default: 300)"
            echo "  -h, --help             Show this help"
            echo ""
            echo "Profiles:"
            echo "  minimal     - Core services only"
            echo "  monitoring  - Include monitoring stack"
            echo "  milvus      - Include Milvus vector database"
            echo "  full        - All services (default)"
            exit 0
            ;;
        *)
            log_warning "Unknown option: $1"
            shift
            ;;
    esac
done

# Load configuration
load_compose_config

log_info "Using Docker Compose file: $DOCKER_COMPOSE_FILE"
log_info "Using profile: $PROFILE"

# Load environment variables
if [[ -f .env.docker ]]; then
    log_info "Loading environment variables from .env.docker"
    export $(cat .env.docker | grep -v '^#' | grep -v '^$' | xargs)
fi

# Prepare compose command flags
COMPOSE_FLAGS="-f $DOCKER_COMPOSE_FILE"

if [[ -n "$PROFILE" && "$PROFILE" != "full" ]]; then
    COMPOSE_FLAGS="--profile $PROFILE $COMPOSE_FLAGS"
fi

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
    fi
    
    # Check available memory
    if command -v free >/dev/null 2>&1; then
        local available_mem=$(free -g | awk '/^Mem:/{print $7}')
        if [[ $available_mem -lt 4 ]]; then
            log_warning "Available memory is ${available_mem}GB. Recommended: 4GB+"
        fi
    fi
    
    # Check disk space
    local available_space=$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')
    if [[ $available_space -lt 10 ]]; then
        log_warning "Available disk space: ${available_space}GB. Recommended: 10GB+"
    fi
    
    # Create required directories
    mkdir -p logs database/backups data monitoring/prometheus monitoring/grafana
    
    log_success "Pre-flight checks completed"
}

# Pull or build images if requested
prepare_images() {
    if [[ "$PULL" == "true" ]]; then
        log_info "Pulling latest Docker images..."
        $COMPOSE_CMD $COMPOSE_FLAGS pull
    fi
    
    if [[ "$BUILD" == "true" ]]; then
        log_info "Building Docker images..."
        $COMPOSE_CMD $COMPOSE_FLAGS build
    fi
}

# Start services with proper ordering
start_all_services() {
    local up_flags=""
    
    if [[ "$DETACHED" == "true" ]]; then
        up_flags="$up_flags -d"
    fi
    
    if [[ "$RECREATE" == "true" ]]; then
        up_flags="$up_flags --force-recreate"
    fi
    
    log_info "Starting all Enhanced CSP services..."
    log_info "This may take several minutes on first run..."
    
    # Start all services
    timeout $TIMEOUT $COMPOSE_CMD $COMPOSE_FLAGS up $up_flags
    
    if [[ $? -eq 124 ]]; then
        log_warning "Startup timed out after ${TIMEOUT} seconds"
        log_info "Services may still be starting in the background"
    fi
}

# Wait for critical services
wait_for_critical_services() {
    log_info "Waiting for critical services to be ready..."
    
    # Define critical services and their health checks
    declare -A health_checks=(
        ["postgres"]="pg_isready -U csp_user -d csp_visual_designer"
        ["redis"]="redis-cli ping"
    )
    
    for service in "${!health_checks[@]}"; do
        local health_cmd="${health_checks[$service]}"
        local attempts=0
        local max_attempts=30
        
        log_info "Checking $service..."
        while [[ $attempts -lt $max_attempts ]]; do
            if $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" exec -T "$service" $health_cmd >/dev/null 2>&1; then
                log_success "$service is ready"
                break
            fi
            
            if [[ $((attempts % 10)) -eq 0 ]]; then
                log_info "Still waiting for $service... ($attempts/$max_attempts)"
            fi
            
            sleep 2
            ((attempts++))
        done
        
        if [[ $attempts -eq $max_attempts ]]; then
            log_warning "$service health check timed out"
        fi
    done
}

# Check external services
check_external_services() {
    log_info "Checking external service endpoints..."
    
    # Vector databases
    local endpoints=(
        "http://localhost:8200/api/v1/heartbeat:Chroma"
        "http://localhost:6333/health:Qdrant"
        "http://localhost:8080/v1/.well-known/ready:Weaviate"
        "http://localhost:9200/_cluster/health:Elasticsearch"
        "http://localhost:8086/health:InfluxDB"
        "http://localhost:9090/-/ready:Prometheus"
        "http://localhost:3000/api/health:Grafana"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        local endpoint="${endpoint_info%%:*}"
        local service="${endpoint_info##*:}"
        
        local attempts=0
        while [[ $attempts -lt 10 ]]; do
            if curl -s "$endpoint" >/dev/null 2>&1; then
                log_success "$service is accessible"
                break
            fi
            ((attempts++))
            sleep 2
        done
        
        if [[ $attempts -eq 10 ]]; then
            log_warning "$service endpoint not accessible"
        fi
    done
}

# Display comprehensive system status
show_system_status() {
    echo ""
    log_success "Enhanced CSP System started successfully!"
    echo ""
    
    # Show container status
    echo "🐳 Container Status:"
    echo "==================="
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps
    echo ""
    
    # Access points
    echo "🌐 Service Access Points:"
    echo "========================="
    echo ""
    echo "📊 Core Databases:"
    echo "  • PostgreSQL (Main):      localhost:5432"
    echo "  • PostgreSQL (AI Models): localhost:5433"
    echo "  • PostgreSQL (Vector):    localhost:5434"
    echo "  • Redis:                  localhost:6379"
    echo ""
    
    echo "🧠 Vector Databases:"
    echo "  • Chroma API:    http://localhost:8200"
    echo "  • Qdrant API:    http://localhost:6333"
    echo "  • Weaviate API:  http://localhost:8080"
    if [[ "$PROFILE" == "milvus" ]] || [[ "$PROFILE" == "full" ]]; then
        echo "  • Milvus:        localhost:19530"
    fi
    echo ""
    
    echo "📊 Additional Services:"
    echo "  • MongoDB:        localhost:27017"
    echo "  • Elasticsearch:  http://localhost:9200"
    echo "  • InfluxDB:       http://localhost:8086"
    echo ""
    
    echo "🛠️ Administration:"
    echo "  • pgAdmin:        http://localhost:5050"
    echo "  • Redis Insight:  http://localhost:8001"
    echo "  • Mongo Express:  http://localhost:8081"
    echo ""
    
    echo "📈 Monitoring & Analytics:"
    echo "  • Prometheus:     http://localhost:9090"
    echo "  • Grafana:        http://localhost:3000"
    echo ""
    
    echo "🔧 Management Commands:"
    echo "  • View status:    ./scripts/status.sh"
    echo "  • View logs:      ./scripts/logs.sh"
    echo "  • Backup data:    ./scripts/backup-all.sh"
    echo "  • Stop system:    ./scripts/stop-all.sh"
    echo ""
    
    # System resources
    if command -v docker &>/dev/null; then
        echo "📊 Resource Usage:"
        docker system df --format "table {{.Type}}\t{{.TotalCount}}\t{{.Size}}\t{{.Reclaimable}}"
        echo ""
    fi
    
    log_info "System is ready for use!"
}

# Main execution
main() {
    preflight_checks
    prepare_images
    start_all_services
    
    if [[ "$DETACHED" == "true" ]]; then
        wait_for_critical_services
        check_external_services
    fi
    
    show_system_status
}

# Execute main function
main

# =============================================================================
# scripts/stop-all.sh
#!/bin/bash

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh" 2>/dev/null || {
    log_info() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
    log_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $1"; }
    log_warning() { echo -e "\033[1;33m[WARNING]\033[0m $1"; }
    log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; exit 1; }
    
    load_compose_config() {
        if [[ -f .docker-compose-path ]]; then
            source .docker-compose-path
        else
            if [[ -f "docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="docker-compose.yml"
            elif [[ -f "deployment/docker/database/docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="deployment/docker/database/docker-compose.yml"
            elif [[ -f "deployment/docker/docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
            else
                log_error "Could not find docker-compose.yml file"
            fi
            
            if command -v docker-compose &> /dev/null; then
                export COMPOSE_CMD="docker-compose"
            else
                export COMPOSE_CMD="docker compose"
            fi
        fi
    }
}

echo "🛑 Stopping Enhanced CSP System"
echo "==============================="

# Parse command line arguments
REMOVE_VOLUMES=false
REMOVE_IMAGES=false
FORCE=false
BACKUP_BEFORE_STOP=false
GRACEFUL_TIMEOUT=30

while [[ $# -gt 0 ]]; do
    case $1 in
        --remove-volumes|-v)
            REMOVE_VOLUMES=true
            shift
            ;;
        --remove-images|-i)
            REMOVE_IMAGES=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --backup|-b)
            BACKUP_BEFORE_STOP=true
            shift
            ;;
        --timeout|-t)
            GRACEFUL_TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --remove-volumes   Remove Docker volumes (DESTRUCTIVE)"
            echo "  -i, --remove-images    Remove Docker images"
            echo "  -f, --force           Force stop without confirmation"
            echo "  -b, --backup          Create backup before stopping"
            echo "  -t, --timeout SECONDS Graceful shutdown timeout (default: 30)"
            echo "  -h, --help            Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                    # Graceful stop"
            echo "  $0 --backup          # Backup then stop"
            echo "  $0 --force --remove-volumes  # Complete cleanup"
            exit 0
            ;;
        *)
            log_warning "Unknown option: $1"
            shift
            ;;
    esac
done

# Load configuration
load_compose_config

log_info "Using Docker Compose file: $DOCKER_COMPOSE_FILE"

# Confirmation for destructive operations
confirm_destructive_action() {
    if [[ "$FORCE" == "true" ]]; then
        return 0
    fi
    
    echo ""
    log_warning "⚠️  DESTRUCTIVE OPERATION WARNING ⚠️"
    echo ""
    
    if [[ "$REMOVE_VOLUMES" == "true" ]]; then
        echo "This will permanently delete ALL database data including:"
        echo "  • PostgreSQL databases (main, AI models, vector)"
        echo "  • Redis cache data"
        echo "  • MongoDB documents"
        echo "  • Elasticsearch indices"
        echo "  • InfluxDB time series data"
        echo "  • Vector database embeddings"
        echo "  • Monitoring data"
        echo ""
    fi
    
    if [[ "$REMOVE_IMAGES" == "true" ]]; then
        echo "This will remove Docker images and require re-download on next start"
        echo ""
    fi
    
    echo -n "Are you absolutely sure? Type 'YES' to confirm: "
    read -r confirmation
    
    if [[ "$confirmation" != "YES" ]]; then
        log_info "Operation cancelled"
        exit 0
    fi
}

# Create backup before stopping
create_backup() {
    if [[ "$BACKUP_BEFORE_STOP" != "true" ]]; then
        return
    fi
    
    log_info "Creating backup before stopping..."
    
    if [[ -f "./scripts/backup-all.sh" ]]; then
        bash "./scripts/backup-all.sh"
    else
        log_warning "Backup script not found, skipping backup"
    fi
}

# Show current status
show_current_status() {
    log_info "Current system status:"
    echo ""
    
    # Show running containers
    local running_containers=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps --services --filter "status=running" 2>/dev/null | wc -l)
    echo "Running containers: $running_containers"
    
    # Show Docker resource usage
    if command -v docker &>/dev/null; then
        echo ""
        echo "Current resource usage:"
        docker system df --format "table {{.Type}}\t{{.TotalCount}}\t{{.Size}}\t{{.Reclaimable}}"
    fi
    echo ""
}

# Graceful service shutdown
graceful_shutdown() {
    log_info "Performing graceful shutdown..."
    
    # Stop application services first
    log_info "Stopping application services..."
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" stop --timeout $GRACEFUL_TIMEOUT \
        grafana prometheus || true
    
    # Stop admin tools
    log_info "Stopping admin tools..."
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" stop --timeout $GRACEFUL_TIMEOUT \
        pgadmin redis-insight mongo-express || true
    
    # Stop vector databases
    log_info "Stopping vector databases..."
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" stop --timeout $GRACEFUL_TIMEOUT \
        chroma qdrant weaviate milvus-standalone || true
    
    # Stop additional databases
    log_info "Stopping additional databases..."
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" stop --timeout $GRACEFUL_TIMEOUT \
        mongodb elasticsearch influxdb || true
    
    # Stop supporting services for Milvus
    log_info "Stopping Milvus support services..."
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" stop --timeout $GRACEFUL_TIMEOUT \
        etcd minio || true
    
    # Stop core databases last
    log_info "Stopping core databases..."
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" stop --timeout $GRACEFUL_TIMEOUT \
        postgres_vector postgres_ai_models redis postgres || true
    
    # Stop any remaining services
    log_info "Stopping remaining services..."
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" stop --timeout $GRACEFUL_TIMEOUT || true
}

# Remove containers
remove_containers() {
    log_info "Removing containers..."
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" rm -f
}

# Remove volumes
remove_volumes() {
    if [[ "$REMOVE_VOLUMES" != "true" ]]; then
        return
    fi
    
    log_warning "Removing Docker volumes..."
    
    # Get project name for volume filtering
    local project_name=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" config --services | head -1 | sed 's/_[^_]*$//' 2>/dev/null || echo "csp")
    
    # Remove volumes
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" down --volumes --remove-orphans
    
    # Clean up any remaining project volumes
    docker volume ls -q | grep "^${project_name}_" | xargs -r docker volume rm 2>/dev/null || true
    
    log_warning "All data volumes have been removed"
}

# Remove images
remove_images() {
    if [[ "$REMOVE_IMAGES" != "true" ]]; then
        return
    fi
    
    log_info "Removing Docker images..."
    
    # Remove images used by the compose file
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" down --rmi all 2>/dev/null || true
    
    # Clean up any dangling images
    docker image prune -f 2>/dev/null || true
    
    log_info "Docker images removed"
}

# Clean up networks
cleanup_networks() {
    log_info "Cleaning up networks..."
    
    # Remove compose networks
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
    
    # Clean up any unused networks
    docker network prune -f 2>/dev/null || true
}

# Show final status
show_final_status() {
    echo ""
    log_success "Enhanced CSP System stopped successfully!"
    echo ""
    
    # Show remaining Docker resources
    if command -v docker &>/dev/null; then
        echo "Remaining Docker resources:"
        docker system df --format "table {{.Type}}\t{{.TotalCount}}\t{{.Size}}\t{{.Reclaimable}}"
        echo ""
    fi
    
    # Show any remaining containers
    local remaining_containers=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps -q 2>/dev/null | wc -l)
    if [[ $remaining_containers -gt 0 ]]; then
        log_warning "Some containers may still be running:"
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps
    else
        log_info "All containers have been stopped"
    fi
    
    echo ""
    echo "To restart the system:"
    echo "  • Start databases: ./scripts/start-databases.sh"
    echo "  • Start all:       ./scripts/start-all.sh"
    echo ""
    
    if [[ "$REMOVE_VOLUMES" == "true" ]]; then
        log_warning "⚠️  All data has been permanently deleted"
        echo "You will need to run initial setup on next start"
    fi
}

# Main execution
main() {
    show_current_status
    
    # Confirm destructive operations
    if [[ "$REMOVE_VOLUMES" == "true" ]] || [[ "$REMOVE_IMAGES" == "true" ]]; then
        confirm_destructive_action
    fi
    
    create_backup
    graceful_shutdown
    remove_containers
    remove_volumes
    remove_images
    cleanup_networks
    show_final_status
}

# Execute main function
main

# =============================================================================
# scripts/backup-all.sh
#!/bin/bash

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh" 2>/dev/null || {
    log_info() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
    log_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $1"; }
    log_warning() { echo -e "\033[1;33m[WARNING]\033[0m $1"; }
    log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; exit 1; }
    
    load_compose_config() {
        if [[ -f .docker-compose-path ]]; then
            source .docker-compose-path
        else
            if [[ -f "docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="docker-compose.yml"
            elif [[ -f "deployment/docker/database/docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="deployment/docker/database/docker-compose.yml"
            elif [[ -f "deployment/docker/docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
            else
                log_error "Could not find docker-compose.yml file"
            fi
            
            if command -v docker-compose &> /dev/null; then
                export COMPOSE_CMD="docker-compose"
            else
                export COMPOSE_CMD="docker compose"
            fi
        fi
    }
}

echo "💾 Enhanced CSP System Backup"
echo "============================="

# Parse command line arguments
BACKUP_TYPE="full"
COMPRESS=true
ENCRYPT=false
RETENTION_DAYS=30
BACKUP_LOCATION=""
S3_BUCKET=""
VERIFY_BACKUP=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --type|-t)
            BACKUP_TYPE="$2"
            shift 2
            ;;
        --no-compress)
            COMPRESS=false
            shift
            ;;
        --encrypt|-e)
            ENCRYPT=true
            shift
            ;;
        --retention|-r)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        --location|-l)
            BACKUP_LOCATION="$2"
            shift 2
            ;;
        --s3-bucket|-s)
            S3_BUCKET="$2"
            shift 2
            ;;
        --no-verify)
            VERIFY_BACKUP=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --type TYPE          Backup type (full|databases|configs|minimal)"
            echo "  --no-compress           Don't compress backup files"
            echo "  -e, --encrypt           Encrypt backup files"
            echo "  -r, --retention DAYS    Retention period in days (default: 30)"
            echo "  -l, --location PATH     Custom backup location"
            echo "  -s, --s3-bucket BUCKET  Upload to S3 bucket"
            echo "  --no-verify             Skip backup verification"
            echo "  -h, --help              Show this help"
            echo ""
            echo "Backup Types:"
            echo "  full       - Complete system backup (default)"
            echo "  databases  - Database data only"
            echo "  configs    - Configuration files only"
            echo "  minimal    - Critical data only"
            exit 0
            ;;
        *)
            log_warning "Unknown option: $1"
            shift
            ;;
    esac
done

# Load configuration
load_compose_config

# Set backup directory
if [[ -n "$BACKUP_LOCATION" ]]; then
    BACKUP_BASE_DIR="$BACKUP_LOCATION"
else
    BACKUP_BASE_DIR="./backups"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${BACKUP_BASE_DIR}/${BACKUP_TYPE}_backup_${TIMESTAMP}"

log_info "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Load environment variables
if [[ -f .env.docker ]]; then
    source .env.docker
fi

# Backup verification
verify_containers_running() {
    log_info "Verifying containers are running..."
    
    local required_services=("postgres" "redis")
    for service in "${required_services[@]}"; do
        if ! $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps "$service" | grep -q "Up"; then
            log_warning "$service is not running - some backups may fail"
        fi
    done
}

# PostgreSQL backup
backup_postgresql() {
    local db_name="$1"
    local db_user="$2"
    local db_container="$3"
    local output_file="$4"
    
    log_info "Backing up PostgreSQL database: $db_name"
    
    # Check if container is running
    if ! $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps "$db_container" | grep -q "Up"; then
        log_warning "$db_container is not running, skipping backup"
        return 1
    fi
    
    # Create backup
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" exec -T "$db_container" \
        pg_dump -U "$db_user" -d "$db_name" --verbose --no-password > "$output_file" 2>/dev/null
    
    if [[ $? -eq 0 && -s "$output_file" ]]; then
        log_success "PostgreSQL backup completed: $output_file"
        
        # Compress if requested
        if [[ "$COMPRESS" == "true" ]]; then
            gzip "$output_file"
            log_info "Backup compressed: ${output_file}.gz"
        fi
        return 0
    else
        log_warning "PostgreSQL backup failed or empty: $db_name"
        rm -f "$output_file"
        return 1
    fi
}

# Redis backup
backup_redis() {
    log_info "Backing up Redis data..."
    
    if ! $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps redis | grep -q "Up"; then
        log_warning "Redis is not running, skipping backup"
        return 1
    fi
    
    # Create Redis backup
    local redis_backup="$BACKUP_DIR/redis_backup.rdb"
    
    # Force Redis to save current state
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli BGSAVE >/dev/null 2>&1
    
    # Wait for background save to complete
    local save_status=""
    local attempts=0
    while [[ $attempts -lt 30 ]]; do
        save_status=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli LASTSAVE 2>/dev/null)
        sleep 1
        local new_status=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli LASTSAVE 2>/dev/null)
        if [[ "$save_status" != "$new_status" ]]; then
            break
        fi
        ((attempts++))
    done
    
    # Copy the RDB file
    local redis_container=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps -q redis)
    if [[ -n "$redis_container" ]]; then
        docker cp "$redis_container:/data/dump.rdb" "$redis_backup" 2>/dev/null
        
        if [[ -f "$redis_backup" && -s "$redis_backup" ]]; then
            log_success "Redis backup completed: $redis_backup"
            
            if [[ "$COMPRESS" == "true" ]]; then
                gzip "$redis_backup"
                log_info "Redis backup compressed"
            fi
            return 0
        fi
    fi
    
    log_warning "Redis backup failed"
    return 1
}

# MongoDB backup
backup_mongodb() {
    log_info "Backing up MongoDB..."
    
    if ! $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps mongodb | grep -q "Up"; then
        log_warning "MongoDB is not running, skipping backup"
        return 1
    fi
    
    local mongo_backup_dir="$BACKUP_DIR/mongodb"
    
    # Create MongoDB backup
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" exec -T mongodb \
        mongodump --host localhost:27017 --out /tmp/mongodb_backup >/dev/null 2>&1
    
    # Copy backup from container
    local mongo_container=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps -q mongodb)
    if [[ -n "$mongo_container" ]]; then
        docker cp "$mongo_container:/tmp/mongodb_backup" "$mongo_backup_dir" 2>/dev/null
        
        if [[ -d "$mongo_backup_dir" ]]; then
            log_success "MongoDB backup completed: $mongo_backup_dir"
            
            if [[ "$COMPRESS" == "true" ]]; then
                tar -czf "${mongo_backup_dir}.tar.gz" -C "$BACKUP_DIR" "mongodb"
                rm -rf "$mongo_backup_dir"
                log_info "MongoDB backup compressed"
            fi
            return 0
        fi
    fi
    
    log_warning "MongoDB backup failed"
    return 1
}

# Vector databases backup
backup_vector_databases() {
    log_info "Backing up vector databases..."
    
    local vector_backup_dir="$BACKUP_DIR/vector_databases"
    mkdir -p "$vector_backup_dir"
    
    # Chroma backup
    if $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps chroma | grep -q "Up"; then
        log_info "Backing up Chroma data..."
        local chroma_container=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps -q chroma)
        if [[ -n "$chroma_container" ]]; then
            docker cp "$chroma_container:/chroma/chroma" "$vector_backup_dir/chroma_data" 2>/dev/null
            if [[ -d "$vector_backup_dir/chroma_data" ]]; then
                log_success "Chroma backup completed"
            fi
        fi
    fi
    
    # Qdrant backup
    if $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps qdrant | grep -q "Up"; then
        log_info "Backing up Qdrant data..."
        local qdrant_container=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps -q qdrant)
        if [[ -n "$qdrant_container" ]]; then
            docker cp "$qdrant_container:/qdrant/storage" "$vector_backup_dir/qdrant_data" 2>/dev/null
            if [[ -d "$vector_backup_dir/qdrant_data" ]]; then
                log_success "Qdrant backup completed"
            fi
        fi
    fi
    
    # Weaviate backup
    if $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps weaviate | grep -q "Up"; then
        log_info "Backing up Weaviate data..."
        local weaviate_container=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps -q weaviate)
        if [[ -n "$weaviate_container" ]]; then
            docker cp "$weaviate_container:/var/lib/weaviate" "$vector_backup_dir/weaviate_data" 2>/dev/null
            if [[ -d "$vector_backup_dir/weaviate_data" ]]; then
                log_success "Weaviate backup completed"
            fi
        fi
    fi
    
    # Compress vector databases backup
    if [[ "$COMPRESS" == "true" && -d "$vector_backup_dir" ]]; then
        tar -czf "${vector_backup_dir}.tar.gz" -C "$BACKUP_DIR" "vector_databases"
        rm -rf "$vector_backup_dir"
        log_info "Vector databases backup compressed"
    fi
}

# Configuration backup
backup_configurations() {
    log_info "Backing up configuration files..."
    
    local config_backup_dir="$BACKUP_DIR/configurations"
    mkdir -p "$config_backup_dir"
    
    # Copy configuration files
    local config_files=(
        ".env.docker"
        "docker-compose.yml"
        "monitoring/prometheus/prometheus.yml"
        "monitoring/grafana/datasources/"
        "config/"
    )
    
    for config_item in "${config_files[@]}"; do
        if [[ -e "$config_item" ]]; then
            cp -r "$config_item" "$config_backup_dir/" 2>/dev/null
            log_info "Backed up: $config_item"
        fi
    done
    
    # Backup Docker Compose file from alternate locations
    if [[ -f "$DOCKER_COMPOSE_FILE" ]]; then
        cp "$DOCKER_COMPOSE_FILE" "$config_backup_dir/docker-compose.yml"
        log_info "Backed up: $DOCKER_COMPOSE_FILE"
    fi
    
    log_success "Configuration backup completed"
}

# Create manifest file
create_backup_manifest() {
    local manifest_file="$BACKUP_DIR/backup_manifest.txt"
    
    cat > "$manifest_file" << EOF
Enhanced CSP System Backup Manifest
==================================
Backup Type: $BACKUP_TYPE
Backup Time: $(date)
Backup Location: $BACKUP_DIR
Compression: $COMPRESS
Encryption: $ENCRYPT

System Information:
- Docker Compose File: $DOCKER_COMPOSE_FILE
- Operating System: $(uname -s)
- Architecture: $(uname -m)

Backup Contents:
EOF
    
    # List backup contents
    find "$BACKUP_DIR" -type f -not -name "backup_manifest.txt" | while read -r file; do
        local size=$(du -h "$file" | cut -f1)
        echo "- $(basename "$file") ($size)" >> "$manifest_file"
    done
    
    log_success "Backup manifest created: $manifest_file"
}

# Verify backup integrity
verify_backup_integrity() {
    if [[ "$VERIFY_BACKUP" != "true" ]]; then
        return
    fi
    
    log_info "Verifying backup integrity..."
    
    # Check if backup directory exists and has content
    if [[ ! -d "$BACKUP_DIR" ]]; then
        log_error "Backup directory not found: $BACKUP_DIR"
    fi
    
    local backup_size=$(du -sh "$BACKUP_DIR" | cut -f1)
    local file_count=$(find "$BACKUP_DIR" -type f | wc -l)
    
    log_info "Backup verification results:"
    echo "  • Total size: $backup_size"
    echo "  • File count: $file_count"
    
    # Verify compressed files
    if [[ "$COMPRESS" == "true" ]]; then
        find "$BACKUP_DIR" -name "*.gz" | while read -r gzfile; do
            if gzip -t "$gzfile" 2>/dev/null; then
                log_success "✓ $(basename "$gzfile") - valid"
            else
                log_warning "✗ $(basename "$gzfile") - corrupted"
            fi
        done
    fi
    
    log_success "Backup verification completed"
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up backups older than $RETENTION_DAYS days..."
    
    find "$BACKUP_BASE_DIR" -type d -name "*_backup_*" -mtime +$RETENTION_DAYS -exec rm -rf {} \; 2>/dev/null
    
    local remaining_backups=$(find "$BACKUP_BASE_DIR" -type d -name "*_backup_*" | wc -l)
    log_info "Remaining backups: $remaining_backups"
}

# Upload to S3 (if configured)
upload_to_s3() {
    if [[ -z "$S3_BUCKET" ]]; then
        return
    fi
    
    log_info "Uploading backup to S3 bucket: $S3_BUCKET"
    
    if command -v aws &> /dev/null; then
        # Create tar archive for upload
        local archive_name="csp_backup_${TIMESTAMP}.tar.gz"
        tar -czf "/tmp/$archive_name" -C "$BACKUP_BASE_DIR" "$(basename "$BACKUP_DIR")"
        
        # Upload to S3
        aws s3 cp "/tmp/$archive_name" "s3://$S3_BUCKET/csp-backups/$archive_name"
        
        if [[ $? -eq 0 ]]; then
            log_success "Backup uploaded to S3: s3://$S3_BUCKET/csp-backups/$archive_name"
            rm -f "/tmp/$archive_name"
        else
            log_warning "S3 upload failed"
        fi
    else
        log_warning "AWS CLI not found, skipping S3 upload"
    fi
}

# Main backup execution
perform_backup() {
    case "$BACKUP_TYPE" in
        "full")
            log_info "Performing full system backup..."
            backup_postgresql "csp_visual_designer" "csp_user" "postgres" "$BACKUP_DIR/main_db.sql"
            backup_postgresql "ai_models_db" "ai_models_user" "postgres_ai_models" "$BACKUP_DIR/ai_models_db.sql"
            backup_postgresql "vector_db" "vector_user" "postgres_vector" "$BACKUP_DIR/vector_db.sql"
            backup_redis
            backup_mongodb
            backup_vector_databases
            backup_configurations
            ;;
        "databases")
            log_info "Performing database-only backup..."
            backup_postgresql "csp_visual_designer" "csp_user" "postgres" "$BACKUP_DIR/main_db.sql"
            backup_postgresql "ai_models_db" "ai_models_user" "postgres_ai_models" "$BACKUP_DIR/ai_models_db.sql"
            backup_postgresql "vector_db" "vector_user" "postgres_vector" "$BACKUP_DIR/vector_db.sql"
            backup_redis
            backup_mongodb
            backup_vector_databases
            ;;
        "configs")
            log_info "Performing configuration backup..."
            backup_configurations
            ;;
        "minimal")
            log_info "Performing minimal backup..."
            backup_postgresql "csp_visual_designer" "csp_user" "postgres" "$BACKUP_DIR/main_db.sql"
            backup_redis
            backup_configurations
            ;;
        *)
            log_error "Unknown backup type: $BACKUP_TYPE"
            ;;
    esac
}

# Main execution
main() {
    log_info "Starting Enhanced CSP System backup..."
    log_info "Backup type: $BACKUP_TYPE"
    log_info "Backup location: $BACKUP_DIR"
    echo ""
    
    verify_containers_running
    perform_backup
    create_backup_manifest
    verify_backup_integrity
    cleanup_old_backups
    upload_to_s3
    
    # Final summary
    local backup_size=$(du -sh "$BACKUP_DIR" | cut -f1)
    echo ""
    log_success "Backup completed successfully!"
    echo ""
    echo "📊 Backup Summary:"
    echo "=================="
    echo "  • Backup Type: $BACKUP_TYPE"
    echo "  • Backup Size: $backup_size"
    echo "  • Location: $BACKUP_DIR"
    echo "  • Compressed: $COMPRESS"
    echo "  • Encrypted: $ENCRYPT"
    echo ""
    echo "🔧 Restoration:"
    echo "  To restore from this backup, see documentation"
    echo "  or use: ./scripts/restore.sh $BACKUP_DIR"
}

# Execute main function
main

# =============================================================================
# scripts/status.sh
#!/bin/bash

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh" 2>/dev/null || {
    log_info() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
    log_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $1"; }
    log_warning() { echo -e "\033[1;33m[WARNING]\033[0m $1"; }
    log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; exit 1; }
    
    load_compose_config() {
        if [[ -f .docker-compose-path ]]; then
            source .docker-compose-path
        else
            if [[ -f "docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="docker-compose.yml"
            elif [[ -f "deployment/docker/database/docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="deployment/docker/database/docker-compose.yml"
            elif [[ -f "deployment/docker/docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
            else
                log_error "Could not find docker-compose.yml file"
            fi
            
            if command -v docker-compose &> /dev/null; then
                export COMPOSE_CMD="docker-compose"
            else
                export COMPOSE_CMD="docker compose"
            fi
        fi
    }
}

echo "📊 Enhanced CSP System Status"
echo "============================="

# Parse command line arguments
VERBOSE=false
CHECK_HEALTH=true
CHECK_PERFORMANCE=false
OUTPUT_FORMAT="table"
WATCH_MODE=false
REFRESH_INTERVAL=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --no-health)
            CHECK_HEALTH=false
            shift
            ;;
        --performance|-p)
            CHECK_PERFORMANCE=true
            shift
            ;;
        --format|-f)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --watch|-w)
            WATCH_MODE=true
            shift
            ;;
        --interval|-i)
            REFRESH_INTERVAL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose          Show detailed information"
            echo "  --no-health           Skip health checks"
            echo "  -p, --performance     Include performance metrics"
            echo "  -f, --format FORMAT   Output format (table|json|yaml)"
            echo "  -w, --watch           Watch mode (continuous updates)"
            echo "  -i, --interval N      Refresh interval for watch mode (default: 5)"
            echo "  -h, --help            Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                    # Basic status"
            echo "  $0 --verbose --performance  # Detailed status with metrics"
            echo "  $0 --watch            # Continuous monitoring"
            exit 0
            ;;
        *)
            log_warning "Unknown option: $1"
            shift
            ;;
    esac
done

# Load configuration
load_compose_config

# Status checking functions
check_docker_status() {
    if [[ "$OUTPUT_FORMAT" == "table" ]]; then
        echo "🐳 Docker System Status"
        echo "======================="
    fi
    
    # Check Docker daemon
    if docker info >/dev/null 2>&1; then
        if [[ "$OUTPUT_FORMAT" == "table" ]]; then
            log_success "Docker daemon is running"
        fi
    else
        if [[ "$OUTPUT_FORMAT" == "table" ]]; then
            log_error "Docker daemon is not running"
        fi
        return 1
    fi
    
    # Docker system info
    if [[ "$VERBOSE" == "true" && "$OUTPUT_FORMAT" == "table" ]]; then
        echo ""
        echo "Docker System Information:"
        docker system df
        echo ""
    fi
}

check_container_status() {
    if [[ "$OUTPUT_FORMAT" == "table" ]]; then
        echo "📦 Container Status"
        echo "=================="
    fi
    
    # Get container status
    local containers=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps --format json 2>/dev/null)
    
    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        echo "$containers"
        return
    fi
    
    # Show container status table
    printf "%-25s %-15s %-10s %-15s\n" "SERVICE" "STATUS" "PORTS" "HEALTH"
    printf "%-25s %-15s %-10s %-15s\n" "-------" "------" "-----" "------"
    
    # Parse and display container info
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps --format "table {{.Service}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null | tail -n +2 | while read -r line; do
        if [[ -n "$line" ]]; then
            echo "$line"
        fi
    done
    
    echo ""
}

check_database_health() {
    if [[ "$CHECK_HEALTH" != "true" ]]; then
        return
    fi
    
    if [[ "$OUTPUT_FORMAT" == "table" ]]; then
        echo "🗄️ Database Health Status"
        echo "========================="
    fi
    
    # Define health checks
    declare -A health_checks=(
        ["postgres"]="pg_isready -U csp_user -d csp_visual_designer"
        ["postgres_ai_models"]="pg_isready -U ai_models_user -d ai_models_db"
        ["postgres_vector"]="pg_isready -U vector_user -d vector_db"
        ["redis"]="redis-cli ping"
        ["mongodb"]="mongosh --quiet --eval 'db.adminCommand(\"ping\")'"
    )
    
    printf "%-20s %-10s %-30s\n" "DATABASE" "STATUS" "RESPONSE TIME"
    printf "%-20s %-10s %-30s\n" "--------" "------" "-------------"
    
    for service in "${!health_checks[@]}"; do
        local health_cmd="${health_checks[$service]}"
        local start_time=$(date +%s%3N)
        
        if $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" exec -T "$service" $health_cmd >/dev/null 2>&1; then
            local end_time=$(date +%s%3N)
            local response_time=$((end_time - start_time))
            printf "%-20s %-10s %-30s\n" "$service" "✅ HEALTHY" "${response_time}ms"
        else
            printf "%-20s %-10s %-30s\n" "$service" "❌ UNHEALTHY" "N/A"
        fi
    done
    
    echo ""
}

check_vector_database_status() {
    if [[ "$OUTPUT_FORMAT" == "table" ]]; then
        echo "🧠 Vector Database Status"
        echo "========================"
    fi
    
    # Vector database endpoints
    declare -A vector_endpoints=(
        ["Chroma"]="http://localhost:8200/api/v1/heartbeat"
        ["Qdrant"]="http://localhost:6333/health"
        ["Weaviate"]="http://localhost:8080/v1/.well-known/ready"
        ["Elasticsearch"]="http://localhost:9200/_cluster/health"
        ["InfluxDB"]="http://localhost:8086/health"
    )
    
    printf "%-15s %-10s %-15s %-30s\n" "SERVICE" "STATUS" "RESPONSE" "ENDPOINT"
    printf "%-15s %-10s %-15s %-30s\n" "-------" "------" "--------" "--------"
    
    for service in "${!vector_endpoints[@]}"; do
        local endpoint="${vector_endpoints[$service]}"
        local start_time=$(date +%s%3N)
        
        if curl -s --max-time 5 "$endpoint" >/dev/null 2>&1; then
            local end_time=$(date +%s%3N)
            local response_time=$((end_time - start_time))
            printf "%-15s %-10s %-15s %-30s\n" "$service" "✅ ONLINE" "${response_time}ms" "$endpoint"
        else
            printf "%-15s %-10s %-15s %-30s\n" "$service" "❌ OFFLINE" "N/A" "$endpoint"
        fi
    done
    
    echo ""
}

check_admin_tools_status() {
    if [[ "$OUTPUT_FORMAT" == "table" ]]; then
        echo "🛠️ Admin Tools Status"
        echo "====================="
    fi
    
    declare -A admin_endpoints=(
        ["pgAdmin"]="http://localhost:5050/misc/ping"
        ["Redis Insight"]="http://localhost:8001"
        ["Mongo Express"]="http://localhost:8081"
        ["Prometheus"]="http://localhost:9090/-/ready"
        ["Grafana"]="http://localhost:3000/api/health"
    )
    
    printf "%-15s %-10s %-15s %-30s\n" "TOOL" "STATUS" "RESPONSE" "URL"
    printf "%-15s %-10s %-15s %-30s\n" "----" "------" "--------" "---"
    
    for tool in "${!admin_endpoints[@]}"; do
        local endpoint="${admin_endpoints[$tool]}"
        local start_time=$(date +%s%3N)
        
        if curl -s --max-time 5 "$endpoint" >/dev/null 2>&1; then
            local end_time=$(date +%s%3N)
            local response_time=$((end_time - start_time))
            printf "%-15s %-10s %-15s %-30s\n" "$tool" "✅ ONLINE" "${response_time}ms" "$endpoint"
        else
            printf "%-15s %-10s %-15s %-30s\n" "$tool" "❌ OFFLINE" "N/A" "$endpoint"
        fi
    done
    
    echo ""
}

check_system_resources() {
    if [[ "$CHECK_PERFORMANCE" != "true" ]]; then
        return
    fi
    
    if [[ "$OUTPUT_FORMAT" == "table" ]]; then
        echo "⚡ System Resources"
        echo "=================="
    fi
    
    # CPU usage
    if command -v top >/dev/null 2>&1; then
        local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
        echo "CPU Usage: ${cpu_usage}%"
    fi
    
    # Memory usage
    if command -v free >/dev/null 2>&1; then
        echo ""
        echo "Memory Usage:"
        free -h
    fi
    
    # Disk usage
    echo ""
    echo "Disk Usage:"
    df -h | grep -E "(Filesystem|/dev/)" | head -5
    
    # Docker resource usage
    echo ""
    echo "Docker Resource Usage:"
    docker system df
    
    echo ""
}

check_container_performance() {
    if [[ "$CHECK_PERFORMANCE" != "true" ]]; then
        return
    fi
    
    if [[ "$OUTPUT_FORMAT" == "table" ]]; then
        echo "📊 Container Performance"
        echo "======================="
    fi
    
    # Get container IDs
    local container_ids=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps -q 2>/dev/null)
    
    if [[ -n "$container_ids" ]]; then
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" $container_ids
    else
        echo "No containers running"
    fi
    
    echo ""
}

check_network_connectivity() {
    if [[ "$VERBOSE" != "true" ]]; then
        return
    fi
    
    if [[ "$OUTPUT_FORMAT" == "table" ]]; then
        echo "🌐 Network Connectivity"
        echo "======================"
    fi
    
    # Check internal Docker network
    local network_name=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" config | grep -A 1 "networks:" | tail -1 | awk '{print $1}' | tr -d ':')
    
    if [[ -n "$network_name" ]]; then
        echo "Docker Network: $network_name"
        docker network inspect "$network_name" --format "{{range .Containers}}{{.Name}}: {{.IPv4Address}}{{println}}{{end}}" 2>/dev/null || echo "Network info not available"
    fi
    
    echo ""
}

show_access_urls() {
    if [[ "$OUTPUT_FORMAT" == "table" ]]; then
        echo "🔗 Service Access URLs"
        echo "====================="
        echo ""
        echo "📊 Databases:"
        echo "  • PostgreSQL (Main):      localhost:5432"
        echo "  • PostgreSQL (AI Models): localhost:5433"
        echo "  • PostgreSQL (Vector):    localhost:5434"
        echo "  • Redis:                  localhost:6379"
        echo "  • MongoDB:                localhost:27017"
        echo ""
        echo "🧠 Vector Databases:"
        echo "  • Chroma:        http://localhost:8200"
        echo "  • Qdrant:        http://localhost:6333"
        echo "  • Weaviate:      http://localhost:8080"
        echo "  • Elasticsearch: http://localhost:9200"
        echo "  • InfluxDB:      http://localhost:8086"
        echo ""
        echo "🛠️ Admin Tools:"
        echo "  • pgAdmin:       http://localhost:5050"
        echo "  • Redis Insight: http://localhost:8001"
        echo "  • Mongo Express: http://localhost:8081"
        echo ""
        echo "📈 Monitoring:"
        echo "  • Prometheus:    http://localhost:9090"
        echo "  • Grafana:       http://localhost:3000"
        echo ""
    fi
}

show_summary() {
    if [[ "$OUTPUT_FORMAT" != "table" ]]; then
        return
    fi
    
    # Count running services
    local total_services=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" config --services | wc -l)
    local running_services=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps --services --filter "status=running" 2>/dev/null | wc -l)
    
    echo "📈 System Summary"
    echo "================="
    echo "Running Services: $running_services/$total_services"
    
    # Calculate uptime
    local oldest_container=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps -q | head -1)
    if [[ -n "$oldest_container" ]]; then
        local start_time=$(docker inspect "$oldest_container" --format='{{.State.StartedAt}}' 2>/dev/null)
        if [[ -n "$start_time" ]]; then
            echo "System Uptime: $(date -d "$start_time" +'%H:%M:%S on %Y-%m-%d' 2>/dev/null || echo 'Unknown')"
        fi
    fi
    
    # System health
    if [[ $running_services -eq $total_services ]]; then
        log_success "All services are running"
    elif [[ $running_services -gt 0 ]]; then
        log_warning "Some services are not running ($((total_services - running_services)) stopped)"
    else
        log_error "No services are running"
    fi
    
    echo ""
}

# Watch mode function
watch_status() {
    while true; do
        clear
        echo "Enhanced CSP System Status - Watch Mode"
        echo "Update: $(date)"
        echo "Refresh Interval: ${REFRESH_INTERVAL}s (Press Ctrl+C to exit)"
        echo ""
        
        check_docker_status
        check_container_status
        check_database_health
        check_vector_database_status
        check_system_resources
        show_summary
        
        sleep "$REFRESH_INTERVAL"
    done
}

# Main execution
main() {
    if [[ "$WATCH_MODE" == "true" ]]; then
        watch_status
        return
    fi
    
    # Load environment variables
    if [[ -f .env.docker ]]; then
        source .env.docker 2>/dev/null
    fi
    
    check_docker_status
    check_container_status
    check_database_health
    check_vector_database_status
    check_admin_tools_status
    check_system_resources
    check_container_performance
    check_network_connectivity
    show_access_urls
    show_summary
}

# Execute main function
main

# =============================================================================
# scripts/logs.sh
#!/bin/bash

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh" 2>/dev/null || {
    log_info() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
    log_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $1"; }
    log_warning() { echo -e "\033[1;33m[WARNING]\033[0m $1"; }
    log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; exit 1; }
    
    load_compose_config() {
        if [[ -f .docker-compose-path ]]; then
            source .docker-compose-path
        else
            if [[ -f "docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="docker-compose.yml"
            elif [[ -f "deployment/docker/database/docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="deployment/docker/database/docker-compose.yml"
            elif [[ -f "deployment/docker/docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
            else
                log_error "Could not find docker-compose.yml file"
            fi
            
            if command -v docker-compose &> /dev/null; then
                export COMPOSE_CMD="docker-compose"
            else
                export COMPOSE_CMD="docker compose"
            fi
        fi
    }
}

echo "📋 Enhanced CSP System Logs"
echo "==========================="

# Parse command line arguments
SERVICE=""
FOLLOW=false
TAIL_LINES=100
SEARCH_PATTERN=""
LOG_LEVEL=""
TIME_FILTER=""
OUTPUT_FILE=""
AGGREGATE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --service|-s)
            SERVICE="$2"
            shift 2
            ;;
        --follow|-f)
            FOLLOW=true
            shift
            ;;
        --tail|-t)
            TAIL_LINES="$2"
            shift 2
            ;;
        --search|-g)
            SEARCH_PATTERN="$2"
            shift 2
            ;;
        --level|-l)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --since)
            TIME_FILTER="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --aggregate|-a)
            AGGREGATE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -s, --service SERVICE    Show logs for specific service"
            echo "  -f, --follow            Follow log output"
            echo "  -t, --tail N            Number of lines to show (default: 100)"
            echo "  -g, --search PATTERN    Search for pattern in logs"
            echo "  -l, --level LEVEL       Filter by log level (error, warn, info, debug)"
            echo "  --since TIME            Show logs since time (e.g., '1h', '30m', '2023-01-01')"
            echo "  -o, --output FILE       Save logs to file"
            echo "  -a, --aggregate         Show aggregated logs from all services"
            echo "  -h, --help              Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 -s postgres -f               # Follow PostgreSQL logs"
            echo "  $0 -g 'ERROR' -t 50             # Search for errors in last 50 lines"
            echo "  $0 --since '1h' --level error   # Show errors from last hour"
            echo "  $0 -a -o system.log             # Save all logs to file"
            exit 0
            ;;
        *)
            log_warning "Unknown option: $1"
            shift
            ;;
    esac
done

# Load configuration
load_compose_config

# Available services
get_available_services() {
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" config --services 2>/dev/null
}

# Show service logs
show_service_logs() {
    local service="$1"
    local cmd_args=""
    
    # Build command arguments
    if [[ "$FOLLOW" == "true" ]]; then
        cmd_args="$cmd_args --follow"
    fi
    
    if [[ -n "$TAIL_LINES" ]]; then
        cmd_args="$cmd_args --tail $TAIL_LINES"
    fi
    
    if [[ -n "$TIME_FILTER" ]]; then
        cmd_args="$cmd_args --since $TIME_FILTER"
    fi
    
    log_info "Showing logs for service: $service"
    
    # Apply filters
    local log_output
    if [[ -n "$OUTPUT_FILE" ]]; then
        log_output="tee $OUTPUT_FILE"
    else
        log_output="cat"
    fi
    
    if [[ -n "$SEARCH_PATTERN" && -n "$LOG_LEVEL" ]]; then
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs $cmd_args "$service" 2>&1 | grep -i "$LOG_LEVEL" | grep -i "$SEARCH_PATTERN" | $log_output
    elif [[ -n "$SEARCH_PATTERN" ]]; then
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs $cmd_args "$service" 2>&1 | grep -i "$SEARCH_PATTERN" | $log_output
    elif [[ -n "$LOG_LEVEL" ]]; then
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs $cmd_args "$service" 2>&1 | grep -i "$LOG_LEVEL" | $log_output
    else
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs $cmd_args "$service" 2>&1 | $log_output
    fi
}

# Show all logs aggregated
show_all_logs() {
    local cmd_args=""
    
    if [[ "$FOLLOW" == "true" ]]; then
        cmd_args="$cmd_args --follow"
    fi
    
    if [[ -n "$TAIL_LINES" ]]; then
        cmd_args="$cmd_args --tail $TAIL_LINES"
    fi
    
    if [[ -n "$TIME_FILTER" ]]; then
        cmd_args="$cmd_args --since $TIME_FILTER"
    fi
    
    log_info "Showing aggregated logs from all services"
    
    local log_output
    if [[ -n "$OUTPUT_FILE" ]]; then
        log_output="tee $OUTPUT_FILE"
    else
        log_output="cat"
    fi
    
    if [[ -n "$SEARCH_PATTERN" && -n "$LOG_LEVEL" ]]; then
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs $cmd_args 2>&1 | grep -i "$LOG_LEVEL" | grep -i "$SEARCH_PATTERN" | $log_output
    elif [[ -n "$SEARCH_PATTERN" ]]; then
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs $cmd_args 2>&1 | grep -i "$SEARCH_PATTERN" | $log_output
    elif [[ -n "$LOG_LEVEL" ]]; then
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs $cmd_args 2>&1 | grep -i "$LOG_LEVEL" | $log_output
    else
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs $cmd_args 2>&1 | $log_output
    fi
}

# Show log statistics
show_log_statistics() {
    log_info "Log Statistics (last 1000 lines):"
    echo ""
    
    local temp_log="/tmp/csp_logs_temp.txt"
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs --tail 1000 > "$temp_log" 2>&1
    
    # Count log levels
    echo "Log Level Distribution:"
    echo "======================"
    printf "%-10s %s\n" "ERROR:" "$(grep -ci 'error' "$temp_log" || echo 0)"
    printf "%-10s %s\n" "WARN:" "$(grep -ci 'warn' "$temp_log" || echo 0)"
    printf "%-10s %s\n" "INFO:" "$(grep -ci 'info' "$temp_log" || echo 0)"
    printf "%-10s %s\n" "DEBUG:" "$(grep -ci 'debug' "$temp_log" || echo 0)"
    echo ""
    
    # Most active services
    echo "Most Active Services:"
    echo "===================="
    grep -o '^[^|]*' "$temp_log" | sort | uniq -c | sort -nr | head -5
    echo ""
    
    # Recent errors
    echo "Recent Errors (last 5):"
    echo "======================="
    grep -i 'error' "$temp_log" | tail -5
    echo ""
    
    rm -f "$temp_log"
}

# Interactive service selection
select_service_interactively() {
    local services=($(get_available_services))
    
    if [[ ${#services[@]} -eq 0 ]]; then
        log_error "No services found in docker-compose.yml"
    fi
    
    echo "Available services:"
    echo "=================="
    for i in "${!services[@]}"; do
        printf "%2d) %s\n" $((i+1)) "${services[i]}"
    done
    echo ""
    
    echo -n "Select service number (or 'all' for aggregated logs): "
    read -r selection
    
    if [[ "$selection" == "all" ]]; then
        AGGREGATE=true
        return
    elif [[ "$selection" =~ ^[0-9]+$ ]] && [[ $selection -ge 1 ]] && [[ $selection -le ${#services[@]} ]]; then
        SERVICE="${services[$((selection-1))]}"
    else
        log_error "Invalid selection"
    fi
}

# Main execution
main() {
    # If no service specified and not aggregating, show interactive selection
    if [[ -z "$SERVICE" && "$AGGREGATE" != "true" ]]; then
        select_service_interactively
    fi
    
    # Show log statistics if verbose
    if [[ -n "$SEARCH_PATTERN" || -n "$LOG_LEVEL" ]]; then
        show_log_statistics
    fi
    
    # Show logs
    if [[ "$AGGREGATE" == "true" ]]; then
        show_all_logs
    elif [[ -n "$SERVICE" ]]; then
        # Verify service exists
        local available_services=($(get_available_services))
        if [[ " ${available_services[@]} " =~ " ${SERVICE} " ]]; then
            show_service_logs "$SERVICE"
        else
            log_error "Service '$SERVICE' not found. Available services: ${available_services[*]}"
        fi
    else
        log_error "No service specified. Use --service or --aggregate"
    fi
    
    # Show save confirmation
    if [[ -n "$OUTPUT_FILE" ]]; then
        log_success "Logs saved to: $OUTPUT_FILE"
    fi
}

# Execute main function
main

# =============================================================================
# scripts/cleanup.sh
#!/bin/bash

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh" 2>/dev/null || {
    log_info() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
    log_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $1"; }
    log_warning() { echo -e "\033[1;33m[WARNING]\033[0m $1"; }
    log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; exit 1; }
    
    load_compose_config() {
        if [[ -f .docker-compose-path ]]; then
            source .docker-compose-path
        else
            if [[ -f "docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="docker-compose.yml"
            elif [[ -f "deployment/docker/database/docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="deployment/docker/database/docker-compose.yml"
            elif [[ -f "deployment/docker/docker-compose.yml" ]]; then
                export DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
            else
                log_error "Could not find docker-compose.yml file"
            fi
            
            if command -v docker-compose &> /dev/null; then
                export COMPOSE_CMD="docker-compose"
            else
                export COMPOSE_CMD="docker compose"
            fi
        fi
    }
}

echo "🧹 Enhanced CSP System Cleanup"
echo "=============================="

# Parse command line arguments
DEEP_CLEAN=false
REMOVE_VOLUMES=false
REMOVE_IMAGES=false
CLEAN_LOGS=false
CLEAN_BACKUPS=false
FORCE=false
DRY_RUN=false
BACKUP_RETENTION_DAYS=30
LOG_RETENTION_DAYS=7

while [[ $# -gt 0 ]]; do
    case $1 in
        --deep|-d)
            DEEP_CLEAN=true
            REMOVE_VOLUMES=true
            REMOVE_IMAGES=true
            CLEAN_LOGS=true
            CLEAN_BACKUPS=true
            shift
            ;;
        --volumes|-v)
            REMOVE_VOLUMES=true
            shift
            ;;
        --images|-i)
            REMOVE_IMAGES=true
            shift
            ;;
        --logs|-l)
            CLEAN_LOGS=true
            shift
            ;;
        --backups|-b)
            CLEAN_BACKUPS=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        --backup-retention)
            BACKUP_RETENTION_DAYS="$2"
            shift 2
            ;;
        --log-retention)
            LOG_RETENTION_DAYS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -d, --deep              Deep clean (everything)"
            echo "  -v, --volumes           Remove Docker volumes (DESTRUCTIVE)"
            echo "  -i, --images            Remove Docker images"
            echo "  -l, --logs              Clean log files"
            echo "  -b, --backups           Clean old backups"
            echo "  -f, --force             Force cleanup without confirmation"
            echo "  -n, --dry-run           Show what would be cleaned without doing it"
            echo "  --backup-retention N    Backup retention in days (default: 30)"
            echo "  --log-retention N       Log retention in days (default: 7)"
            echo "  -h, --help              Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --logs --backups     # Clean logs and old backups"
            echo "  $0 --deep --dry-run     # Show what deep clean would do"
            echo "  $0 --volumes --force    # Remove volumes without confirmation"
            exit 0
            ;;
        *)
            log_warning "Unknown option: $1"
            shift
            ;;
    esac
done

# Load configuration
load_compose_config

# Confirmation for destructive operations
confirm_destructive_action() {
    if [[ "$FORCE" == "true" || "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    echo ""
    log_warning "⚠️  DESTRUCTIVE OPERATION WARNING ⚠️"
    echo ""
    
    if [[ "$REMOVE_VOLUMES" == "true" ]]; then
        echo "🗃️  This will permanently delete ALL database data including:"
        echo "     • PostgreSQL databases (main, AI models, vector)"
        echo "     • Redis cache data"
        echo "     • MongoDB documents"
        echo "     • Elasticsearch indices"
        echo "     • InfluxDB time series data"
        echo "     • Vector database embeddings (Chroma, Qdrant, Weaviate)"
        echo "     • Monitoring and metrics data"
        echo ""
    fi
    
    if [[ "$REMOVE_IMAGES" == "true" ]]; then
        echo "🐳 This will remove Docker images requiring re-download on next start"
        echo ""
    fi
    
    if [[ "$CLEAN_LOGS" == "true" ]]; then
        echo "📝 This will delete log files older than $LOG_RETENTION_DAYS days"
        echo ""
    fi
    
    if [[ "$CLEAN_BACKUPS" == "true" ]]; then
        echo "💾 This will delete backup files older than $BACKUP_RETENTION_DAYS days"
        echo ""
    fi
    
    echo -n "Are you absolutely sure? Type 'YES' to confirm: "
    read -r confirmation
    
    if [[ "$confirmation" != "YES" ]]; then
        log_info "Cleanup cancelled"
        exit 0
    fi
}

# Show current disk usage
show_disk_usage() {
    log_info "Current system disk usage:"
    echo ""
    
    # Docker system usage
    if command -v docker >/dev/null 2>&1; then
        echo "Docker System Usage:"
        docker system df
        echo ""
    fi
    
    # Directory sizes
    echo "Directory Sizes:"
    for dir in "logs" "backups" "data" "."; do
        if [[ -d "$dir" ]]; then
            local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            printf "  %-10s %s\n" "$dir:" "$size"
        fi
    done
    echo ""
}

# Clean Docker containers and networks
cleanup_docker_containers() {
    log_info "Cleaning up Docker containers and networks..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would stop and remove containers"
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps
        return
    fi
    
    # Stop all services
    log_info "Stopping all services..."
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" down --remove-orphans
    
    # Remove stopped containers
    log_info "Removing stopped containers..."
    docker container prune -f >/dev/null 2>&1
    
    # Remove unused networks
    log_info "Removing unused networks..."
    docker network prune -f >/dev/null 2>&1
    
    log_success "Docker containers and networks cleaned"
}

# Clean Docker volumes
cleanup_docker_volumes() {
    if [[ "$REMOVE_VOLUMES" != "true" ]]; then
        return
    fi
    
    log_warning "Cleaning up Docker volumes (this will delete all data)..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would remove the following volumes:"
        docker volume ls --filter label=com.docker.compose.project 2>/dev/null | grep -v DRIVER || echo "No volumes found"
        return
    fi
    
    # Remove compose volumes
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" down --volumes --remove-orphans >/dev/null 2>&1
    
    # Remove all unused volumes
    docker volume prune -f >/dev/null 2>&1
    
    log_warning "All Docker volumes removed - data permanently deleted"
}

# Clean Docker images
cleanup_docker_images() {
    if [[ "$REMOVE_IMAGES" != "true" ]]; then
        return
    fi
    
    log_info "Cleaning up Docker images..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would remove the following images:"
        docker image ls --filter reference="*postgres*" --filter reference="*redis*" --filter reference="*mongo*" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"
        return
    fi
    
    # Remove images used by compose
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" down --rmi all >/dev/null 2>&1
    
    # Remove dangling images
    docker image prune -f >/dev/null 2>&1
    
    # Remove unused images
    if [[ "$DEEP_CLEAN" == "true" ]]; then
        docker image prune -a -f >/dev/null 2>&1
    fi
    
    log_success "Docker images cleaned"
}

# Clean log files
cleanup_logs() {
    if [[ "$CLEAN_LOGS" != "true" ]]; then
        return
    fi
    
    log_info "Cleaning up log files older than $LOG_RETENTION_DAYS days..."
    
    local cleaned_count=0
    
    # Clean application logs
    if [[ -d "logs" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            local log_files=$(find logs/ -name "*.log*" -type f -mtime +$LOG_RETENTION_DAYS 2>/dev/null | wc -l)
            log_info "[DRY RUN] Would remove $log_files log files"
        else
            cleaned_count=$(find logs/ -name "*.log*" -type f -mtime +$LOG_RETENTION_DAYS -delete -print 2>/dev/null | wc -l)
            log_info "Removed $cleaned_count log files"
        fi
    fi
    
    # Clean Docker container logs
    if [[ "$DEEP_CLEAN" == "true" && "$DRY_RUN" != "true" ]]; then
        log_info "Truncating Docker container logs..."
        for container in $(docker ps -a --format '{{.Names}}'); do
            local log_file=$(docker inspect "$container" --format='{{.LogPath}}' 2>/dev/null)
            if [[ -n "$log_file" && -f "$log_file" ]]; then
                truncate -s 0 "$log_file" 2>/dev/null || true
            fi
        done
    fi
    
    log_success "Log cleanup completed"
}

# Clean backup files
cleanup_backups() {
    if [[ "$CLEAN_BACKUPS" != "true" ]]; then
        return
    fi
    
    log_info "Cleaning up backup files older than $BACKUP_RETENTION_DAYS days..."
    
    if [[ ! -d "backups" ]]; then
        log_info "No backups directory found"
        return
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        local backup_count=$(find backups/ -type f -mtime +$BACKUP_RETENTION_DAYS 2>/dev/null | wc -l)
        local backup_dirs=$(find backups/ -type d -mtime +$BACKUP_RETENTION_DAYS 2>/dev/null | wc -l)
        log_info "[DRY RUN] Would remove $backup_count backup files and $backup_dirs backup directories"
    else
        # Remove old backup files
        local file_count=$(find backups/ -type f -mtime +$BACKUP_RETENTION_DAYS -delete -print 2>/dev/null | wc -l)
        
        # Remove empty backup directories
        local dir_count=$(find backups/ -type d -empty -delete -print 2>/dev/null | wc -l)
        
        log_info "Removed $file_count backup files and $dir_count empty directories"
    fi
    
    log_success "Backup cleanup completed"
}

# Clean temporary files
cleanup_temp_files() {
    log_info "Cleaning up temporary files..."
    
    local temp_dirs=("data/temp" "data/cache" "/tmp/csp_*")
    local cleaned_size=0
    
    for temp_pattern in "${temp_dirs[@]}"; do
        if [[ "$DRY_RUN" == "true" ]]; then
            if [[ -d "$temp_pattern" ]]; then
                local size=$(du -sh "$temp_pattern" 2>/dev/null | cut -f1)
                log_info "[DRY RUN] Would clean: $temp_pattern ($size)"
            fi
        else
            if [[ -d "$temp_pattern" ]]; then
                rm -rf "$temp_pattern"/* 2>/dev/null || true
                log_info "Cleaned: $temp_pattern"
            fi
        fi
    done
    
    log_success "Temporary files cleanup completed"
}

# System-wide cleanup
cleanup_system() {
    if [[ "$DEEP_CLEAN" != "true" ]]; then
        return
    fi
    
    log_info "Performing system-wide cleanup..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would perform system cleanup:"
        log_info "  • Docker system prune"
        log_info "  • Clear package manager cache"
        log_info "  • Clean thumbnail cache"
        return
    fi
    
    # Docker system cleanup
    if command -v docker >/dev/null 2>&1; then
        log_info "Running Docker system prune..."
        docker system prune -f --volumes >/dev/null 2>&1
    fi
    
    # Clear package manager cache (if available)
    if command -v apt-get >/dev/null 2>&1; then
        log_info "Cleaning apt cache..."
        sudo apt-get clean >/dev/null 2>&1 || true
    fi
    
    if command -v yum >/dev/null 2>&1; then
        log_info "Cleaning yum cache..."
        sudo yum clean all >/dev/null 2>&1 || true
    fi
    
    # Clear user cache (be careful)
    if [[ -d "$HOME/.cache" ]]; then
        log_info "Cleaning user cache..."
        find "$HOME/.cache" -type f -atime +30 -delete 2>/dev/null || true
    fi
    
    log_success "System cleanup completed"
}

# Generate cleanup report
generate_cleanup_report() {
    local report_file="cleanup_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "Enhanced CSP System Cleanup Report"
        echo "=================================="
        echo "Date: $(date)"
        echo "Type: $([ "$DEEP_CLEAN" == "true" ] && echo "Deep Clean" || echo "Standard Clean")"
        echo "Dry Run: $DRY_RUN"
        echo ""
        
        echo "Cleanup Actions:"
        echo "==============="
        [ "$REMOVE_VOLUMES" == "true" ] && echo "✓ Remove Docker volumes"
        [ "$REMOVE_IMAGES" == "true" ] && echo "✓ Remove Docker images"
        [ "$CLEAN_LOGS" == "true" ] && echo "✓ Clean log files (>${LOG_RETENTION_DAYS} days)"
        [ "$CLEAN_BACKUPS" == "true" ] && echo "✓ Clean backup files (>${BACKUP_RETENTION_DAYS} days)"
        [ "$DEEP_CLEAN" == "true" ] && echo "✓ System-wide cleanup"
        echo ""
        
        echo "System State After Cleanup:"
        echo "=========================="
        docker system df 2>/dev/null || echo "Docker not available"
        echo ""
        
        echo "Directory Sizes:"
        for dir in "logs" "backups" "data"; do
            if [[ -d "$dir" ]]; then
                du -sh "$dir" 2>/dev/null || true
            fi
        done
        
    } > "$report_file"
    
    log_success "Cleanup report generated: $report_file"
}

# Main execution
main() {
    log_info "Starting Enhanced CSP System cleanup..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "🔍 DRY RUN MODE - No changes will be made"
    fi
    
    echo ""
    show_disk_usage
    
    # Confirm destructive operations
    if [[ "$REMOVE_VOLUMES" == "true" || "$REMOVE_IMAGES" == "true" || "$DEEP_CLEAN" == "true" ]]; then
        confirm_destructive_action
    fi
    
    # Perform cleanup operations
    cleanup_docker_containers
    cleanup_docker_volumes
    cleanup_docker_images
    cleanup_logs
    cleanup_backups
    cleanup_temp_files
    cleanup_system
    
    echo ""
    show_disk_usage
    generate_cleanup_report
    
    echo ""
    if [[ "$DRY_RUN" == "true" ]]; then
        log_success "Dry run completed! No changes were made."
    else
        log_success "Cleanup completed successfully!"
        
        if [[ "$REMOVE_VOLUMES" == "true" ]]; then
            echo ""
            log_warning "⚠️  Database volumes were removed"
            echo "You will need to run initial setup on next start:"
            echo "  ./scripts/start-databases.sh"
        fi
    fi
    
    echo ""
    echo "🔧 Available commands:"
    echo "  • Start databases: ./scripts/start-databases.sh"
    echo "  • Start all:       ./scripts/start-all.sh"
    echo "  • Check status:    ./scripts/status.sh"
}

# Execute main function
main