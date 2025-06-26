#!/bin/bash
# =============================================================================
# ADDITIONAL UTILITY SCRIPTS FOR ENHANCED CSP DOCKER MANAGEMENT
# =============================================================================

# scripts/logs.sh - Enhanced log viewing and analysis
#!/bin/bash

# Load common functions
source "$(dirname "$0")/common.sh" 2>/dev/null || true

echo "📋 Enhanced CSP System Logs"
echo "==========================="

# Parse arguments
SERVICE=""
FOLLOW=false
TAIL_LINES=100
SEARCH_PATTERN=""
LOG_LEVEL=""
TIME_FILTER=""

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
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -s, --service SERVICE    Show logs for specific service"
            echo "  -f, --follow            Follow log output"
            echo "  -t, --tail N            Number of lines to show (default: 100)"
            echo "  -g, --search PATTERN    Search for pattern in logs"
            echo "  -l, --level LEVEL       Filter by log level (error, warn, info, debug)"
            echo "  --since TIME            Show logs since time (e.g., '1h', '30m')"
            echo "  -h, --help              Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 -s postgres -f       # Follow PostgreSQL logs"
            echo "  $0 -g 'ERROR' -t 50     # Search for errors in last 50 lines"
            echo "  $0 --since '1h'         # Show logs from last hour"
            exit 0
            ;;
        *)
            log_warning "Unknown option: $1"
            shift
            ;;
    esac
done

# Load docker-compose path
load_compose_config

# Available services
AVAILABLE_SERVICES=(
    "postgres" "postgres_ai_models" "postgres_vector"
    "redis" "mongodb" "elasticsearch" "influxdb"
    "chroma" "qdrant" "weaviate" "milvus-standalone"
    "pgadmin" "redis-insight" "mongo-express"
    "prometheus" "grafana"
)

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
    
    # Show logs
    if [[ -n "$SEARCH_PATTERN" ]]; then
        log_info "Showing logs for $service (searching for: $SEARCH_PATTERN)"
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs $cmd_args "$service" | grep -i "$SEARCH_PATTERN"
    elif [[ -n "$LOG_LEVEL" ]]; then
        log_info "Showing logs for $service (level: $LOG_LEVEL)"
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs $cmd_args "$service" | grep -i "$LOG_LEVEL"
    else
        log_info "Showing logs for $service"
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs $cmd_args "$service"
    fi
}

show_all_logs() {
    log_info "Showing logs for all services"
    
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
    
    if [[ -n "$SEARCH_PATTERN" ]]; then
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs $cmd_args | grep -i "$SEARCH_PATTERN"
    elif [[ -n "$LOG_LEVEL" ]]; then
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs $cmd_args | grep -i "$LOG_LEVEL"
    else
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" logs $cmd_args
    fi
}

# Main execution
if [[ -n "$SERVICE" ]]; then
    if [[ " ${AVAILABLE_SERVICES[@]} " =~ " ${SERVICE} " ]]; then
        show_service_logs "$SERVICE"
    else
        log_error "Service '$SERVICE' not found. Available services: ${AVAILABLE_SERVICES[*]}"
    fi
else
    show_all_logs
fi

# =============================================================================
# scripts/cleanup.sh - System cleanup and maintenance
#!/bin/bash

source "$(dirname "$0")/common.sh" 2>/dev/null || true

echo "🧹 CSP System Cleanup & Maintenance"
echo "==================================="

# Parse arguments
FORCE=false
DEEP_CLEAN=false
PRUNE_VOLUMES=false
CLEAN_LOGS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE=true
            shift
            ;;
        --deep)
            DEEP_CLEAN=true
            shift
            ;;
        --volumes|-v)
            PRUNE_VOLUMES=true
            shift
            ;;
        --logs|-l)
            CLEAN_LOGS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -f, --force     Force cleanup without confirmation"
            echo "  --deep          Deep cleaning (removes everything)"
            echo "  -v, --volumes   Also remove Docker volumes"
            echo "  -l, --logs      Clean log files"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *)
            log_warning "Unknown option: $1"
            shift
            ;;
    esac
done

load_compose_config

confirm_action() {
    if [[ "$FORCE" == "true" ]]; then
        return 0
    fi
    
    echo -n "Are you sure? [y/N]: "
    read -r response
    case $response in
        [yY]|[yY][eE][sS])
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

cleanup_docker() {
    log_info "Cleaning up Docker resources..."
    
    # Stop all containers
    log_info "Stopping all CSP containers..."
    $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" down
    
    # Remove stopped containers
    log_info "Removing stopped containers..."
    docker container prune -f
    
    # Remove unused images
    log_info "Removing unused Docker images..."
    docker image prune -f
    
    # Remove unused networks
    log_info "Removing unused networks..."
    docker network prune -f
    
    if [[ "$PRUNE_VOLUMES" == "true" ]]; then
        log_warning "Removing Docker volumes (this will delete all data!)"
        if confirm_action; then
            docker volume prune -f
        fi
    fi
    
    if [[ "$DEEP_CLEAN" == "true" ]]; then
        log_warning "Deep cleaning - removing all unused Docker resources"
        if confirm_action; then
            docker system prune -a -f --volumes
        fi
    fi
    
    log_success "Docker cleanup completed"
}

cleanup_logs() {
    log_info "Cleaning up log files..."
    
    # Clean application logs
    if [[ -d "logs" ]]; then
        find logs/ -name "*.log" -type f -mtime +7 -delete 2>/dev/null || true
        find logs/ -name "*.log.*" -type f -mtime +7 -delete 2>/dev/null || true
    fi
    
    # Clean Docker logs
    docker system events --since 24h --until 1h >/dev/null 2>&1 || true
    
    log_success "Log cleanup completed"
}

cleanup_temp_files() {
    log_info "Cleaning up temporary files..."
    
    # Clean temp directories
    [[ -d "data/temp" ]] && rm -rf data/temp/* 2>/dev/null || true
    [[ -d "data/cache" ]] && rm -rf data/cache/* 2>/dev/null || true
    
    # Clean backup files older than 30 days
    if [[ -d "backups" ]]; then
        find backups/ -name "*.sql.gz" -type f -mtime +30 -delete 2>/dev/null || true
        find backups/ -name "*.tar.gz" -type f -mtime +30 -delete 2>/dev/null || true
    fi
    
    log_success "Temporary files cleanup completed"
}

show_disk_usage() {
    log_info "Current disk usage:"
    echo ""
    
    # Show Docker space usage
    docker system df
    echo ""
    
    # Show directory sizes
    if command -v du &> /dev/null; then
        echo "Directory sizes:"
        du -sh data/ backups/ logs/ 2>/dev/null || true
    fi
}

# Main execution
log_info "Starting cleanup process..."

if [[ "$CLEAN_LOGS" == "true" ]]; then
    cleanup_logs
fi

cleanup_temp_files
cleanup_docker

if [[ "$CLEAN_LOGS" == "true" ]]; then
    cleanup_logs
fi

echo ""
show_disk_usage
echo ""
log_success "Cleanup process completed!"

# =============================================================================
# scripts/migrate.sh - Database migration management
#!/bin/bash

source "$(dirname "$0")/common.sh" 2>/dev/null || true

echo "🗃️ CSP Database Migration Manager"
echo "================================="

# Parse arguments
ACTION=""
TARGET_VERSION=""
DATABASE="main"

while [[ $# -gt 0 ]]; do
    case $1 in
        upgrade|downgrade|current|history|reset)
            ACTION="$1"
            shift
            ;;
        --version|-v)
            TARGET_VERSION="$2"
            shift 2
            ;;
        --database|-d)
            DATABASE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 ACTION [OPTIONS]"
            echo ""
            echo "Actions:"
            echo "  upgrade     Upgrade to latest or specific version"
            echo "  downgrade   Downgrade to specific version"
            echo "  current     Show current migration version"
            echo "  history     Show migration history"
            echo "  reset       Reset database (WARNING: destructive)"
            echo ""
            echo "Options:"
            echo "  -v, --version VERSION   Target version for upgrade/downgrade"
            echo "  -d, --database DB       Database to migrate (main|ai_models|vector)"
            echo "  -h, --help              Show this help"
            exit 0
            ;;
        *)
            log_warning "Unknown option: $1"
            shift
            ;;
    esac
done

if [[ -z "$ACTION" ]]; then
    log_error "No action specified. Use --help for usage information."
fi

load_compose_config

# Database connection settings
get_db_connection() {
    case "$DATABASE" in
        "main")
            echo "postgresql://csp_user:csp_password@localhost:5432/csp_visual_designer"
            ;;
        "ai_models")
            echo "postgresql://ai_models_user:ai_models_password@localhost:5433/ai_models_db"
            ;;
        "vector")
            echo "postgresql://vector_user:vector_password@localhost:5434/vector_db"
            ;;
        *)
            log_error "Unknown database: $DATABASE"
            ;;
    esac
}

run_migration() {
    local db_url=$(get_db_connection)
    local migration_dir="database/migrations/${DATABASE}"
    
    case "$ACTION" in
        "upgrade")
            log_info "Upgrading $DATABASE database..."
            if [[ -n "$TARGET_VERSION" ]]; then
                alembic -c "$migration_dir/alembic.ini" upgrade "$TARGET_VERSION"
            else
                alembic -c "$migration_dir/alembic.ini" upgrade head
            fi
            ;;
        "downgrade")
            if [[ -z "$TARGET_VERSION" ]]; then
                log_error "Downgrade requires --version parameter"
            fi
            log_warning "Downgrading $DATABASE database to version $TARGET_VERSION"
            alembic -c "$migration_dir/alembic.ini" downgrade "$TARGET_VERSION"
            ;;
        "current")
            log_info "Current $DATABASE database version:"
            alembic -c "$migration_dir/alembic.ini" current
            ;;
        "history")
            log_info "Migration history for $DATABASE database:"
            alembic -c "$migration_dir/alembic.ini" history
            ;;
        "reset")
            log_warning "This will completely reset the $DATABASE database!"
            echo -n "Are you sure? Type 'YES' to confirm: "
            read -r confirmation
            if [[ "$confirmation" == "YES" ]]; then
                alembic -c "$migration_dir/alembic.ini" downgrade base
                alembic -c "$migration_dir/alembic.ini" upgrade head
                log_success "Database reset completed"
            else
                log_info "Reset cancelled"
            fi
            ;;
    esac
}

# Check if migration directory exists
migration_dir="database/migrations/${DATABASE}"
if [[ ! -d "$migration_dir" ]]; then
    log_error "Migration directory not found: $migration_dir"
fi

# Check if database is accessible
log_info "Checking database connectivity..."
case "$DATABASE" in
    "main")
        if ! $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" exec postgres pg_isready -U csp_user -d csp_visual_designer >/dev/null 2>&1; then
            log_error "Main database is not accessible"
        fi
        ;;
    "ai_models")
        if ! $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" exec postgres_ai_models pg_isready -U ai_models_user -d ai_models_db >/dev/null 2>&1; then
            log_error "AI models database is not accessible"
        fi
        ;;
    "vector")
        if ! $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" exec postgres_vector pg_isready -U vector_user -d vector_db >/dev/null 2>&1; then
            log_error "Vector database is not accessible"
        fi
        ;;
esac

run_migration
log_success "Migration completed for $DATABASE database"

# =============================================================================
# scripts/performance.sh - Performance monitoring and optimization
#!/bin/bash

source "$(dirname "$0")/common.sh" 2>/dev/null || true

echo "⚡ CSP Performance Monitor"
echo "========================="

# Parse arguments
MONITOR_TIME=60
OUTPUT_FORMAT="table"
SERVICES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --time|-t)
            MONITOR_TIME="$2"
            shift 2
            ;;
        --format|-f)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --services|-s)
            SERVICES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --time SECONDS     Monitoring duration (default: 60)"
            echo "  -f, --format FORMAT    Output format (table|json|csv)"
            echo "  -s, --services LIST    Comma-separated list of services to monitor"
            echo "  -h, --help             Show this help"
            exit 0
            ;;
        *)
            log_warning "Unknown option: $1"
            shift
            ;;
    esac
done

load_compose_config

show_system_resources() {
    log_info "System Resources Overview"
    echo "========================="
    
    # CPU usage
    if command -v top &> /dev/null; then
        echo "CPU Usage:"
        top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | head -1
    fi
    
    # Memory usage
    if command -v free &> /dev/null; then
        echo ""
        echo "Memory Usage:"
        free -h
    fi
    
    # Disk usage
    echo ""
    echo "Disk Usage:"
    df -h | grep -E "(Filesystem|/dev/)"
    
    echo ""
}

monitor_docker_containers() {
    log_info "Docker Container Performance"
    echo "============================"
    
    # Get container stats
    if [[ -n "$SERVICES" ]]; then
        local service_list=$(echo "$SERVICES" | tr ',' ' ')
        for service in $service_list; do
            local container_id=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps -q "$service" 2>/dev/null)
            if [[ -n "$container_id" ]]; then
                echo "Service: $service"
                docker stats --no-stream "$container_id"
                echo ""
            fi
        done
    else
        echo "All CSP containers:"
        docker stats --no-stream $($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps -q 2>/dev/null)
    fi
}

monitor_database_performance() {
    log_info "Database Performance Metrics"
    echo "============================"
    
    # PostgreSQL main database
    if $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" exec postgres pg_isready -U csp_user -d csp_visual_designer >/dev/null 2>&1; then
        echo "PostgreSQL Main Database:"
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" exec postgres psql -U csp_user -d csp_visual_designer -c "
            SELECT 
                datname,
                numbackends as connections,
                xact_commit as commits,
                xact_rollback as rollbacks,
                blks_read,
                blks_hit,
                round((blks_hit::float/(blks_read+blks_hit+1)*100)::numeric, 2) as cache_hit_ratio
            FROM pg_stat_database 
            WHERE datname = 'csp_visual_designer';
        " 2>/dev/null || echo "Could not fetch PostgreSQL stats"
        echo ""
    fi
    
    # Redis performance
    if $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" exec redis redis-cli ping >/dev/null 2>&1; then
        echo "Redis Performance:"
        $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" exec redis redis-cli info stats | grep -E "(keyspace_hits|keyspace_misses|total_commands_processed)" || echo "Could not fetch Redis stats"
        echo ""
    fi
}

monitor_vector_databases() {
    log_info "Vector Database Performance"
    echo "==========================="
    
    # Chroma
    if curl -s http://localhost:8200/api/v1/heartbeat >/dev/null 2>&1; then
        echo "Chroma Status: ✅ Running"
        # Add Chroma-specific metrics if available
    else
        echo "Chroma Status: ❌ Not accessible"
    fi
    
    # Qdrant
    if curl -s http://localhost:6333/health >/dev/null 2>&1; then
        echo "Qdrant Status: ✅ Running"
        # Add Qdrant-specific metrics
        curl -s http://localhost:6333/metrics 2>/dev/null | head -10 || true
    else
        echo "Qdrant Status: ❌ Not accessible"
    fi
    
    # Weaviate
    if curl -s http://localhost:8080/v1/.well-known/ready >/dev/null 2>&1; then
        echo "Weaviate Status: ✅ Running"
    else
        echo "Weaviate Status: ❌ Not accessible"
    fi
    
    echo ""
}

generate_performance_report() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local report_file="performance_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "CSP System Performance Report"
        echo "Generated: $timestamp"
        echo "=============================="
        echo ""
        
        show_system_resources
        monitor_docker_containers
        monitor_database_performance
        monitor_vector_databases
        
    } > "$report_file"
    
    log_success "Performance report saved to: $report_file"
}

# Main execution
show_system_resources
monitor_docker_containers

if [[ "$MONITOR_TIME" -gt 5 ]]; then
    log_info "Monitoring containers for $MONITOR_TIME seconds..."
    docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" $($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps -q) &
    STATS_PID=$!
    sleep "$MONITOR_TIME"
    kill $STATS_PID 2>/dev/null || true
fi

monitor_database_performance
monitor_vector_databases

if [[ "$OUTPUT_FORMAT" == "json" || "$OUTPUT_FORMAT" == "csv" ]]; then
    generate_performance_report
fi

log_success "Performance monitoring completed"

# =============================================================================
# scripts/common.sh - Common functions and utilities
#!/bin/bash

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
log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
log_debug() { echo -e "${PURPLE}[DEBUG]${NC} $1"; }

# Load Docker Compose configuration
load_compose_config() {
    if [[ -f .docker-compose-path ]]; then
        source .docker-compose-path
    else
        # Fallback detection
        if [[ -f "docker-compose.yml" ]]; then
            export DOCKER_COMPOSE_FILE="docker-compose.yml"
        elif [[ -f "deployment/docker/database/docker-compose.yml" ]]; then
            export DOCKER_COMPOSE_FILE="deployment/docker/database/docker-compose.yml"
        else
            log_error "Could not find docker-compose.yml file"
        fi
        
        if command -v docker-compose &> /dev/null; then
            export COMPOSE_CMD="docker-compose"
        else
            export COMPOSE_CMD="docker compose"
        fi
    fi
    
    if [[ -z "$DOCKER_COMPOSE_FILE" ]]; then
        log_error "DOCKER_COMPOSE_FILE not set"
    fi
    
    if [[ -z "$COMPOSE_CMD" ]]; then
        log_error "COMPOSE_CMD not set"
    fi
}

# Check if service is running
is_service_running() {
    local service="$1"
    local container_id=$($COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" ps -q "$service" 2>/dev/null)
    [[ -n "$container_id" ]] && docker inspect "$container_id" | grep -q '"Running": true'
}

# Wait for service to be ready
wait_for_service() {
    local service="$1"
    local health_check="$2"
    local max_attempts="${3:-30}"
    local attempt=1
    
    log_info "Waiting for $service to be ready..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if $COMPOSE_CMD -f "$DOCKER_COMPOSE_FILE" exec -T "$service" $health_check >/dev/null 2>&1; then
            log_success "$service is ready"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    log_warning "$service not ready after $max_attempts attempts"
    return 1
}