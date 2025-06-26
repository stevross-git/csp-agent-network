#!/bin/bash
# =============================================================================
# FIXED DOCKER MANAGEMENT SCRIPTS FOR YOUR DIRECTORY STRUCTURE
# =============================================================================

# scripts/setup-docker.sh (Fixed)
#!/bin/bash

echo "🚀 Setting up Enhanced CSP Docker Environment"
echo "=============================================="

# Detect docker-compose file location
DOCKER_COMPOSE_FILE=""
if [ -f "docker-compose.yml" ]; then
    DOCKER_COMPOSE_FILE="docker-compose.yml"
elif [ -f "deployment/docker/database/docker-compose.yml" ]; then
    DOCKER_COMPOSE_FILE="deployment/docker/database/docker-compose.yml"
    echo "📍 Found docker-compose.yml at: $DOCKER_COMPOSE_FILE"
elif [ -f "deployment/docker/docker-compose.yml" ]; then
    DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
    echo "📍 Found docker-compose.yml at: $DOCKER_COMPOSE_FILE"
else
    echo "❌ Could not find docker-compose.yml file"
    echo "Please ensure docker-compose.yml exists in one of these locations:"
    echo "  - ./docker-compose.yml"
    echo "  - ./deployment/docker/database/docker-compose.yml"
    echo "  - ./deployment/docker/docker-compose.yml"
    exit 1
fi

# Export for other scripts
export DOCKER_COMPOSE_FILE
echo "export DOCKER_COMPOSE_FILE='$DOCKER_COMPOSE_FILE'" > .docker-compose-path

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p database/{init,ai_models_init,mongodb/init,redis,pgadmin,scripts,backups,chroma/init,qdrant/config,pgvector/init,vector-admin}
mkdir -p monitoring/{prometheus,grafana/{dashboards,datasources}}
mkdir -p logs
mkdir -p data
mkdir -p backups

# Create backup script
echo "📝 Creating backup script..."
cat > database/scripts/backup.sh << 'EOF'
#!/bin/bash
# Database backup script for Enhanced CSP Project
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
echo "Starting database backups at $(date)"
mkdir -p "$BACKUP_DIR"

# Backup main PostgreSQL database
echo "Backing up main PostgreSQL database..."
pg_dump -h postgres -U csp_user -d csp_visual_designer > "$BACKUP_DIR/main_db_$DATE.sql"
if [ $? -eq 0 ]; then
    echo "✅ Main database backup completed"
    gzip "$BACKUP_DIR/main_db_$DATE.sql"
else
    echo "❌ Main database backup failed"
fi

echo "Backup process completed at $(date)"
EOF

# Set permissions
chmod +x database/scripts/backup.sh

# Create environment file if it doesn't exist
if [ ! -f .env.docker ]; then
    echo "📝 Creating .env.docker file..."
    cat > .env.docker << 'EOL'
# Enhanced CSP Docker Environment Variables

# Main Database
DB_HOST=postgres
DB_PORT=5432
DB_NAME=csp_visual_designer
DB_USER=csp_user
DB_PASSWORD=csp_strong_password_123

# AI Models Database  
AI_MODELS_DB_HOST=postgres_ai_models
AI_MODELS_DB_PORT=5432
AI_MODELS_DB_NAME=ai_models_db
AI_MODELS_DB_USER=ai_models_user
AI_MODELS_DB_PASSWORD=ai_models_strong_password_123

# Vector Database
VECTOR_DB_HOST=postgres_vector
VECTOR_DB_PORT=5432
VECTOR_DB_NAME=vector_db
VECTOR_DB_USER=vector_user
VECTOR_DB_PASSWORD=vector_strong_password_123

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=redis_strong_password_123

# MongoDB
MONGODB_HOST=mongodb
MONGODB_PORT=27017
MONGODB_USERNAME=csp_mongo_user
MONGODB_PASSWORD=mongo_strong_password_123
MONGODB_DATABASE=csp_documents

# Vector Databases
CHROMA_HOST=chroma
CHROMA_PORT=8200
QDRANT_HOST=qdrant
QDRANT_PORT=6333
WEAVIATE_HOST=weaviate
WEAVIATE_PORT=8080

# Security
SECRET_KEY=your_super_secret_key_change_in_production
JWT_SECRET_KEY=your_jwt_secret_key_change_in_production

# External APIs (add your keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Monitoring
GRAFANA_ADMIN_PASSWORD=grafana_admin_123
PROMETHEUS_RETENTION=30d
EOL
    echo "✅ Created .env.docker - Please update with your API keys!"
else
    echo "✅ .env.docker already exists"
fi

# Create Prometheus configuration
echo "📊 Creating monitoring configuration..."
mkdir -p monitoring/prometheus
cat > monitoring/prometheus/prometheus.yml << 'EOL'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'csp-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'chroma'
    static_configs:
      - targets: ['chroma:8200']
    metrics_path: '/api/v1/heartbeat'

  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant:6333']
    metrics_path: '/health'
EOL

# Create Grafana datasource configuration
mkdir -p monitoring/grafana/datasources
cat > monitoring/grafana/datasources/prometheus.yml << 'EOL'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOL

echo "✅ Docker environment setup completed!"
echo ""
echo "📍 Using docker-compose file: $DOCKER_COMPOSE_FILE"
echo ""
echo "Next steps:"
echo "1. Update .env.docker with your API keys"
echo "2. Run: ./scripts/start-databases.sh"
echo "3. Run: ./scripts/start-all.sh"

# scripts/start-databases.sh (Fixed)
#!/bin/bash

echo "🗄️ Starting Enhanced CSP Databases"
echo "=================================="

# Load docker-compose file path
if [ -f .docker-compose-path ]; then
    source .docker-compose-path
fi

# If not set, try to detect
if [ -z "$DOCKER_COMPOSE_FILE" ]; then
    if [ -f "docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="docker-compose.yml"
    elif [ -f "deployment/docker/database/docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="deployment/docker/database/docker-compose.yml"
    elif [ -f "deployment/docker/docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
    else
        echo "❌ Could not find docker-compose.yml file"
        exit 1
    fi
fi

echo "📍 Using docker-compose file: $DOCKER_COMPOSE_FILE"

# Load environment variables
if [ -f .env.docker ]; then
    export $(cat .env.docker | grep -v '^#' | xargs)
fi

# Start core databases first
echo "🚀 Starting core databases (PostgreSQL, Redis)..."
docker-compose -f "$DOCKER_COMPOSE_FILE" up -d postgres redis postgres_ai_models postgres_vector

# Wait for databases to be ready
echo "⏳ Waiting for databases to be ready..."
sleep 15

# Check database health
echo "🔍 Checking database health..."
docker-compose -f "$DOCKER_COMPOSE_FILE" exec postgres pg_isready -U csp_user -d csp_visual_designer || echo "Main DB not ready yet"
docker-compose -f "$DOCKER_COMPOSE_FILE" exec redis redis-cli ping || echo "Redis not ready yet"

# Start vector databases
echo "🧠 Starting vector databases (Chroma, Qdrant, Weaviate)..."
docker-compose -f "$DOCKER_COMPOSE_FILE" up -d chroma qdrant weaviate

# Start optional databases
echo "🚀 Starting optional databases (MongoDB, Elasticsearch, InfluxDB)..."
docker-compose -f "$DOCKER_COMPOSE_FILE" up -d mongodb elasticsearch influxdb

# Start Milvus if profile is enabled
echo "🔮 Starting Milvus (if enabled)..."
docker-compose -f "$DOCKER_COMPOSE_FILE" --profile milvus up -d etcd minio milvus-standalone || echo "Milvus profile not enabled"

# Start admin tools
echo "🛠️ Starting admin tools..."
docker-compose -f "$DOCKER_COMPOSE_FILE" up -d pgadmin redis-insight mongo-express vector-admin

echo "✅ All databases started!"
echo ""
echo "📊 Access Points:"
echo "- PostgreSQL: localhost:5432"
echo "- AI Models DB: localhost:5433" 
echo "- Vector DB (pgvector): localhost:5434"
echo "- Redis: localhost:6379"
echo "- MongoDB: localhost:27017"
echo "- Elasticsearch: localhost:9200"
echo "- InfluxDB: localhost:8086"
echo ""
echo "🧠 Vector Databases:"
echo "- Chroma: localhost:8200"
echo "- Qdrant: localhost:6333"
echo "- Weaviate: localhost:8080"
echo "- Milvus: localhost:19530"
echo ""
echo "🛠️ Admin Tools:"
echo "- pgAdmin: http://localhost:5050"
echo "- Redis Insight: http://localhost:8001"
echo "- Mongo Express: http://localhost:8081"
echo "- Vector Admin: http://localhost:3001"

# scripts/start-all.sh (Fixed)
#!/bin/bash

echo "🚀 Starting Complete Enhanced CSP System"
echo "========================================"

# Load docker-compose file path
if [ -f .docker-compose-path ]; then
    source .docker-compose-path
fi

# If not set, try to detect
if [ -z "$DOCKER_COMPOSE_FILE" ]; then
    if [ -f "docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="docker-compose.yml"
    elif [ -f "deployment/docker/database/docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="deployment/docker/database/docker-compose.yml"
    elif [ -f "deployment/docker/docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
    else
        echo "❌ Could not find docker-compose.yml file"
        exit 1
    fi
fi

echo "📍 Using docker-compose file: $DOCKER_COMPOSE_FILE"

# Load environment variables
if [ -f .env.docker ]; then
    export $(cat .env.docker | grep -v '^#' | xargs)
fi

# Start all services
echo "🚀 Starting all services..."
docker-compose -f "$DOCKER_COMPOSE_FILE" up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 20

# Check service health
echo "🔍 Checking service health..."
docker-compose -f "$DOCKER_COMPOSE_FILE" ps

echo "✅ All services started!"
echo ""
echo "🌐 Access Points:"
echo "- Main Application: http://localhost:8000"
echo "- pgAdmin: http://localhost:5050 (admin@csp.com / pgadmin_password)"
echo "- Redis Insight: http://localhost:8001"
echo "- Mongo Express: http://localhost:8081"
echo "- Vector Admin: http://localhost:3001"
echo "- Grafana: http://localhost:3000 (admin / grafana_admin_123)"
echo "- Prometheus: http://localhost:9090"

# scripts/stop-all.sh (Fixed)
#!/bin/bash

echo "🛑 Stopping Enhanced CSP System"
echo "==============================="

# Load docker-compose file path
if [ -f .docker-compose-path ]; then
    source .docker-compose-path
fi

if [ -z "$DOCKER_COMPOSE_FILE" ]; then
    if [ -f "docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="docker-compose.yml"
    elif [ -f "deployment/docker/database/docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="deployment/docker/database/docker-compose.yml"
    elif [ -f "deployment/docker/docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
    else
        echo "❌ Could not find docker-compose.yml file"
        exit 1
    fi
fi

# Stop all services
docker-compose -f "$DOCKER_COMPOSE_FILE" down

echo "✅ All services stopped!"

# scripts/stop-databases.sh (Fixed)
#!/bin/bash

echo "🛑 Stopping Enhanced CSP Databases"
echo "=================================="

# Load docker-compose file path
if [ -f .docker-compose-path ]; then
    source .docker-compose-path
fi

if [ -z "$DOCKER_COMPOSE_FILE" ]; then
    if [ -f "docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="docker-compose.yml"
    elif [ -f "deployment/docker/database/docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="deployment/docker/database/docker-compose.yml"
    elif [ -f "deployment/docker/docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
    else
        echo "❌ Could not find docker-compose.yml file"
        exit 1
    fi
fi

# Stop only database services
docker-compose -f "$DOCKER_COMPOSE_FILE" stop postgres redis postgres_ai_models postgres_vector mongodb elasticsearch influxdb chroma qdrant weaviate
docker-compose -f "$DOCKER_COMPOSE_FILE" stop pgadmin redis-insight mongo-express vector-admin

echo "✅ All databases stopped!"

# scripts/restart-databases.sh (Fixed)
#!/bin/bash

echo "🔄 Restarting Enhanced CSP Databases"
echo "===================================="

# Load docker-compose file path
if [ -f .docker-compose-path ]; then
    source .docker-compose-path
fi

if [ -z "$DOCKER_COMPOSE_FILE" ]; then
    if [ -f "docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="docker-compose.yml"
    elif [ -f "deployment/docker/database/docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="deployment/docker/database/docker-compose.yml"
    elif [ -f "deployment/docker/docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
    else
        echo "❌ Could not find docker-compose.yml file"
        exit 1
    fi
fi

# Restart core databases
docker-compose -f "$DOCKER_COMPOSE_FILE" restart postgres redis postgres_ai_models postgres_vector

# Wait for restart
sleep 10

# Restart vector databases
docker-compose -f "$DOCKER_COMPOSE_FILE" restart chroma qdrant weaviate

# Restart optional databases
docker-compose -f "$DOCKER_COMPOSE_FILE" restart mongodb elasticsearch influxdb

# Restart admin tools  
docker-compose -f "$DOCKER_COMPOSE_FILE" restart pgadmin redis-insight mongo-express vector-admin

echo "✅ All databases restarted!"

# scripts/backup-all.sh (Fixed)
#!/bin/bash

echo "💾 Backing up Enhanced CSP Databases"
echo "===================================="

# Load docker-compose file path
if [ -f .docker-compose-path ]; then
    source .docker-compose-path
fi

if [ -z "$DOCKER_COMPOSE_FILE" ]; then
    if [ -f "docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="docker-compose.yml"
    elif [ -f "deployment/docker/database/docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="deployment/docker/database/docker-compose.yml"
    elif [ -f "deployment/docker/docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
    else
        echo "❌ Could not find docker-compose.yml file"
        exit 1
    fi
fi

# Create backup directory with timestamp
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📁 Backup directory: $BACKUP_DIR"

# Check if containers are running
POSTGRES_CONTAINER=$(docker-compose -f "$DOCKER_COMPOSE_FILE" ps -q postgres)
REDIS_CONTAINER=$(docker-compose -f "$DOCKER_COMPOSE_FILE" ps -q redis)
MONGODB_CONTAINER=$(docker-compose -f "$DOCKER_COMPOSE_FILE" ps -q mongodb)

# Backup PostgreSQL main database
if [ ! -z "$POSTGRES_CONTAINER" ]; then
    echo "🗄️ Backing up main PostgreSQL database..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_dump -U csp_user csp_visual_designer > "$BACKUP_DIR/main_db.sql" 2>/dev/null || echo "Main DB backup failed"
else
    echo "⚠️ PostgreSQL container not running, skipping main DB backup"
fi

# Backup AI models database
AI_MODELS_CONTAINER=$(docker-compose -f "$DOCKER_COMPOSE_FILE" ps -q postgres_ai_models)
if [ ! -z "$AI_MODELS_CONTAINER" ]; then
    echo "🧠 Backing up AI models database..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres_ai_models pg_dump -U ai_models_user ai_models_db > "$BACKUP_DIR/ai_models_db.sql" 2>/dev/null || echo "AI Models DB backup failed"
else
    echo "⚠️ AI Models PostgreSQL container not running, skipping backup"
fi

# Backup Vector database
VECTOR_CONTAINER=$(docker-compose -f "$DOCKER_COMPOSE_FILE" ps -q postgres_vector)
if [ ! -z "$VECTOR_CONTAINER" ]; then
    echo "🧠 Backing up Vector database..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres_vector pg_dump -U vector_user vector_db > "$BACKUP_DIR/vector_db.sql" 2>/dev/null || echo "Vector DB backup failed"
else
    echo "⚠️ Vector PostgreSQL container not running, skipping backup"
fi

# Backup MongoDB
if [ ! -z "$MONGODB_CONTAINER" ]; then
    echo "🍃 Backing up MongoDB..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T mongodb mongodump --host localhost --port 27017 --out /tmp/mongodb_backup 2>/dev/null || echo "MongoDB backup failed"
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T mongodb tar -czf /tmp/mongodb_backup.tar.gz -C /tmp mongodb_backup 2>/dev/null
    docker cp "$MONGODB_CONTAINER":/tmp/mongodb_backup.tar.gz "$BACKUP_DIR/" 2>/dev/null || echo "MongoDB backup copy failed"
else
    echo "⚠️ MongoDB container not running, skipping backup"
fi

# Backup Redis
if [ ! -z "$REDIS_CONTAINER" ]; then
    echo "🔄 Backing up Redis..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli --rdb /tmp/redis_backup.rdb 2>/dev/null || echo "Redis backup failed"
    docker cp "$REDIS_CONTAINER":/tmp/redis_backup.rdb "$BACKUP_DIR/" 2>/dev/null || echo "Redis backup copy failed"
else
    echo "⚠️ Redis container not running, skipping backup"
fi

# Compress backups
echo "🗜️ Compressing backups..."
cd "$BACKUP_DIR"
for file in *.sql; do
    if [ -f "$file" ]; then
        gzip "$file"
    fi
done

for file in *.rdb; do
    if [ -f "$file" ]; then
        gzip "$file"
    fi
done

echo "✅ Backup completed in: $BACKUP_DIR"

# scripts/status.sh (Fixed)
#!/bin/bash

echo "📊 Enhanced CSP System Status"
echo "============================="

# Load docker-compose file path
if [ -f .docker-compose-path ]; then
    source .docker-compose-path
fi

if [ -z "$DOCKER_COMPOSE_FILE" ]; then
    if [ -f "docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="docker-compose.yml"
    elif [ -f "deployment/docker/database/docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="deployment/docker/database/docker-compose.yml"
    elif [ -f "deployment/docker/docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"
    else
        echo "❌ Could not find docker-compose.yml file"
        exit 1
    fi
fi

echo "📍 Using docker-compose file: $DOCKER_COMPOSE_FILE"

# Show container status
echo ""
echo "🐳 Container Status:"
docker-compose -f "$DOCKER_COMPOSE_FILE" ps

echo ""
echo "💾 Database Status:"

# Check PostgreSQL
echo -n "PostgreSQL (Main): "
if docker-compose -f "$DOCKER_COMPOSE_FILE" exec postgres pg_isready -U csp_user -d csp_visual_designer >/dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not Ready"
fi

# Check AI Models Database
echo -n "PostgreSQL (AI Models): "
if docker-compose -f "$DOCKER_COMPOSE_FILE" exec postgres_ai_models pg_isready -U ai_models_user -d ai_models_db >/dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not Ready"
fi

# Check Vector Database
echo -n "PostgreSQL (Vector): "
if docker-compose -f "$DOCKER_COMPOSE_FILE" exec postgres_vector pg_isready -U vector_user -d vector_db >/dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not Ready"
fi

# Check Redis
echo -n "Redis: "
if docker-compose -f "$DOCKER_COMPOSE_FILE" exec redis redis-cli ping >/dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not Ready"
fi

# Check MongoDB
echo -n "MongoDB: "
if docker-compose -f "$DOCKER_COMPOSE_FILE" exec mongodb mongosh --quiet --eval "db.adminCommand('ping')" >/dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not Ready"
fi

echo ""
echo "🧠 Vector Database Status:"

# Check Chroma
echo -n "Chroma: "
if curl -s http://localhost:8200/api/v1/heartbeat >/dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not Ready"
fi

# Check Qdrant
echo -n "Qdrant: "
if curl -s http://localhost:6333/health >/dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not Ready"
fi

# Check Weaviate
echo -n "Weaviate: "
if curl -s http://localhost:8080/v1/.well-known/ready >/dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not Ready"
fi

echo ""
echo "🌐 Web Interfaces:"
echo "- Main App: http://localhost:8000"
echo "- pgAdmin: http://localhost:5050"
echo "- Redis Insight: http://localhost:8001"
echo "- Mongo Express: http://localhost:8081"
echo "- Vector Admin: http://localhost:3001"
echo "- Grafana: http://localhost:3000"
echo ""
echo "🧠 Vector Database APIs:"
echo "- Chroma API: http://localhost:8200"
echo "- Qdrant API: http://localhost:6333"
echo "- Weaviate API: http://localhost:8080"

# Make all scripts executable
chmod +x scripts/*.sh

echo ""
echo "🎉 All Docker management scripts have been fixed!"
echo ""
echo "📋 Updated Scripts:"
echo "- ./scripts/setup-docker.sh     - Initial setup (FIXED)"
echo "- ./scripts/start-databases.sh  - Start databases only (FIXED)" 
echo "- ./scripts/start-all.sh        - Start all services (FIXED)"
echo "- ./scripts/stop-all.sh         - Stop all services (FIXED)"
echo "- ./scripts/backup-all.sh       - Backup all databases (FIXED)"
echo "- ./scripts/status.sh           - Check system status (FIXED)"
echo ""
echo "🔧 Key Changes:"
echo "- Auto-detects docker-compose.yml location"
echo "- Works with your deployment/docker/database/docker-compose.yml"
echo "- Proper error handling for missing containers"
echo "- Better backup error handling"
echo "- Path persistence between script runs"