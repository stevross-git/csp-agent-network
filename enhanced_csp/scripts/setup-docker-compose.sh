#!/bin/bash
# scripts/setup-docker-compose.sh
# Script to set up the enhanced docker-compose.yml with vector databases

echo "🔧 Setting up Enhanced Docker Compose Configuration"
echo "=================================================="

# Check if the existing docker-compose file exists
EXISTING_COMPOSE=""
if [ -f "deployment/docker/database/docker-compose.yml" ]; then
    EXISTING_COMPOSE="deployment/docker/database/docker-compose.yml"
    echo "📍 Found existing docker-compose.yml at: $EXISTING_COMPOSE"
elif [ -f "deployment/docker/docker-compose.yml" ]; then
    EXISTING_COMPOSE="deployment/docker/docker-compose.yml"
    echo "📍 Found existing docker-compose.yml at: $EXISTING_COMPOSE"
else
    echo "❌ Could not find existing docker-compose.yml file"
    echo "Creating new docker-compose.yml in root directory..."
fi

# Create enhanced docker-compose.yml in root directory
echo "📝 Creating enhanced docker-compose.yml with vector databases..."

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: csp_postgres
    environment:
      POSTGRES_DB: csp_visual_designer
      POSTGRES_USER: csp_user
      POSTGRES_PASSWORD: csp_password
      PGDATA: /var/lib/postgresql/data/pgdata
    volumes:
      - postgres_data:/var/lib/postgresql/data/pgdata
      - ./database/init:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    networks:
      - csp-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U csp_user -d csp_visual_designer"]
      interval: 10s
      timeout: 5s
      retries: 5

  postgres_ai_models:
    image: postgres:15-alpine
    container_name: csp_ai_models_db
    environment:
      POSTGRES_DB: ai_models_db
      POSTGRES_USER: ai_models_user
      POSTGRES_PASSWORD: ai_models_password
      PGDATA: /var/lib/postgresql/data/pgdata
    volumes:
      - postgres_ai_models_data:/var/lib/postgresql/data/pgdata
      - ./database/ai_models_init:/docker-entrypoint-initdb.d
    ports:
      - "5433:5432"
    networks:
      - csp-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ai_models_user -d ai_models_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  postgres_vector:
    image: pgvector/pgvector:pg15
    container_name: csp_postgres_vector
    environment:
      POSTGRES_DB: vector_db
      POSTGRES_USER: vector_user
      POSTGRES_PASSWORD: vector_password
      PGDATA: /var/lib/postgresql/data/pgdata
    volumes:
      - postgres_vector_data:/var/lib/postgresql/data/pgdata
      - ./database/pgvector/init:/docker-entrypoint-initdb.d
    ports:
      - "5434:5432"
    networks:
      - csp-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U vector_user -d vector_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: csp_redis
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - csp-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  chroma:
    image: chromadb/chroma:latest
    container_name: csp_chroma
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_HTTP_PORT=8000
      - ANONYMIZED_TELEMETRY=False
    volumes:
      - chroma_data:/chroma/chroma
    ports:
      - "8200:8000"
    networks:
      - csp-network
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    container_name: csp_qdrant
    environment:
      QDRANT__SERVICE__HTTP_PORT: 6333
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"
    networks:
      - csp-network
    restart: unless-stopped

  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.23.7
    container_name: csp_weaviate
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate
    ports:
      - "8080:8080"
    networks:
      - csp-network
    restart: unless-stopped

  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: csp_pgadmin
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@csp.com
      PGADMIN_DEFAULT_PASSWORD: pgadmin_password
      PGADMIN_CONFIG_SERVER_MODE: 'False'
    volumes:
      - pgadmin_data:/var/lib/pgadmin
    ports:
      - "5050:80"
    networks:
      - csp-network
    restart: unless-stopped
    depends_on:
      - postgres
      - postgres_ai_models
      - postgres_vector

  redis-insight:
    image: redislabs/redisinsight:latest
    container_name: csp_redis_insight
    volumes:
      - redis_insight_data:/db
    ports:
      - "8001:8001"
    networks:
      - csp-network
    restart: unless-stopped
    depends_on:
      - redis

networks:
  csp-network:
    driver: bridge

volumes:
  postgres_data:
    driver: local
  postgres_ai_models_data:
    driver: local
  postgres_vector_data:
    driver: local
  redis_data:
    driver: local
  chroma_data:
    driver: local
  weaviate_data:
    driver: local
  qdrant_data:
    driver: local
  pgadmin_data:
    driver: local
  redis_insight_data:
    driver: local
EOF

echo "✅ Enhanced docker-compose.yml created in root directory"

# Create backup of existing file if it exists
if [ ! -z "$EXISTING_COMPOSE" ]; then
    echo "📋 Creating backup of existing docker-compose.yml..."
    cp "$EXISTING_COMPOSE" "${EXISTING_COMPOSE}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "✅ Backup created: ${EXISTING_COMPOSE}.backup.$(date +%Y%m%d_%H%M%S)"
fi

echo ""
echo "🎉 Docker Compose setup completed!"
echo ""
echo "📍 New docker-compose.yml location: ./docker-compose.yml"
echo "📋 Backup of original: ${EXISTING_COMPOSE}.backup.* (if existed)"
echo ""
echo "🚀 Next steps:"
echo "1. Run: ./scripts/setup-docker.sh"
echo "2. Run: ./scripts/start-databases.sh"
echo "3. Check status: ./scripts/status.sh"