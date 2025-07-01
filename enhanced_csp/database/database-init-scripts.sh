#!/bin/bash
# setup-databases.sh - Complete database setup script

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Create directory structure
log_info "Creating database directory structure..."
mkdir -p database/{init,ai_models_init,pgvector/init,mongodb/init,qdrant/config,pgadmin}

# =============================================================================
# PostgreSQL Main Database Initialization
# =============================================================================
log_info "Creating main PostgreSQL initialization script..."
cat > database/init/01-init-main.sql << 'EOF'
-- Main CSP System Database Initialization

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS csp;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Create main tables
CREATE TABLE IF NOT EXISTS csp.designs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    configuration JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    version INTEGER DEFAULT 1,
    status VARCHAR(50) DEFAULT 'draft'
);

CREATE TABLE IF NOT EXISTS csp.executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    design_id UUID REFERENCES csp.designs(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    result JSONB,
    error TEXT,
    metrics JSONB
);

CREATE TABLE IF NOT EXISTS csp.components (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    configuration JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create monitoring tables
CREATE TABLE IF NOT EXISTS monitoring.metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    labels JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_designs_created_at ON csp.designs(created_at);
CREATE INDEX idx_designs_status ON csp.designs(status);
CREATE INDEX idx_executions_design_id ON csp.executions(design_id);
CREATE INDEX idx_executions_status ON csp.executions(status);
CREATE INDEX idx_metrics_timestamp ON monitoring.metrics(timestamp);
CREATE INDEX idx_metrics_name ON monitoring.metrics(metric_name);

-- Create update trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_designs_updated_at BEFORE UPDATE ON csp.designs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA csp TO csp_user;
GRANT ALL PRIVILEGES ON SCHEMA monitoring TO csp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA csp TO csp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO csp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA csp TO csp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO csp_user;
EOF

# =============================================================================
# AI Models Database Initialization
# =============================================================================
log_info "Creating AI models database initialization script..."
cat > database/ai_models_init/01-init-ai-models.sql << 'EOF'
-- AI Models Database Initialization

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS models;
CREATE SCHEMA IF NOT EXISTS training;

-- Model registry table
CREATE TABLE IF NOT EXISTS models.model_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    version VARCHAR(50) NOT NULL,
    type VARCHAR(100) NOT NULL,
    provider VARCHAR(100),
    configuration JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active'
);

-- Model metrics table
CREATE TABLE IF NOT EXISTS models.model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models.model_registry(id) ON DELETE CASCADE,
    metric_type VARCHAR(100) NOT NULL,
    metric_value JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Training data table
CREATE TABLE IF NOT EXISTS training.datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    data_type VARCHAR(100),
    size_bytes BIGINT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Training runs table
CREATE TABLE IF NOT EXISTS training.runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models.model_registry(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES training.datasets(id) ON DELETE SET NULL,
    parameters JSONB NOT NULL DEFAULT '{}',
    metrics JSONB NOT NULL DEFAULT '{}',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'pending'
);

-- Create indexes
CREATE INDEX idx_model_registry_name ON models.model_registry(name);
CREATE INDEX idx_model_registry_type ON models.model_registry(type);
CREATE INDEX idx_model_metrics_model_id ON models.model_metrics(model_id);
CREATE INDEX idx_training_runs_model_id ON training.runs(model_id);

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA models TO csp_user;
GRANT ALL PRIVILEGES ON SCHEMA training TO csp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA models TO csp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA training TO csp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA models TO csp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA training TO csp_user;
EOF

# =============================================================================
# Vector Database Initialization
# =============================================================================
log_info "Creating vector database initialization script..."
cat > database/pgvector/init/01-init-vector-db.sql << 'EOF'
-- Vector Database Initialization with pgvector

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schema
CREATE SCHEMA IF NOT EXISTS embeddings;

-- Embeddings table
CREATE TABLE IF NOT EXISTS embeddings.documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for similarity search
CREATE INDEX ON embeddings.documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_documents_metadata ON embeddings.documents USING GIN(metadata);
CREATE INDEX idx_documents_created_at ON embeddings.documents(created_at);

-- Function to search similar documents
CREATE OR REPLACE FUNCTION embeddings.search_similar(
    query_embedding vector(1536),
    limit_count INTEGER DEFAULT 10,
    threshold FLOAT DEFAULT 0.7
)
RETURNS TABLE(
    id UUID,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_embedding) as similarity
    FROM embeddings.documents d
    WHERE 1 - (d.embedding <=> query_embedding) > threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA embeddings TO csp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA embeddings TO csp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA embeddings TO csp_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA embeddings TO csp_user;
EOF

# =============================================================================
# MongoDB Initialization
# =============================================================================
log_info "Creating MongoDB initialization script..."
cat > database/mongodb/init/01-init-mongo.js << 'EOF'
// MongoDB Initialization Script

// Switch to csp_nosql database
db = db.getSiblingDB('csp_nosql');

// Create collections with validation
db.createCollection('events', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['event_type', 'timestamp', 'data'],
            properties: {
                event_type: {
                    bsonType: 'string',
                    description: 'Type of event'
                },
                timestamp: {
                    bsonType: 'date',
                    description: 'Event timestamp'
                },
                data: {
                    bsonType: 'object',
                    description: 'Event data'
                }
            }
        }
    }
});

db.createCollection('logs', {
    capped: true,
    size: 104857600, // 100MB
    max: 1000000
});

db.createCollection('configurations', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['name', 'version', 'config'],
            properties: {
                name: {
                    bsonType: 'string',
                    description: 'Configuration name'
                },
                version: {
                    bsonType: 'string',
                    description: 'Configuration version'
                },
                config: {
                    bsonType: 'object',
                    description: 'Configuration data'
                }
            }
        }
    }
});

// Create indexes
db.events.createIndex({ timestamp: -1 });
db.events.createIndex({ event_type: 1, timestamp: -1 });
db.logs.createIndex({ timestamp: -1 });
db.configurations.createIndex({ name: 1, version: 1 }, { unique: true });

// Create user for application
db.createUser({
    user: 'csp_app',
    pwd: 'csp_app_pass_2024!',
    roles: [
        { role: 'readWrite', db: 'csp_nosql' }
    ]
});
EOF

# =============================================================================
# Qdrant Configuration
# =============================================================================
log_info "Creating Qdrant configuration..."
cat > database/qdrant/config/config.yaml << 'EOF'
service:
  http_port: 6333
  grpc_port: 6334
  max_request_size_mb: 1024
  max_workers: 0
  enable_cors: true

storage:
  storage_path: /qdrant/storage
  snapshots_path: /qdrant/snapshots
  on_disk_payload: true
  performance:
    max_search_threads: 0
    max_optimization_threads: 1

cluster:
  enabled: false

log_level: INFO
EOF

# =============================================================================
# pgAdmin Configuration
# =============================================================================
log_info "Creating pgAdmin servers configuration..."
cat > database/pgadmin/servers.json << 'EOF'
{
    "Servers": {
        "1": {
            "Name": "CSP Main Database",
            "Group": "CSP Databases",
            "Port": 5432,
            "Username": "csp_user",
            "Host": "postgres",
            "SSLMode": "prefer",
            "MaintenanceDB": "postgres"
        },
        "2": {
            "Name": "CSP AI Models Database",
            "Group": "CSP Databases",
            "Port": 5432,
            "Username": "csp_user",
            "Host": "postgres_ai_models",
            "SSLMode": "prefer",
            "MaintenanceDB": "postgres"
        },
        "3": {
            "Name": "CSP Vector Database",
            "Group": "CSP Databases",
            "Port": 5432,
            "Username": "csp_user",
            "Host": "postgres_vector",
            "SSLMode": "prefer",
            "MaintenanceDB": "postgres"
        }
    }
}
EOF

# =============================================================================
# Environment File
# =============================================================================
log_info "Creating environment file..."
cat > .env.databases << 'EOF'
# Database Passwords
DB_PASSWORD=csp_secure_pass_2024!
REDIS_PASSWORD=redis_secure_pass_2024!
MONGO_PASSWORD=mongo_secure_pass_2024!

# Admin Tool Passwords
PGADMIN_EMAIL=admin@csp.local
PGADMIN_PASSWORD=pgadmin_pass_2024!

# Connection Strings for Application
DATABASE_URL=postgresql://csp_user:csp_secure_pass_2024!@localhost:5432/csp_system
AI_MODELS_DATABASE_URL=postgresql://csp_user:csp_secure_pass_2024!@localhost:5433/ai_models
VECTOR_DATABASE_URL=postgresql://csp_user:csp_secure_pass_2024!@localhost:5434/csp_vectors
REDIS_URL=redis://:redis_secure_pass_2024!@localhost:6379/0
MONGODB_URL=mongodb://csp_admin:mongo_secure_pass_2024!@localhost:27017/csp_nosql
EOF

# =============================================================================
# Network Creation
# =============================================================================
log_info "Creating Docker network..."
docker network create scripts_csp-network 2>/dev/null || log_warn "Network already exists"

# =============================================================================
# Start Databases
# =============================================================================
log_info "Starting database services..."

# Stop any existing containers
docker-compose -f docker-compose.databases.yml down

# Start all database services
docker-compose -f docker-compose.databases.yml up -d

# Wait for services to be ready
log_info "Waiting for databases to be ready..."
sleep 10

# =============================================================================
# Verify Services
# =============================================================================
log_info "Verifying database services..."

# Check PostgreSQL
docker exec csp_postgres pg_isready -U csp_user -d csp_system && \
    log_info "✓ Main PostgreSQL is ready" || \
    log_error "✗ Main PostgreSQL is not ready"

docker exec csp_ai_models_db pg_isready -U csp_user -d ai_models && \
    log_info "✓ AI Models PostgreSQL is ready" || \
    log_error "✗ AI Models PostgreSQL is not ready"

docker exec csp_postgres_vector pg_isready -U csp_user -d csp_vectors && \
    log_info "✓ Vector PostgreSQL is ready" || \
    log_error "✗ Vector PostgreSQL is not ready"

# Check Redis
docker exec csp_redis redis-cli -a redis_secure_pass_2024! ping | grep -q PONG && \
    log_info "✓ Redis is ready" || \
    log_error "✗ Redis is not ready"

# Check MongoDB
docker exec csp_mongodb mongosh --eval "db.adminCommand('ping')" --quiet && \
    log_info "✓ MongoDB is ready" || \
    log_error "✗ MongoDB is not ready"

# =============================================================================
# Summary
# =============================================================================
log_info "Database setup complete!"
echo ""
echo "=================================="
echo "DATABASE ACCESS INFORMATION"
echo "=================================="
echo ""
echo "PostgreSQL Databases:"
echo "  Main DB:        postgresql://csp_user:csp_secure_pass_2024!@localhost:5432/csp_system"
echo "  AI Models DB:   postgresql://csp_user:csp_secure_pass_2024!@localhost:5433/ai_models"
echo "  Vector DB:      postgresql://csp_user:csp_secure_pass_2024!@localhost:5434/csp_vectors"
echo ""
echo "Redis:"
echo "  Connection:     redis://:redis_secure_pass_2024!@localhost:6379/0"
echo ""
echo "MongoDB:"
echo "  Connection:     mongodb://csp_admin:mongo_secure_pass_2024!@localhost:27017/csp_nosql"
echo ""
echo "Vector Databases:"
echo "  ChromaDB:       http://localhost:8200"
echo "  Qdrant:         http://localhost:6333"
echo "  Weaviate:       http://localhost:8080"
echo ""
echo "Admin Tools:"
echo "  pgAdmin:        http://localhost:5050 (admin@csp.local / pgadmin_pass_2024!)"
echo "  RedisInsight:   http://localhost:8001"
echo ""
echo "To stop all databases: docker-compose -f docker-compose.databases.yml down"
echo "To view logs: docker-compose -f docker-compose.databases.yml logs -f"
echo "=================================="