#!/bin/bash
# setup-secure-databases.sh
# Sets up secure database configurations and initialization scripts

set -e

echo "🔒 Setting up secure database configurations"
echo "=========================================="

# Create directory structure
mkdir -p init-scripts/{postgres,mongo,vector}
mkdir -p config/{redis,nginx/conf.d}

# ============================================================================
# PostgreSQL Initialization Script
# ============================================================================
cat > init-scripts/postgres/01-init-security.sql << 'EOF'
-- PostgreSQL Security Initialization Script

-- Revoke default public permissions
REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;

-- Create application user with limited permissions
CREATE USER app_user WITH PASSWORD '${APP_USER_PASSWORD}';
GRANT CONNECT ON DATABASE ${DB_NAME} TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT CREATE ON SCHEMA public TO app_user;

-- Create read-only user for reporting
CREATE USER readonly_user WITH PASSWORD '${READONLY_PASSWORD}';
GRANT CONNECT ON DATABASE ${DB_NAME} TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;

-- Enable row level security
ALTER DATABASE ${DB_NAME} SET row_security = on;

-- Set secure connection parameters
ALTER DATABASE ${DB_NAME} SET ssl = on;
ALTER DATABASE ${DB_NAME} SET ssl_min_protocol_version = 'TLSv1.2';

-- Create audit table
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    user_name TEXT,
    action TEXT,
    table_name TEXT,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    client_ip INET,
    details JSONB
);

-- Function to log activities
CREATE OR REPLACE FUNCTION log_activity() RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log(user_name, action, table_name, client_ip, details)
    VALUES (current_user, TG_OP, TG_TABLE_NAME, inet_client_addr(), row_to_json(NEW));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
EOF

# ============================================================================
# Vector Database Initialization
# ============================================================================
cat > init-scripts/vector/01-init-pgvector.sql << 'EOF'
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create vector storage schema
CREATE SCHEMA IF NOT EXISTS vectors;

-- Create embeddings table with proper indexes
CREATE TABLE IF NOT EXISTS vectors.embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),  -- OpenAI embedding size
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for similarity search
CREATE INDEX IF NOT EXISTS embeddings_embedding_idx ON vectors.embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Add metadata GIN index for fast JSON queries
CREATE INDEX IF NOT EXISTS embeddings_metadata_idx ON vectors.embeddings 
USING GIN (metadata);

-- Function to search similar vectors
CREATE OR REPLACE FUNCTION vectors.search_similar(
    query_embedding vector,
    limit_count INT DEFAULT 10,
    threshold FLOAT DEFAULT 0.8
) RETURNS TABLE (
    id INT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.id,
        e.content,
        e.metadata,
        1 - (e.embedding <=> query_embedding) as similarity
    FROM vectors.embeddings e
    WHERE 1 - (e.embedding <=> query_embedding) > threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;
EOF

# ============================================================================
# MongoDB Initialization Script
# ============================================================================
cat > init-scripts/mongo/01-init-users.js << 'EOF'
// MongoDB User Initialization Script

// Switch to admin database
db = db.getSiblingDB('admin');

// Create application user
db.createUser({
  user: process.env.MONGO_APP_USER,
  pwd: process.env.MONGO_APP_PASSWORD,
  roles: [
    {
      role: "readWrite",
      db: process.env.MONGO_DB_NAME
    }
  ]
});

// Create read-only user
db.createUser({
  user: process.env.MONGO_READONLY_USER,
  pwd: process.env.MONGO_READONLY_PASSWORD,
  roles: [
    {
      role: "read",
      db: process.env.MONGO_DB_NAME
    }
  ]
});

// Switch to application database
db = db.getSiblingDB(process.env.MONGO_DB_NAME);

// Create collections with validation
db.createCollection("sessions", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["user_id", "token", "expires_at"],
      properties: {
        user_id: {
          bsonType: "string",
          description: "User ID is required"
        },
        token: {
          bsonType: "string",
          description: "Session token is required"
        },
        expires_at: {
          bsonType: "date",
          description: "Expiration date is required"
        }
      }
    }
  }
});

// Create indexes
db.sessions.createIndex({ "token": 1 }, { unique: true });
db.sessions.createIndex({ "expires_at": 1 }, { expireAfterSeconds: 0 });
EOF

# ============================================================================
# Redis Configuration
# ============================================================================
cat > config/redis/redis.conf << 'EOF'
# Redis Security Configuration

# Network security
bind 0.0.0.0
protected-mode yes
port 6379

# Authentication
requirepass ${REDIS_PASSWORD}

# Persistence
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb

# Append only file
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no

# Memory management
maxmemory 512mb
maxmemory-policy allkeys-lru

# Security
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command KEYS ""
rename-command CONFIG ""

# Logging
loglevel notice
logfile /data/redis.log

# Slow log
slowlog-log-slower-than 10000
slowlog-max-len 128

# Client management
timeout 300
tcp-keepalive 300
tcp-backlog 511
maxclients 10000
EOF

# ============================================================================
# Environment Variables Update Script
# ============================================================================
cat > update-backend-env.sh << 'EOF'
#!/bin/bash
# Updates backend/.env with secure database passwords

# Function to generate secure password
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
}

# Generate passwords
REDIS_PASSWORD=$(generate_password)
DB_PASSWORD=$(generate_password)
AI_MODELS_DB_PASSWORD=$(generate_password)
VECTOR_DB_PASSWORD=$(generate_password)
MONGO_ROOT_PASSWORD=$(generate_password)
MONGO_APP_PASSWORD=$(generate_password)
APP_USER_PASSWORD=$(generate_password)
READONLY_PASSWORD=$(generate_password)

# Backup existing env
cp backend/.env backend/.env.backup

# Update backend/.env with secure passwords
cat >> backend/.env << EOL

# ============================================================================
# SECURE DATABASE PASSWORDS - Generated on $(date)
# ============================================================================
# Redis
REDIS_PASSWORD=${REDIS_PASSWORD}
REDIS_URL=redis://default:${REDIS_PASSWORD}@localhost:6379/0

# PostgreSQL Databases
DB_PASSWORD=${DB_PASSWORD}
DATABASE_URL=postgresql://csp_user:${DB_PASSWORD}@localhost:5432/csp_visual_designer

AI_MODELS_DB_PASSWORD=${AI_MODELS_DB_PASSWORD}
AI_MODELS_DB_URL=postgresql://ai_models_user:${AI_MODELS_DB_PASSWORD}@localhost:5433/ai_models_db

VECTOR_DB_PASSWORD=${VECTOR_DB_PASSWORD}
VECTOR_DB_URL=postgresql://vector_user:${VECTOR_DB_PASSWORD}@localhost:5434/vector_db

# MongoDB
MONGO_ROOT_USER=root
MONGO_ROOT_PASSWORD=${MONGO_ROOT_PASSWORD}
MONGO_APP_USER=csp_app
MONGO_APP_PASSWORD=${MONGO_APP_PASSWORD}
MONGO_READONLY_USER=csp_readonly
MONGO_READONLY_PASSWORD=${READONLY_PASSWORD}
MONGO_DB_NAME=csp_platform
MONGO_URI=mongodb://csp_app:${MONGO_APP_PASSWORD}@localhost:27017/csp_platform?authSource=csp_platform

# PostgreSQL App Users
APP_USER_PASSWORD=${APP_USER_PASSWORD}
READONLY_PASSWORD=${READONLY_PASSWORD}
EOL

echo "✅ Updated backend/.env with secure passwords"
echo "⚠️  Save these passwords in a secure location!"
EOF

chmod +x update-backend-env.sh

# ============================================================================
# Python Database Connection Updates
# ============================================================================
cat > backend/database/secure_connections.py << 'EOF'
"""
Secure Database Connection Configuration
"""
import os
from typing import Optional
from urllib.parse import quote_plus
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, QueuePool
from motor.motor_asyncio import AsyncIOMotorClient
import asyncpg
import ssl

class SecureDatabaseConfig:
    """Secure database configuration with validation"""
    
    def __init__(self):
        self._validate_environment()
        
    def _validate_environment(self):
        """Validate required environment variables"""
        required_vars = [
            'DB_PASSWORD', 'REDIS_PASSWORD', 
            'AI_MODELS_DB_PASSWORD', 'VECTOR_DB_PASSWORD'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
    
    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL with auth"""
        user = os.getenv('DB_USER', 'csp_user')
        password = quote_plus(os.getenv('DB_PASSWORD'))
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        database = os.getenv('DB_NAME', 'csp_visual_designer')
        
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL with auth"""
        password = quote_plus(os.getenv('REDIS_PASSWORD'))
        host = os.getenv('REDIS_HOST', 'localhost')
        port = os.getenv('REDIS_PORT', '6379')
        db = os.getenv('REDIS_DB', '0')
        
        return f"redis://default:{password}@{host}:{port}/{db}"
    
    @property
    def mongodb_url(self) -> str:
        """Get MongoDB connection URL with auth"""
        user = os.getenv('MONGO_APP_USER', 'csp_app')
        password = quote_plus(os.getenv('MONGO_APP_PASSWORD'))
        host = os.getenv('MONGO_HOST', 'localhost')
        port = os.getenv('MONGO_PORT', '27017')
        database = os.getenv('MONGO_DB_NAME', 'csp_platform')
        
        return f"mongodb://{user}:{password}@{host}:{port}/{database}?authSource={database}"

class SecureConnections:
    """Manage secure database connections"""
    
    def __init__(self):
        self.config = SecureDatabaseConfig()
        self._postgres_engine = None
        self._redis_pool = None
        self._mongo_client = None
        
    async def init_postgres(self) -> None:
        """Initialize PostgreSQL connection with SSL"""
        self._postgres_engine = create_async_engine(
            self.config.postgres_url,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,
            pool_recycle=3600,
            poolclass=QueuePool,
            connect_args={
                "server_settings": {
                    "application_name": "csp_backend",
                    "jit": "off"
                },
                "ssl": ssl.create_default_context(),
                "command_timeout": 60,
            }
        )
        
        self.AsyncSessionLocal = sessionmaker(
            self._postgres_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def init_redis(self) -> None:
        """Initialize Redis connection pool with auth"""
        self._redis_pool = redis.ConnectionPool.from_url(
            self.config.redis_url,
            max_connections=50,
            decode_responses=True,
            health_check_interval=30,
            socket_keepalive=True,
            socket_keepalive_options={
                1: 3,   # TCP_KEEPIDLE
                2: 3,   # TCP_KEEPINTVL
                3: 3,   # TCP_KEEPCNT
            }
        )
        
    async def init_mongodb(self) -> None:
        """Initialize MongoDB connection with auth"""
        self._mongo_client = AsyncIOMotorClient(
            self.config.mongodb_url,
            maxPoolSize=50,
            minPoolSize=10,
            maxIdleTimeMS=45000,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            retryWrites=True,
            retryReads=True,
            tlsAllowInvalidCertificates=False  # Enforce TLS validation
        )
    
    async def get_postgres_session(self) -> AsyncSession:
        """Get PostgreSQL session"""
        if not self._postgres_engine:
            await self.init_postgres()
        return self.AsyncSessionLocal()
    
    async def get_redis_client(self) -> redis.Redis:
        """Get Redis client"""
        if not self._redis_pool:
            await self.init_redis()
        return redis.Redis(connection_pool=self._redis_pool)
    
    def get_mongo_database(self):
        """Get MongoDB database"""
        if not self._mongo_client:
            raise RuntimeError("MongoDB not initialized")
        return self._mongo_client[self.config.config.MONGO_DB_NAME]
    
    async def close_all(self):
        """Close all connections"""
        if self._postgres_engine:
            await self._postgres_engine.dispose()
        if self._redis_pool:
            await self._redis_pool.disconnect()
        if self._mongo_client:
            self._mongo_client.close()

# Global instance
secure_db = SecureConnections()
EOF

echo "✅ Created secure database setup scripts"
echo
echo "Next steps:"
echo "1. Run ./update-backend-env.sh to generate secure passwords"
echo "2. Review docker-compose.secure.yml"
echo "3. Update your backend code to use secure_connections.py"
echo "4. Run: docker-compose -f docker-compose.secure.yml up"