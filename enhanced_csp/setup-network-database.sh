#!/bin/bash
# setup-network-database.sh - Network Database Setup Script

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

# Database configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-enhanced_csp}"
DB_USER="${DB_USER:-csp_user}"
DB_PASSWORD="${DB_PASSWORD:-csp_password}"

log_info "Setting up Enhanced CSP Network Database..."

# Create network database initialization directory
mkdir -p database/network_init

# Save the schema to a file
cat > database/network_init/01-init-network.sql << 'EOF'
-- ============================================================================
-- Enhanced CSP Network Database Schema
-- ============================================================================

-- Create network schema
CREATE SCHEMA IF NOT EXISTS network;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- [Rest of the schema from the artifact above]
EOF

# Function to execute SQL
execute_sql() {
    local sql_file=$1
    log_info "Executing $sql_file..."
    
    PGPASSWORD=$DB_PASSWORD psql \
        -h $DB_HOST \
        -p $DB_PORT \
        -U $DB_USER \
        -d $DB_NAME \
        -f "$sql_file" \
        --single-transaction \
        --set ON_ERROR_STOP=on
}

# Check if database exists
if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
    log_info "Database $DB_NAME exists"
else
    log_error "Database $DB_NAME does not exist. Please create it first."
    exit 1
fi

# Check if network schema already exists
if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "\dn" | grep -q network; then
    log_warn "Network schema already exists. Do you want to drop and recreate it? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        log_info "Dropping existing network schema..."
        PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DROP SCHEMA network CASCADE;"
    else
        log_info "Keeping existing schema. Exiting."
        exit 0
    fi
fi

# Execute the network schema
execute_sql "database/network_init/01-init-network.sql"

# Create additional helper functions
log_info "Creating helper functions..."

cat > database/network_init/02-helper-functions.sql << 'EOF'
-- Helper function to get node connectivity
CREATE OR REPLACE FUNCTION network.get_node_connectivity(p_node_id UUID)
RETURNS TABLE (
    connected_nodes INTEGER,
    active_links INTEGER,
    avg_link_quality NUMERIC,
    total_bandwidth NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(DISTINCT CASE 
            WHEN ml.local_node_id = p_node_id THEN ml.remote_node_id
            ELSE ml.local_node_id 
        END)::INTEGER as connected_nodes,
        COUNT(*)::INTEGER as active_links,
        AVG(ml.quality)::NUMERIC as avg_link_quality,
        SUM(ml.bandwidth_mbps)::NUMERIC as total_bandwidth
    FROM network.mesh_links ml
    WHERE (ml.local_node_id = p_node_id OR ml.remote_node_id = p_node_id)
    AND ml.link_state = 'active';
END;
$$ LANGUAGE plpgsql;

-- Function to record network metrics
CREATE OR REPLACE FUNCTION network.record_metric(
    p_node_id UUID,
    p_metric_type VARCHAR,
    p_metric_name VARCHAR,
    p_metric_value DECIMAL,
    p_unit VARCHAR DEFAULT NULL,
    p_tags JSONB DEFAULT '{}'
) RETURNS UUID AS $$
DECLARE
    v_metric_id UUID;
BEGIN
    INSERT INTO network.metrics (
        node_id, metric_type, metric_name, metric_value, unit, tags
    ) VALUES (
        p_node_id, p_metric_type, p_metric_name, p_metric_value, p_unit, p_tags
    ) RETURNING metric_id INTO v_metric_id;
    
    RETURN v_metric_id;
END;
$$ LANGUAGE plpgsql;

-- Function to update link quality
CREATE OR REPLACE FUNCTION network.update_link_quality(
    p_link_id UUID,
    p_quality DECIMAL,
    p_latency_ms DECIMAL,
    p_packet_loss DECIMAL DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    UPDATE network.mesh_links
    SET 
        quality = p_quality,
        latency_ms = p_latency_ms,
        packet_loss = COALESCE(p_packet_loss, packet_loss),
        last_probe = CURRENT_TIMESTAMP
    WHERE link_id = p_link_id;
    
    -- Update link state based on quality
    UPDATE network.mesh_links
    SET link_state = CASE
        WHEN quality >= 0.9 THEN 'active'
        WHEN quality >= 0.7 THEN 'degraded'
        WHEN quality >= 0.5 THEN 'congested'
        ELSE 'failing'
    END
    WHERE link_id = p_link_id;
END;
$$ LANGUAGE plpgsql;
EOF

execute_sql "database/network_init/02-helper-functions.sql"

# Create monitoring views
log_info "Creating monitoring views..."

cat > database/network_init/03-monitoring-views.sql << 'EOF'
-- Real-time network status view
CREATE OR REPLACE VIEW network.realtime_status AS
SELECT 
    n.node_name,
    n.node_type,
    n.role,
    n.status,
    n.last_seen,
    COALESCE(conn.connected_nodes, 0) as connected_nodes,
    COALESCE(conn.active_links, 0) as active_links,
    ROUND(COALESCE(conn.avg_link_quality, 0), 2) as avg_link_quality,
    COALESCE(conn.total_bandwidth, 0) as total_bandwidth_mbps,
    CASE 
        WHEN n.last_seen > CURRENT_TIMESTAMP - INTERVAL '1 minute' THEN 'online'
        WHEN n.last_seen > CURRENT_TIMESTAMP - INTERVAL '5 minutes' THEN 'degraded'
        ELSE 'offline'
    END as health_status
FROM network.nodes n
LEFT JOIN LATERAL network.get_node_connectivity(n.node_id) conn ON true
ORDER BY n.node_name;

-- Link utilization view
CREATE OR REPLACE VIEW network.link_utilization AS
SELECT 
    n1.node_name as local_node,
    n2.node_name as remote_node,
    ml.link_state,
    ml.quality,
    ml.latency_ms,
    ml.bandwidth_mbps,
    ml.packet_loss,
    ml.last_probe,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ml.last_probe)) as seconds_since_probe
FROM network.mesh_links ml
JOIN network.nodes n1 ON ml.local_node_id = n1.node_id
JOIN network.nodes n2 ON ml.remote_node_id = n2.node_id
ORDER BY ml.quality DESC;

-- Optimization effectiveness view
CREATE OR REPLACE VIEW network.optimization_effectiveness AS
SELECT 
    mt.mesh_name,
    mt.topology_type,
    COUNT(to_opt.optimization_id) as total_optimizations,
    AVG(to_opt.improvement_percentage) as avg_improvement,
    SUM(CASE WHEN to_opt.success THEN 1 ELSE 0 END) as successful_optimizations,
    AVG(to_opt.execution_time_ms) as avg_execution_time_ms,
    MAX(to_opt.performed_at) as last_optimization
FROM network.mesh_topologies mt
LEFT JOIN network.topology_optimizations to_opt ON mt.topology_id = to_opt.topology_id
GROUP BY mt.topology_id, mt.mesh_name, mt.topology_type;
EOF

execute_sql "database/network_init/03-monitoring-views.sql"

# Create partitioning for metrics table
log_info "Setting up table partitioning..."

cat > database/network_init/04-partitioning.sql << 'EOF'
-- Convert metrics table to partitioned table
ALTER TABLE network.metrics RENAME TO metrics_old;

-- Create partitioned metrics table
CREATE TABLE network.metrics (
    metric_id UUID DEFAULT uuid_generate_v4(),
    node_id UUID REFERENCES network.nodes(node_id) ON DELETE CASCADE,
    metric_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(20,6) NOT NULL,
    unit VARCHAR(50),
    tags JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (metric_id, recorded_at)
) PARTITION BY RANGE (recorded_at);

-- Create indexes on partitioned table
CREATE INDEX idx_metrics_node_time ON network.metrics (node_id, recorded_at DESC);
CREATE INDEX idx_metrics_type_name ON network.metrics (metric_type, metric_name);

-- Create monthly partitions for the next 12 months
DO $$
DECLARE
    start_date DATE := DATE_TRUNC('month', CURRENT_DATE);
    end_date DATE;
    partition_name TEXT;
BEGIN
    FOR i IN 0..11 LOOP
        end_date := start_date + INTERVAL '1 month';
        partition_name := 'metrics_' || TO_CHAR(start_date, 'YYYY_MM');
        
        EXECUTE format('
            CREATE TABLE IF NOT EXISTS network.%I PARTITION OF network.metrics
            FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
        
        start_date := end_date;
    END LOOP;
END;
$$;

-- Copy data from old table if it exists
INSERT INTO network.metrics SELECT * FROM network.metrics_old;
DROP TABLE network.metrics_old;

-- Create automatic partition management function
CREATE OR REPLACE FUNCTION network.create_monthly_partition()
RETURNS VOID AS $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    partition_date := DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month');
    partition_name := 'metrics_' || TO_CHAR(partition_date, 'YYYY_MM');
    start_date := partition_date;
    end_date := partition_date + INTERVAL '1 month';
    
    -- Check if partition already exists
    IF NOT EXISTS (
        SELECT 1 FROM pg_tables 
        WHERE schemaname = 'network' 
        AND tablename = partition_name
    ) THEN
        EXECUTE format('
            CREATE TABLE network.%I PARTITION OF network.metrics
            FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
        
        RAISE NOTICE 'Created partition % for dates % to %', 
            partition_name, start_date, end_date;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Schedule monthly partition creation (requires pg_cron extension)
-- CREATE EXTENSION IF NOT EXISTS pg_cron;
-- SELECT cron.schedule('create-network-partitions', '0 0 1 * *', 'SELECT network.create_monthly_partition()');
EOF

execute_sql "database/network_init/04-partitioning.sql"

# Verify installation
log_info "Verifying network database setup..."

TABLES=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "
    SELECT COUNT(*) 
    FROM information_schema.tables 
    WHERE table_schema = 'network' 
    AND table_type = 'BASE TABLE'
")

VIEWS=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "
    SELECT COUNT(*) 
    FROM information_schema.views 
    WHERE table_schema = 'network'
")

log_info "Created $TABLES tables and $VIEWS views in network schema"

# Create configuration file for the network module
log_info "Creating network database configuration..."

cat > network/network_db_config.py << 'EOF'
"""
Network Database Configuration
"""
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Network database URL
NETWORK_DATABASE_URL = os.getenv(
    "NETWORK_DATABASE_URL",
    "postgresql+asyncpg://csp_user:csp_password@localhost:5432/enhanced_csp"
)

# Create async engine for network database
network_engine = create_async_engine(
    NETWORK_DATABASE_URL,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    echo=False
)

# Create session factory
NetworkSessionLocal = sessionmaker(
    bind=network_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_network_db():
    """Get network database session"""
    async with NetworkSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
EOF

log_info "Network database setup complete!"
log_info ""
log_info "Next steps:"
log_info "1. Update your .env file with NETWORK_DATABASE_URL if needed"
log_info "2. Import network_db_config in your network module"
log_info "3. Use get_network_db() for database operations"
log_info ""
log_info "Example usage:"
log_info "  from enhanced_csp.network.network_db_config import get_network_db"
log_info "  async for db in get_network_db():"
log_info "      result = await db.execute('SELECT * FROM network.nodes')"
