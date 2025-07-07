-- ============================================================================
-- Enhanced CSP Network Database Schema
-- ============================================================================
-- This schema is designed to support the advanced networking features including:
-- - Mesh topology management
-- - BATMAN routing protocol
-- - Connection pooling
-- - Adaptive optimization
-- - Network telemetry and metrics
-- ============================================================================

-- Create network schema
CREATE SCHEMA IF NOT EXISTS network;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gist"; -- For exclusion constraints

-- ============================================================================
-- CORE NETWORK TABLES
-- ============================================================================

-- Network nodes table
CREATE TABLE network.nodes (
    node_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_name VARCHAR(255) NOT NULL UNIQUE,
    node_type VARCHAR(50) NOT NULL DEFAULT 'peer',
    role VARCHAR(50) DEFAULT 'peer',
    address VARCHAR(255) NOT NULL,
    port INTEGER NOT NULL,
    public_key TEXT,
    capabilities JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'inactive',
    last_seen TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT node_type_check CHECK (node_type IN ('peer', 'relay', 'gateway', 'bootstrap')),
    CONSTRAINT role_check CHECK (role IN ('peer', 'super_peer', 'relay', 'gateway', 'bootstrap', 'coordinator', 'witness')),
    CONSTRAINT status_check CHECK (status IN ('active', 'inactive', 'connecting', 'disconnected', 'quarantined'))
);

-- Mesh topology configurations
CREATE TABLE network.mesh_topologies (
    topology_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mesh_name VARCHAR(255) NOT NULL,
    topology_type VARCHAR(50) NOT NULL DEFAULT 'adaptive_hybrid',
    configuration JSONB NOT NULL DEFAULT '{}',
    optimization_enabled BOOLEAN DEFAULT true,
    learning_rate DECIMAL(5,4) DEFAULT 0.01,
    connectivity_threshold DECIMAL(3,2) DEFAULT 0.8,
    max_connections_per_node INTEGER DEFAULT 50,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT topology_type_check CHECK (topology_type IN (
        'full_mesh', 'partial_mesh', 'dynamic_partial', 'hierarchical',
        'small_world', 'scale_free', 'quantum_inspired', 'neural_mesh', 'adaptive_hybrid'
    ))
);

-- Mesh links between nodes
CREATE TABLE network.mesh_links (
    link_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    topology_id UUID REFERENCES network.mesh_topologies(topology_id) ON DELETE CASCADE,
    local_node_id UUID REFERENCES network.nodes(node_id) ON DELETE CASCADE,
    remote_node_id UUID REFERENCES network.nodes(node_id) ON DELETE CASCADE,
    link_state VARCHAR(50) NOT NULL DEFAULT 'establishing',
    quality DECIMAL(3,2) DEFAULT 0.0,
    latency_ms DECIMAL(10,2),
    bandwidth_mbps DECIMAL(10,2),
    packet_loss DECIMAL(5,4) DEFAULT 0.0,
    jitter_ms DECIMAL(10,2),
    weight DECIMAL(10,4) DEFAULT 1.0,
    last_probe TIMESTAMP WITH TIME ZONE,
    established_at TIMESTAMP WITH TIME ZONE,
    terminated_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT link_state_check CHECK (link_state IN (
        'establishing', 'active', 'degraded', 'congested', 'failing', 'dormant', 'quarantined'
    )),
    CONSTRAINT quality_check CHECK (quality >= 0 AND quality <= 1),
    CONSTRAINT packet_loss_check CHECK (packet_loss >= 0 AND packet_loss <= 1),
    CONSTRAINT different_nodes CHECK (local_node_id != remote_node_id),
    UNIQUE(topology_id, local_node_id, remote_node_id)
);

-- ============================================================================
-- ROUTING TABLES
-- ============================================================================

-- BATMAN routing entries
CREATE TABLE network.routing_entries (
    entry_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id UUID REFERENCES network.nodes(node_id) ON DELETE CASCADE,
    destination_id UUID REFERENCES network.nodes(node_id) ON DELETE CASCADE,
    next_hop_id UUID REFERENCES network.nodes(node_id) ON DELETE SET NULL,
    sequence_number BIGINT NOT NULL DEFAULT 0,
    quality DECIMAL(3,2) DEFAULT 0.0,
    hop_count INTEGER DEFAULT 0,
    flags INTEGER DEFAULT 0,
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_best_route BOOLEAN DEFAULT false,
    
    CONSTRAINT quality_check CHECK (quality >= 0 AND quality <= 1),
    UNIQUE(node_id, destination_id, next_hop_id)
);

-- Routing metrics history
CREATE TABLE network.routing_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entry_id UUID REFERENCES network.routing_entries(entry_id) ON DELETE CASCADE,
    metric_type VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT metric_type_check CHECK (metric_type IN (
        'latency', 'bandwidth', 'packet_loss', 'jitter', 'quality', 'hop_count'
    ))
);

-- ============================================================================
-- CONNECTION POOL MANAGEMENT
-- ============================================================================

-- Connection pools
CREATE TABLE network.connection_pools (
    pool_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id UUID REFERENCES network.nodes(node_id) ON DELETE CASCADE,
    pool_name VARCHAR(255) NOT NULL,
    min_connections INTEGER DEFAULT 5,
    max_connections INTEGER DEFAULT 100,
    current_connections INTEGER DEFAULT 0,
    in_use_connections INTEGER DEFAULT 0,
    keepalive_timeout INTEGER DEFAULT 300,
    enable_http2 BOOLEAN DEFAULT true,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT status_check CHECK (status IN ('active', 'draining', 'stopped'))
);

-- Individual connections
CREATE TABLE network.connections (
    connection_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pool_id UUID REFERENCES network.connection_pools(pool_id) ON DELETE CASCADE,
    endpoint VARCHAR(255) NOT NULL,
    protocol VARCHAR(50) DEFAULT 'http/1.1',
    state VARCHAR(50) DEFAULT 'idle',
    requests_handled BIGINT DEFAULT 0,
    bytes_sent BIGINT DEFAULT 0,
    bytes_received BIGINT DEFAULT 0,
    errors_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP WITH TIME ZONE,
    closed_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT protocol_check CHECK (protocol IN ('http/1.1', 'http/2', 'http/3', 'websocket')),
    CONSTRAINT state_check CHECK (state IN ('idle', 'active', 'closing', 'closed'))
);

-- ============================================================================
-- OPTIMIZATION AND COMPRESSION
-- ============================================================================

-- Optimization parameters
CREATE TABLE network.optimization_params (
    param_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id UUID REFERENCES network.nodes(node_id) ON DELETE CASCADE,
    param_set_name VARCHAR(255) NOT NULL,
    batch_size INTEGER DEFAULT 50,
    compression_algorithm VARCHAR(50) DEFAULT 'lz4',
    connection_pool_size INTEGER DEFAULT 20,
    retry_strategy VARCHAR(50) DEFAULT 'exponential',
    max_retries INTEGER DEFAULT 3,
    circuit_breaker_threshold DECIMAL(3,2) DEFAULT 0.5,
    adaptive_learning_enabled BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    active_since TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT compression_check CHECK (compression_algorithm IN (
        'none', 'gzip', 'lz4', 'brotli', 'snappy', 'zstd'
    )),
    CONSTRAINT retry_strategy_check CHECK (retry_strategy IN (
        'exponential', 'linear', 'fixed', 'adaptive'
    ))
);

-- Compression statistics
CREATE TABLE network.compression_stats (
    stat_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id UUID REFERENCES network.nodes(node_id) ON DELETE CASCADE,
    algorithm VARCHAR(50) NOT NULL,
    messages_compressed BIGINT DEFAULT 0,
    original_bytes BIGINT DEFAULT 0,
    compressed_bytes BIGINT DEFAULT 0,
    compression_time_ms BIGINT DEFAULT 0,
    decompression_time_ms BIGINT DEFAULT 0,
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    CONSTRAINT algorithm_check CHECK (algorithm IN (
        'none', 'gzip', 'lz4', 'brotli', 'snappy', 'zstd'
    ))
);

-- ============================================================================
-- MESSAGE BATCHING
-- ============================================================================

-- Batch configurations
CREATE TABLE network.batch_configs (
    config_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id UUID REFERENCES network.nodes(node_id) ON DELETE CASCADE,
    max_batch_size INTEGER DEFAULT 100,
    max_wait_time_ms INTEGER DEFAULT 50,
    priority_threshold DECIMAL(3,2) DEFAULT 0.8,
    adaptive_sizing BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Batch metrics
CREATE TABLE network.batch_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_id UUID REFERENCES network.batch_configs(config_id) ON DELETE CASCADE,
    total_batches BIGINT DEFAULT 0,
    total_messages BIGINT DEFAULT 0,
    average_batch_size DECIMAL(10,2),
    priority_bypasses BIGINT DEFAULT 0,
    queue_overflows BIGINT DEFAULT 0,
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL
);

-- ============================================================================
-- NETWORK METRICS AND TELEMETRY
-- ============================================================================

-- Real-time network metrics
CREATE TABLE network.metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id UUID REFERENCES network.nodes(node_id) ON DELETE CASCADE,
    metric_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(20,6) NOT NULL,
    unit VARCHAR(50),
    tags JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_metrics_node_time (node_id, recorded_at DESC),
    INDEX idx_metrics_type_name (metric_type, metric_name)
);

-- Network events
CREATE TABLE network.events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id UUID REFERENCES network.nodes(node_id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,
    event_name VARCHAR(255) NOT NULL,
    severity VARCHAR(50) DEFAULT 'info',
    description TEXT,
    metadata JSONB DEFAULT '{}',
    occurred_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT severity_check CHECK (severity IN ('debug', 'info', 'warning', 'error', 'critical'))
);

-- Performance snapshots
CREATE TABLE network.performance_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    topology_id UUID REFERENCES network.mesh_topologies(topology_id) ON DELETE CASCADE,
    total_nodes INTEGER NOT NULL,
    total_links INTEGER NOT NULL,
    average_latency DECIMAL(10,2),
    network_diameter INTEGER,
    clustering_coefficient DECIMAL(5,4),
    connectivity_ratio DECIMAL(5,4),
    fault_tolerance_score DECIMAL(5,4),
    load_balance_index DECIMAL(5,4),
    partition_resilience DECIMAL(5,4),
    quantum_coherence DECIMAL(5,4),
    snapshot_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- DNS AND SERVICE DISCOVERY
-- ============================================================================

-- DNS cache entries
CREATE TABLE network.dns_cache (
    cache_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hostname VARCHAR(255) NOT NULL,
    ip_addresses INET[] NOT NULL,
    record_type VARCHAR(10) DEFAULT 'A',
    ttl INTEGER DEFAULT 300,
    cached_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    
    CONSTRAINT record_type_check CHECK (record_type IN ('A', 'AAAA', 'CNAME', 'SRV')),
    UNIQUE(hostname, record_type)
);

-- Service registry
CREATE TABLE network.service_registry (
    service_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(255) NOT NULL,
    service_type VARCHAR(100) NOT NULL,
    endpoints TEXT[] NOT NULL,
    health_check_url VARCHAR(500),
    metadata JSONB DEFAULT '{}',
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_health_check TIMESTAMP WITH TIME ZONE,
    health_status VARCHAR(50) DEFAULT 'unknown',
    
    CONSTRAINT health_status_check CHECK (health_status IN ('healthy', 'unhealthy', 'degraded', 'unknown'))
);

-- ============================================================================
-- OPTIMIZATION HISTORY
-- ============================================================================

-- Topology optimizations
CREATE TABLE network.topology_optimizations (
    optimization_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    topology_id UUID REFERENCES network.mesh_topologies(topology_id) ON DELETE CASCADE,
    optimization_type VARCHAR(50) NOT NULL,
    algorithm_used VARCHAR(100),
    metrics_before JSONB NOT NULL,
    metrics_after JSONB NOT NULL,
    improvement_percentage DECIMAL(5,2),
    execution_time_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    performed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT optimization_type_check CHECK (optimization_type IN (
        'automatic', 'manual', 'scheduled', 'triggered'
    ))
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Node indexes
CREATE INDEX idx_nodes_status ON network.nodes(status);
CREATE INDEX idx_nodes_last_seen ON network.nodes(last_seen DESC);
CREATE INDEX idx_nodes_type_role ON network.nodes(node_type, role);

-- Link indexes
CREATE INDEX idx_links_topology ON network.mesh_links(topology_id);
CREATE INDEX idx_links_state ON network.mesh_links(link_state);
CREATE INDEX idx_links_quality ON network.mesh_links(quality DESC);
CREATE INDEX idx_links_nodes ON network.mesh_links(local_node_id, remote_node_id);

-- Routing indexes
CREATE INDEX idx_routing_destination ON network.routing_entries(destination_id);
CREATE INDEX idx_routing_best ON network.routing_entries(is_best_route) WHERE is_best_route = true;

-- Metrics indexes
CREATE INDEX idx_metrics_time_window ON network.metrics(recorded_at DESC);
CREATE INDEX idx_metrics_node_type ON network.metrics(node_id, metric_type);

-- Performance indexes
CREATE INDEX idx_perf_snapshots_time ON network.performance_snapshots(snapshot_time DESC);
CREATE INDEX idx_events_severity_time ON network.events(severity, occurred_at DESC);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION network.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_nodes_timestamp
    BEFORE UPDATE ON network.nodes
    FOR EACH ROW
    EXECUTE FUNCTION network.update_updated_at();

CREATE TRIGGER update_topologies_timestamp
    BEFORE UPDATE ON network.mesh_topologies
    FOR EACH ROW
    EXECUTE FUNCTION network.update_updated_at();

-- Best route maintenance trigger
CREATE OR REPLACE FUNCTION network.maintain_best_routes()
RETURNS TRIGGER AS $$
BEGIN
    -- Mark all routes to this destination as not best
    UPDATE network.routing_entries
    SET is_best_route = false
    WHERE node_id = NEW.node_id 
      AND destination_id = NEW.destination_id
      AND entry_id != NEW.entry_id;
    
    -- Mark the new/updated route as best if it has highest quality
    IF NEW.quality = (
        SELECT MAX(quality)
        FROM network.routing_entries
        WHERE node_id = NEW.node_id 
          AND destination_id = NEW.destination_id
    ) THEN
        NEW.is_best_route = true;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER maintain_best_routes_trigger
    BEFORE INSERT OR UPDATE ON network.routing_entries
    FOR EACH ROW
    EXECUTE FUNCTION network.maintain_best_routes();

-- DNS cache cleanup trigger
CREATE OR REPLACE FUNCTION network.cleanup_expired_dns()
RETURNS void AS $$
BEGIN
    DELETE FROM network.dns_cache
    WHERE expires_at < CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Active connections view
CREATE VIEW network.active_connections AS
SELECT 
    n.node_name,
    cp.pool_name,
    cp.current_connections,
    cp.in_use_connections,
    cp.max_connections,
    ROUND((cp.in_use_connections::numeric / cp.max_connections) * 100, 2) as utilization_percent
FROM network.connection_pools cp
JOIN network.nodes n ON cp.node_id = n.node_id
WHERE cp.status = 'active';

-- Network health view
CREATE VIEW network.network_health AS
SELECT 
    COUNT(DISTINCT n.node_id) as total_nodes,
    COUNT(DISTINCT CASE WHEN n.status = 'active' THEN n.node_id END) as active_nodes,
    COUNT(DISTINCT ml.link_id) as total_links,
    COUNT(DISTINCT CASE WHEN ml.link_state = 'active' THEN ml.link_id END) as active_links,
    AVG(ml.quality) as avg_link_quality,
    AVG(ml.latency_ms) as avg_latency_ms,
    AVG(ml.packet_loss) as avg_packet_loss
FROM network.nodes n
LEFT JOIN network.mesh_links ml ON n.node_id IN (ml.local_node_id, ml.remote_node_id);

-- Best routes view
CREATE VIEW network.best_routes AS
SELECT 
    n1.node_name as source_node,
    n2.node_name as destination_node,
    n3.node_name as next_hop,
    re.quality,
    re.hop_count,
    re.last_seen
FROM network.routing_entries re
JOIN network.nodes n1 ON re.node_id = n1.node_id
JOIN network.nodes n2 ON re.destination_id = n2.node_id
LEFT JOIN network.nodes n3 ON re.next_hop_id = n3.node_id
WHERE re.is_best_route = true;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Calculate network statistics
CREATE OR REPLACE FUNCTION network.calculate_network_stats(p_topology_id UUID)
RETURNS TABLE (
    total_nodes INTEGER,
    total_links INTEGER,
    avg_degree NUMERIC,
    network_diameter INTEGER,
    clustering_coefficient NUMERIC
) AS $$
DECLARE
    v_total_nodes INTEGER;
    v_total_links INTEGER;
BEGIN
    -- Get basic counts
    SELECT COUNT(DISTINCT node_id) INTO v_total_nodes
    FROM network.nodes n
    WHERE EXISTS (
        SELECT 1 FROM network.mesh_links ml
        WHERE ml.topology_id = p_topology_id
        AND n.node_id IN (ml.local_node_id, ml.remote_node_id)
    );
    
    SELECT COUNT(*) INTO v_total_links
    FROM network.mesh_links
    WHERE topology_id = p_topology_id
    AND link_state = 'active';
    
    RETURN QUERY
    SELECT 
        v_total_nodes,
        v_total_links,
        CASE WHEN v_total_nodes > 0 
            THEN (2.0 * v_total_links / v_total_nodes)::NUMERIC 
            ELSE 0 
        END,
        0, -- Network diameter calculation would be more complex
        0.0; -- Clustering coefficient calculation would be more complex
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PERMISSIONS
-- ============================================================================

-- Grant permissions to application user
GRANT ALL PRIVILEGES ON SCHEMA network TO csp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA network TO csp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA network TO csp_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA network TO csp_user;

-- ============================================================================
-- SAMPLE DATA (Optional - for testing)
-- ============================================================================

-- Insert sample nodes
INSERT INTO network.nodes (node_name, node_type, role, address, port, status) VALUES
('csp-node-01', 'peer', 'coordinator', '10.0.1.10', 8080, 'active'),
('csp-node-02', 'peer', 'super_peer', '10.0.1.11', 8080, 'active'),
('csp-node-03', 'relay', 'relay', '10.0.1.12', 8080, 'active'),
('csp-gateway-01', 'gateway', 'gateway', '10.0.1.20', 8443, 'active');

-- Insert sample topology
INSERT INTO network.mesh_topologies (mesh_name, topology_type, optimization_enabled) VALUES
('production-mesh', 'adaptive_hybrid', true);

-- Create sample links
INSERT INTO network.mesh_links (
    topology_id, 
    local_node_id, 
    remote_node_id, 
    link_state, 
    quality, 
    latency_ms
)
SELECT 
    (SELECT topology_id FROM network.mesh_topologies WHERE mesh_name = 'production-mesh'),
    n1.node_id,
    n2.node_id,
    'active',
    0.95,
    5.0
FROM network.nodes n1
CROSS JOIN network.nodes n2
WHERE n1.node_id < n2.node_id
AND n1.status = 'active'
AND n2.status = 'active';
