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
