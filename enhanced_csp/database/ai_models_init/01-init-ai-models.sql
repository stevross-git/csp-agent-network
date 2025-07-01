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
