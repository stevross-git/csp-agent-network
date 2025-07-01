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
