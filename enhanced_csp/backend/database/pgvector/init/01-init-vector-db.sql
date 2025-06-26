# =============================================================================
# VECTOR DATABASE CONFIGURATION FILES
# =============================================================================

# database/pgvector/init/01-init-vector-db.sql
-- PostgreSQL with pgvector extension initialization

\c vector_db;

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create embeddings table for AI documents
CREATE TABLE IF NOT EXISTS document_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536), -- OpenAI embedding dimension
    metadata JSONB DEFAULT '{}',
    source VARCHAR(255),
    chunk_index INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create conversations embeddings table
CREATE TABLE IF NOT EXISTS conversation_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id VARCHAR(255) NOT NULL,
    message_id VARCHAR(255) NOT NULL,
    message_content TEXT NOT NULL,
    embedding vector(1536),
    role VARCHAR(50), -- 'user', 'assistant', 'system'
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create code embeddings table
CREATE TABLE IF NOT EXISTS code_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_path VARCHAR(500) NOT NULL,
    function_name VARCHAR(255),
    code_content TEXT NOT NULL,
    embedding vector(1536),
    language VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create AI models knowledge embeddings
CREATE TABLE IF NOT EXISTS ai_model_knowledge (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(255) NOT NULL,
    knowledge_type VARCHAR(100), -- 'capability', 'limitation', 'example', 'documentation'
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for similarity search
CREATE INDEX ON document_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON conversation_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON code_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON ai_model_knowledge USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create regular indexes
CREATE INDEX idx_document_embeddings_document_id ON document_embeddings(document_id);
CREATE INDEX idx_conversation_embeddings_conversation_id ON conversation_embeddings(conversation_id);
CREATE INDEX idx_code_embeddings_file_path ON code_embeddings(file_path);
CREATE INDEX idx_ai_model_knowledge_model_id ON ai_model_knowledge(model_id);

-- Create search functions
CREATE OR REPLACE FUNCTION search_similar_documents(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.8,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id uuid,
    document_id varchar,
    content text,
    similarity float,
    metadata jsonb
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        d.id,
        d.document_id,
        d.content,
        1 - (d.embedding <=> query_embedding) as similarity,
        d.metadata
    FROM document_embeddings d
    WHERE 1 - (d.embedding <=> query_embedding) > match_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
$$;

CREATE OR REPLACE FUNCTION search_similar_conversations(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.8,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id uuid,
    conversation_id varchar,
    message_content text,
    similarity float,
    role varchar,
    metadata jsonb
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        c.id,
        c.conversation_id,
        c.message_content,
        1 - (c.embedding <=> query_embedding) as similarity,
        c.role,
        c.metadata
    FROM conversation_embeddings c
    WHERE 1 - (c.embedding <=> query_embedding) > match_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- Insert sample data
INSERT INTO document_embeddings (document_id, content, metadata, source) VALUES
('doc_001', 'How to use Enhanced CSP system for AI model management', '{"type": "documentation", "category": "getting_started"}', 'user_manual'),
('doc_002', 'Best practices for vector database integration', '{"type": "documentation", "category": "best_practices"}', 'user_manual'),
('doc_003', 'Troubleshooting common AI model deployment issues', '{"type": "documentation", "category": "troubleshooting"}', 'user_manual');

# =============================================================================
# QDRANT CONFIGURATION
# =============================================================================

# database/qdrant/config/config.yaml
log_level: INFO

storage:
  storage_path: ./storage
  snapshots_path: ./snapshots
  on_disk_payload: true

service:
  http_port: 6333
  grpc_port: 6334
  max_request_size_mb: 32
  max_workers: 0  # Auto-detect

cluster:
  enabled: false

telemetry_disabled: true

# =============================================================================
# CHROMA INITIALIZATION
# =============================================================================

# database/chroma/init/init_collections.py
#!/usr/bin/env python3
"""
Initialize Chroma collections for Enhanced CSP
"""

import chromadb
import uuid
from datetime import datetime

def initialize_chroma_collections():
    """Initialize default collections in Chroma"""
    
    # Connect to Chroma
    client = chromadb.HttpClient(host="localhost", port=8200)
    
    # Create collections
    collections = [
        {
            "name": "csp_documents",
            "metadata": {"description": "CSP system documentation and guides"},
            "embedding_function": None  # Use default
        },
        {
            "name": "ai_conversations", 
            "metadata": {"description": "AI conversation history and context"},
            "embedding_function": None
        },
        {
            "name": "code_snippets",
            "metadata": {"description": "Code examples and snippets"},
            "embedding_function": None
        },
        {
            "name": "ai_model_knowledge",
            "metadata": {"description": "AI model capabilities and knowledge base"},
            "embedding_function": None
        }
    ]
    
    for collection_config in collections:
        try:
            collection = client.create_collection(
                name=collection_config["name"],
                metadata=collection_config["metadata"]
            )
            print(f"✅ Created collection: {collection_config['name']}")
            
            # Add sample documents
            if collection_config["name"] == "csp_documents":
                collection.add(
                    documents=[
                        "Enhanced CSP system provides a visual interface for designing and managing AI-powered processes.",
                        "Vector databases enable semantic search and retrieval-augmented generation for AI applications.",
                        "The system supports multiple AI models including GPT-4, Claude, and open-source alternatives."
                    ],
                    metadatas=[
                        {"type": "overview", "category": "introduction"},
                        {"type": "technical", "category": "architecture"},
                        {"type": "features", "category": "ai_models"}
                    ],
                    ids=[str(uuid.uuid4()) for _ in range(3)]
                )
                print(f"  ↳ Added sample documents to {collection_config['name']}")
                
        except Exception as e:
            if "already exists" in str(e):
                print(f"⚠️ Collection {collection_config['name']} already exists")
            else:
                print(f"❌ Failed to create collection {collection_config['name']}: {e}")

if __name__ == "__main__":
    initialize_chroma_collections()

# =============================================================================
# WEAVIATE SCHEMA CONFIGURATION
# =============================================================================

# database/weaviate/init/schema.json
{
  "classes": [
    {
      "class": "CSPDocument",
      "description": "CSP system documentation and guides",
      "vectorizer": "text2vec-openai",
      "moduleConfig": {
        "text2vec-openai": {
          "model": "ada",
          "modelVersion": "002",
          "type": "text"
        }
      },
      "properties": [
        {
          "name": "title",
          "dataType": ["text"],
          "description": "Document title"
        },
        {
          "name": "content",
          "dataType": ["text"],
          "description": "Document content"
        },
        {
          "name": "category",
          "dataType": ["text"],
          "description": "Document category"
        },
        {
          "name": "source",
          "dataType": ["text"],
          "description": "Document source"
        },
        {
          "name": "createdAt",
          "dataType": ["date"],
          "description": "Creation timestamp"
        }
      ]
    },
    {
      "class": "AIConversation",
      "description": "AI conversation messages and context",
      "vectorizer": "text2vec-openai",
      "moduleConfig": {
        "text2vec-openai": {
          "model": "ada",
          "modelVersion": "002",
          "type": "text"
        }
      },
      "properties": [
        {
          "name": "conversationId",
          "dataType": ["text"],
          "description": "Conversation identifier"
        },
        {
          "name": "messageContent",
          "dataType": ["text"],
          "description": "Message content"
        },
        {
          "name": "role",
          "dataType": ["text"],
          "description": "Message role (user, assistant, system)"
        },
        {
          "name": "modelUsed",
          "dataType": ["text"],
          "description": "AI model used for response"
        },
        {
          "name": "timestamp",
          "dataType": ["date"],
          "description": "Message timestamp"
        }
      ]
    },
    {
      "class": "CodeSnippet",
      "description": "Code examples and snippets",
      "vectorizer": "text2vec-openai",
      "moduleConfig": {
        "text2vec-openai": {
          "model": "ada",
          "modelVersion": "002",
          "type": "text"
        }
      },
      "properties": [
        {
          "name": "filename",
          "dataType": ["text"],
          "description": "File name or path"
        },
        {
          "name": "codeContent",
          "dataType": ["text"],
          "description": "Code content"
        },
        {
          "name": "language",
          "dataType": ["text"],
          "description": "Programming language"
        },
        {
          "name": "functionality",
          "dataType": ["text"],
          "description": "What the code does"
        },
        {
          "name": "createdAt",
          "dataType": ["date"],
          "description": "Creation timestamp"
        }
      ]
    },
    {
      "class": "AIModelKnowledge",
      "description": "AI model capabilities and knowledge",
      "vectorizer": "text2vec-openai",
      "moduleConfig": {
        "text2vec-openai": {
          "model": "ada",
          "modelVersion": "002",
          "type": "text"
        }
      },
      "properties": [
        {
          "name": "modelId",
          "dataType": ["text"],
          "description": "AI model identifier"
        },
        {
          "name": "knowledgeType",
          "dataType": ["text"],
          "description": "Type of knowledge (capability, limitation, example)"
        },
        {
          "name": "content",
          "dataType": ["text"],
          "description": "Knowledge content"
        },
        {
          "name": "provider",
          "dataType": ["text"],
          "description": "Model provider (OpenAI, Anthropic, etc.)"
        },
        {
          "name": "createdAt",
          "dataType": ["date"],
          "description": "Creation timestamp"
        }
      ]
    }
  ]
}

# =============================================================================
# MILVUS CONFIGURATION
# =============================================================================

# database/milvus/configs/milvus.yaml
etcd:
  endpoints:
    - etcd:2379

minio:
  address: minio
  port: 9000
  accessKeyID: minioadmin
  secretAccessKey: minioadmin
  useSSL: false
  bucketName: "milvus-bucket"
  rootPath: "files"

common:
  defaultPartitionName: "_default"
  defaultIndexName: "_default_idx"
  
rootCoord:
  address: localhost
  port: 53100

dataCoord:
  address: localhost
  port: 13333

queryCoord:
  address: localhost
  port: 19531

indexCoord:
  address: localhost
  port: 31000

dataNode:
  port: 21124

indexNode:
  port: 21121

queryNode:
  port: 21123

proxy:
  port: 19530