# backend/services/vector_database.py
"""
Vector Database Integration Service
==================================
Unified interface for multiple vector databases (Chroma, Qdrant, Weaviate, pgvector)
"""

import uuid
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import asyncpg
import httpx
import json

logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

class VectorDBType(str, Enum):
    CHROMA = "chroma"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    PGVECTOR = "pgvector"

@dataclass
class Document:
    """Document for vector storage"""
    id: str
    content: str
    metadata: Dict[str, Any] = None
    embedding: Optional[List[float]] = None
    source: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass 
class SearchResult:
    """Vector search result"""
    id: str
    content: str
    similarity: float
    metadata: Dict[str, Any] = None
    source: Optional[str] = None

@dataclass
class Collection:
    """Vector database collection"""
    name: str
    description: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class VectorDatabaseInterface(ABC):
    """Abstract interface for vector databases"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the vector database"""
        pass
    
    @abstractmethod
    async def create_collection(self, collection: Collection) -> bool:
        """Create a new collection"""
        pass
    
    @abstractmethod
    async def add_documents(self, collection_name: str, documents: List[Document]) -> bool:
        """Add documents to a collection"""
        pass
    
    @abstractmethod
    async def search(self, collection_name: str, query_text: str, limit: int = 10, threshold: float = 0.8) -> List[SearchResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    async def search_by_embedding(self, collection_name: str, embedding: List[float], limit: int = 10, threshold: float = 0.8) -> List[SearchResult]:
        """Search using a vector embedding"""
        pass
    
    @abstractmethod
    async def delete_document(self, collection_name: str, document_id: str) -> bool:
        """Delete a document from collection"""
        pass
    
    @abstractmethod
    async def list_collections(self) -> List[str]:
        """List all collections"""
        pass
    
    @abstractmethod
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information"""
        pass

# ============================================================================
# CHROMA IMPLEMENTATION
# ============================================================================

class ChromaVectorDB(VectorDatabaseInterface):
    """ChromaDB implementation"""
    
    def __init__(self, host: str = "chroma", port: int = 8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.client = None
        
    async def connect(self) -> bool:
        """Connect to ChromaDB"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/v1/heartbeat")
                if response.status_code == 200:
                    logger.info("✅ Connected to ChromaDB")
                    return True
                return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to ChromaDB: {e}")
            return False
    
    async def create_collection(self, collection: Collection) -> bool:
        """Create a new collection in ChromaDB"""
        try:
            payload = {
                "name": collection.name,
                "metadata": {**collection.metadata, "description": collection.description}
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/collections",
                    json=payload
                )
                
                if response.status_code in [200, 201]:
                    logger.info(f"✅ Created ChromaDB collection: {collection.name}")
                    return True
                elif response.status_code == 409:
                    logger.info(f"⚠️ ChromaDB collection already exists: {collection.name}")
                    return True
                else:
                    logger.error(f"❌ Failed to create ChromaDB collection: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error creating ChromaDB collection: {e}")
            return False
    
    async def add_documents(self, collection_name: str, documents: List[Document]) -> bool:
        """Add documents to ChromaDB collection"""
        try:
            payload = {
                "ids": [doc.id for doc in documents],
                "documents": [doc.content for doc in documents],
                "metadatas": [doc.metadata for doc in documents]
            }
            
            # Add embeddings if provided
            if any(doc.embedding for doc in documents):
                payload["embeddings"] = [doc.embedding for doc in documents]
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/collections/{collection_name}/add",
                    json=payload
                )
                
                if response.status_code in [200, 201]:
                    logger.info(f"✅ Added {len(documents)} documents to ChromaDB collection: {collection_name}")
                    return True
                else:
                    logger.error(f"❌ Failed to add documents to ChromaDB: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error adding documents to ChromaDB: {e}")
            return False
    
    async def search(self, collection_name: str, query_text: str, limit: int = 10, threshold: float = 0.8) -> List[SearchResult]:
        """Search ChromaDB collection"""
        try:
            payload = {
                "query_texts": [query_text],
                "n_results": limit,
                "include": ["documents", "metadatas", "distances"]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/collections/{collection_name}/query",
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    if data.get("ids") and len(data["ids"]) > 0:
                        for i, doc_id in enumerate(data["ids"][0]):
                            similarity = 1.0 - data["distances"][0][i]  # Convert distance to similarity
                            if similarity >= threshold:
                                results.append(SearchResult(
                                    id=doc_id,
                                    content=data["documents"][0][i],
                                    similarity=similarity,
                                    metadata=data["metadatas"][0][i] if data.get("metadatas") else {}
                                ))
                    
                    logger.info(f"✅ ChromaDB search found {len(results)} results")
                    return results
                else:
                    logger.error(f"❌ ChromaDB search failed: {response.text}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error searching ChromaDB: {e}")
            return []
    
    async def search_by_embedding(self, collection_name: str, embedding: List[float], limit: int = 10, threshold: float = 0.8) -> List[SearchResult]:
        """Search by embedding vector"""
        try:
            payload = {
                "query_embeddings": [embedding],
                "n_results": limit,
                "include": ["documents", "metadatas", "distances"]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/collections/{collection_name}/query",
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    if data.get("ids") and len(data["ids"]) > 0:
                        for i, doc_id in enumerate(data["ids"][0]):
                            similarity = 1.0 - data["distances"][0][i]
                            if similarity >= threshold:
                                results.append(SearchResult(
                                    id=doc_id,
                                    content=data["documents"][0][i],
                                    similarity=similarity,
                                    metadata=data["metadatas"][0][i] if data.get("metadatas") else {}
                                ))
                    
                    return results
                else:
                    logger.error(f"❌ ChromaDB embedding search failed: {response.text}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error searching ChromaDB by embedding: {e}")
            return []
    
    async def delete_document(self, collection_name: str, document_id: str) -> bool:
        """Delete document from ChromaDB"""
        try:
            payload = {"ids": [document_id]}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/collections/{collection_name}/delete",
                    json=payload
                )
                
                if response.status_code in [200, 204]:
                    logger.info(f"✅ Deleted document {document_id} from ChromaDB")
                    return True
                else:
                    logger.error(f"❌ Failed to delete document from ChromaDB: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error deleting document from ChromaDB: {e}")
            return False
    
    async def list_collections(self) -> List[str]:
        """List ChromaDB collections"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/v1/collections")
                
                if response.status_code == 200:
                    collections = response.json()
                    return [col["name"] for col in collections]
                else:
                    logger.error(f"❌ Failed to list ChromaDB collections: {response.text}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error listing ChromaDB collections: {e}")
            return []
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get ChromaDB collection info"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/v1/collections/{collection_name}")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"❌ Failed to get ChromaDB collection info: {response.text}")
                    return {}
                    
        except Exception as e:
            logger.error(f"❌ Error getting ChromaDB collection info: {e}")
            return {}

# ============================================================================
# QDRANT IMPLEMENTATION
# ============================================================================

class QdrantVectorDB(VectorDatabaseInterface):
    """Qdrant implementation"""
    
    def __init__(self, host: str = "qdrant", port: int = 6333):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
    
    async def connect(self) -> bool:
        """Connect to Qdrant"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    logger.info("✅ Connected to Qdrant")
                    return True
                return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            return False
    
    async def create_collection(self, collection: Collection) -> bool:
        """Create Qdrant collection"""
        try:
            payload = {
                "vectors": {
                    "size": 1536,  # OpenAI embedding dimension
                    "distance": "Cosine"
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.base_url}/collections/{collection.name}",
                    json=payload
                )
                
                if response.status_code in [200, 201]:
                    logger.info(f"✅ Created Qdrant collection: {collection.name}")
                    return True
                elif response.status_code == 409:
                    logger.info(f"⚠️ Qdrant collection already exists: {collection.name}")
                    return True
                else:
                    logger.error(f"❌ Failed to create Qdrant collection: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error creating Qdrant collection: {e}")
            return False
    
    async def add_documents(self, collection_name: str, documents: List[Document]) -> bool:
        """Add documents to Qdrant"""
        try:
            points = []
            for doc in documents:
                if doc.embedding:
                    points.append({
                        "id": doc.id,
                        "vector": doc.embedding,
                        "payload": {
                            "content": doc.content,
                            "metadata": doc.metadata,
                            "source": doc.source,
                            "created_at": doc.created_at.isoformat() if doc.created_at else None
                        }
                    })
            
            if not points:
                logger.warning("No documents with embeddings to add to Qdrant")
                return False
            
            payload = {"points": points}
            
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.base_url}/collections/{collection_name}/points",
                    json=payload
                )
                
                if response.status_code in [200, 201]:
                    logger.info(f"✅ Added {len(points)} documents to Qdrant collection: {collection_name}")
                    return True
                else:
                    logger.error(f"❌ Failed to add documents to Qdrant: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error adding documents to Qdrant: {e}")
            return False
    
    async def search_by_embedding(self, collection_name: str, embedding: List[float], limit: int = 10, threshold: float = 0.8) -> List[SearchResult]:
        """Search Qdrant by embedding"""
        try:
            payload = {
                "vector": embedding,
                "limit": limit,
                "score_threshold": threshold,
                "with_payload": True
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/collections/{collection_name}/points/search",
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    for result in data.get("result", []):
                        payload = result.get("payload", {})
                        results.append(SearchResult(
                            id=str(result["id"]),
                            content=payload.get("content", ""),
                            similarity=result["score"],
                            metadata=payload.get("metadata", {}),
                            source=payload.get("source")
                        ))
                    
                    logger.info(f"✅ Qdrant search found {len(results)} results")
                    return results
                else:
                    logger.error(f"❌ Qdrant search failed: {response.text}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error searching Qdrant: {e}")
            return []
    
    async def search(self, collection_name: str, query_text: str, limit: int = 10, threshold: float = 0.8) -> List[SearchResult]:
        """Search Qdrant (requires embedding generation)"""
        # This would require an embedding service to convert text to vector
        # For now, return empty results with a note
        logger.warning("Text search in Qdrant requires embedding generation - use search_by_embedding instead")
        return []
    
    async def delete_document(self, collection_name: str, document_id: str) -> bool:
        """Delete document from Qdrant"""
        try:
            payload = {
                "points": [document_id]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/collections/{collection_name}/points/delete",
                    json=payload
                )
                
                if response.status_code in [200, 204]:
                    logger.info(f"✅ Deleted document {document_id} from Qdrant")
                    return True
                else:
                    logger.error(f"❌ Failed to delete document from Qdrant: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error deleting document from Qdrant: {e}")
            return False
    
    async def list_collections(self) -> List[str]:
        """List Qdrant collections"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/collections")
                
                if response.status_code == 200:
                    data = response.json()
                    return [col["name"] for col in data.get("result", {}).get("collections", [])]
                else:
                    logger.error(f"❌ Failed to list Qdrant collections: {response.text}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error listing Qdrant collections: {e}")
            return []
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get Qdrant collection info"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/collections/{collection_name}")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"❌ Failed to get Qdrant collection info: {response.text}")
                    return {}
                    
        except Exception as e:
            logger.error(f"❌ Error getting Qdrant collection info: {e}")
            return {}

# ============================================================================
# PGVECTOR IMPLEMENTATION
# ============================================================================

class PgVectorDB(VectorDatabaseInterface):
    """PostgreSQL with pgvector extension implementation"""
    
    def __init__(self, host: str = "postgres_vector", port: int = 5432, 
                 database: str = "vector_db", user: str = "vector_user", 
                 password: str = "vector_password"):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool = None
    
    async def connect(self) -> bool:
        """Connect to PostgreSQL with pgvector"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=1,
                max_size=10
            )
            logger.info("✅ Connected to PostgreSQL with pgvector")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to pgvector: {e}")
            return False
    
    async def create_collection(self, collection: Collection) -> bool:
        """Create table for collection in pgvector"""
        if not self.pool:
            await self.connect()
        
        try:
            async with self.pool.acquire() as conn:
                # Create table for the collection
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {collection.name} (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        document_id VARCHAR(255) NOT NULL,
                        content TEXT NOT NULL,
                        embedding vector(1536),
                        metadata JSONB DEFAULT '{{}}',
                        source VARCHAR(255),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # Create index for similarity search
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {collection.name}_embedding_idx 
                    ON {collection.name} USING ivfflat (embedding vector_cosine_ops) 
                    WITH (lists = 100)
                """)
                
                logger.info(f"✅ Created pgvector collection table: {collection.name}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error creating pgvector collection: {e}")
            return False
    
    async def add_documents(self, collection_name: str, documents: List[Document]) -> bool:
        """Add documents to pgvector table"""
        if not self.pool:
            await self.connect()
        
        try:
            async with self.pool.acquire() as conn:
                for doc in documents:
                    if doc.embedding:
                        await conn.execute(f"""
                            INSERT INTO {collection_name} 
                            (document_id, content, embedding, metadata, source, created_at)
                            VALUES ($1, $2, $3, $4, $5, $6)
                            ON CONFLICT (document_id) DO UPDATE SET
                                content = EXCLUDED.content,
                                embedding = EXCLUDED.embedding,
                                metadata = EXCLUDED.metadata,
                                source = EXCLUDED.source
                        """, 
                        doc.id, doc.content, doc.embedding, 
                        json.dumps(doc.metadata), doc.source, doc.created_at)
                
                logger.info(f"✅ Added {len(documents)} documents to pgvector table: {collection_name}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error adding documents to pgvector: {e}")
            return False
    
    async def search_by_embedding(self, collection_name: str, embedding: List[float], limit: int = 10, threshold: float = 0.8) -> List[SearchResult]:
        """Search pgvector by embedding"""
        if not self.pool:
            await self.connect()
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(f"""
                    SELECT 
                        document_id,
                        content,
                        1 - (embedding <=> $1) as similarity,
                        metadata,
                        source
                    FROM {collection_name}
                    WHERE 1 - (embedding <=> $1) > $2
                    ORDER BY embedding <=> $1
                    LIMIT $3
                """, embedding, threshold, limit)
                
                results = []
                for row in rows:
                    results.append(SearchResult(
                        id=row['document_id'],
                        content=row['content'],
                        similarity=float(row['similarity']),
                        metadata=row['metadata'] or {},
                        source=row['source']
                    ))
                
                logger.info(f"✅ pgvector search found {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"❌ Error searching pgvector: {e}")
            return []
    
    async def search(self, collection_name: str, query_text: str, limit: int = 10, threshold: float = 0.8) -> List[SearchResult]:
        """Search pgvector (requires embedding generation)"""
        logger.warning("Text search in pgvector requires embedding generation - use search_by_embedding instead")
        return []
    
    async def delete_document(self, collection_name: str, document_id: str) -> bool:
        """Delete document from pgvector"""
        if not self.pool:
            await self.connect()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(f"""
                    DELETE FROM {collection_name} WHERE document_id = $1
                """, document_id)
                
                if result == "DELETE 1":
                    logger.info(f"✅ Deleted document {document_id} from pgvector")
                    return True
                else:
                    logger.warning(f"⚠️ Document {document_id} not found in pgvector")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error deleting document from pgvector: {e}")
            return False
    
    async def list_collections(self) -> List[str]:
        """List pgvector tables (collections)"""
        if not self.pool:
            await self.connect()
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_type = 'BASE TABLE'
                    AND table_name != 'spatial_ref_sys'
                """)
                
                return [row['table_name'] for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Error listing pgvector collections: {e}")
            return []
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get pgvector table info"""
        if not self.pool:
            await self.connect()
        
        try:
            async with self.pool.acquire() as conn:
                count_row = await conn.fetchrow(f"SELECT COUNT(*) as count FROM {collection_name}")
                
                return {
                    "name": collection_name,
                    "type": "pgvector_table",
                    "document_count": count_row['count'] if count_row else 0
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting pgvector collection info: {e}")
            return {}

# ============================================================================
# UNIFIED VECTOR DATABASE MANAGER
# ============================================================================

class VectorDatabaseManager:
    """Unified manager for multiple vector databases"""
    
    def __init__(self):
        self.databases: Dict[VectorDBType, VectorDatabaseInterface] = {}
        self.primary_db: Optional[VectorDBType] = None
        self.connected_dbs: List[VectorDBType] = []
    
    def add_database(self, db_type: VectorDBType, db_instance: VectorDatabaseInterface):
        """Add a vector database instance"""
        self.databases[db_type] = db_instance
        logger.info(f"Added {db_type} vector database")
    
    def set_primary(self, db_type: VectorDBType):
        """Set primary vector database"""
        if db_type in self.databases:
            self.primary_db = db_type
            logger.info(f"Set {db_type} as primary vector database")
        else:
            logger.error(f"Database {db_type} not found")
    
    async def connect_all(self) -> Dict[VectorDBType, bool]:
        """Connect to all configured databases"""
        results = {}
        
        for db_type, db_instance in self.databases.items():
            try:
                connected = await db_instance.connect()
                results[db_type] = connected
                if connected:
                    self.connected_dbs.append(db_type)
                    logger.info(f"✅ Connected to {db_type}")
                else:
                    logger.warning(f"⚠️ Failed to connect to {db_type}")
            except Exception as e:
                logger.error(f"❌ Error connecting to {db_type}: {e}")
                results[db_type] = False
        
        return results
    
    async def get_primary_db(self) -> Optional[VectorDatabaseInterface]:
        """Get primary database instance"""
        if self.primary_db and self.primary_db in self.connected_dbs:
            return self.databases[self.primary_db]
        elif self.connected_dbs:
            # Return first connected database as fallback
            return self.databases[self.connected_dbs[0]]
        return None
    
    async def add_documents(self, collection_name: str, documents: List[Document], db_type: Optional[VectorDBType] = None) -> bool:
        """Add documents to specified or primary database"""
        target_db = self.databases.get(db_type) if db_type else await self.get_primary_db()
        
        if not target_db:
            logger.error("No target database available")
            return False
        
        return await target_db.add_documents(collection_name, documents)
    
    async def search(self, collection_name: str, query_text: str, limit: int = 10, 
                    threshold: float = 0.8, db_type: Optional[VectorDBType] = None) -> List[SearchResult]:
        """Search in specified or primary database"""
        target_db = self.databases.get(db_type) if db_type else await self.get_primary_db()
        
        if not target_db:
            logger.error("No target database available")
            return []
        
        return await target_db.search(collection_name, query_text, limit, threshold)
    
    async def search_by_embedding(self, collection_name: str, embedding: List[float], 
                                 limit: int = 10, threshold: float = 0.8, 
                                 db_type: Optional[VectorDBType] = None) -> List[SearchResult]:
        """Search by embedding in specified or primary database"""
        target_db = self.databases.get(db_type) if db_type else await self.get_primary_db()
        
        if not target_db:
            logger.error("No target database available")
            return []
        
        return await target_db.search_by_embedding(collection_name, embedding, limit, threshold)
    
    async def search_all_databases(self, collection_name: str, query_text: str, 
                                  limit: int = 10, threshold: float = 0.8) -> Dict[VectorDBType, List[SearchResult]]:
        """Search across all connected databases"""
        results = {}
        
        for db_type in self.connected_dbs:
            try:
                db_instance = self.databases[db_type]
                search_results = await db_instance.search(collection_name, query_text, limit, threshold)
                results[db_type] = search_results
            except Exception as e:
                logger.error(f"Error searching {db_type}: {e}")
                results[db_type] = []
        
        return results
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all vector databases"""
        status = {
            "primary_db": self.primary_db.value if self.primary_db else None,
            "connected_databases": [db.value for db in self.connected_dbs],
            "database_status": {}
        }
        
        for db_type, db_instance in self.databases.items():
            try:
                connected = await db_instance.connect()
                collections = await db_instance.list_collections()
                status["database_status"][db_type.value] = {
                    "connected": connected,
                    "collections": collections,
                    "collection_count": len(collections)
                }
            except Exception as e:
                status["database_status"][db_type.value] = {
                    "connected": False,
                    "error": str(e),
                    "collections": [],
                    "collection_count": 0
                }
        
        return status

# ============================================================================
# EMBEDDING SERVICE INTEGRATION
# ============================================================================

class EmbeddingService:
    """Service for generating embeddings using various providers"""
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text"""
        try:
            if self.provider == "openai":
                return await self._generate_openai_embedding(text)
            elif self.provider == "huggingface":
                return await self._generate_huggingface_embedding(text)
            else:
                logger.error(f"Unsupported embedding provider: {self.provider}")
                return None
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def _generate_openai_embedding(self, text: str) -> Optional[List[float]]:
        """Generate OpenAI embedding"""
        if not self.api_key:
            logger.error("OpenAI API key not provided")
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "input": text,
                        "model": "text-embedding-ada-002"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["data"][0]["embedding"]
                else:
                    logger.error(f"OpenAI API error: {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return None
    
    async def _generate_huggingface_embedding(self, text: str) -> Optional[List[float]]:
        """Generate HuggingFace embedding (placeholder for local model)"""
        # This would integrate with a local HuggingFace model
        logger.warning("HuggingFace embedding generation not implemented yet")
        return None

# ============================================================================
# GLOBAL VECTOR DATABASE MANAGER INSTANCE
# ============================================================================

# Global instance
vector_db_manager = VectorDatabaseManager()

async def initialize_vector_databases():
    """Initialize all vector databases"""
    
    # Add database instances
    vector_db_manager.add_database(VectorDBType.CHROMA, ChromaVectorDB())
    vector_db_manager.add_database(VectorDBType.QDRANT, QdrantVectorDB())
    vector_db_manager.add_database(VectorDBType.PGVECTOR, PgVectorDB())
    
    # Set primary database (Chroma as default)
    vector_db_manager.set_primary(VectorDBType.CHROMA)
    
    # Connect to all databases
    connection_results = await vector_db_manager.connect_all()
    
    # Initialize default collections
    await vector_db_manager.initialize_default_collections()
    
    logger.info("✅ Vector databases initialized successfully")
    return connection_results

def get_vector_db_manager() -> VectorDatabaseManager:
    """Get the global vector database manager"""
    return vector_db_manager
    