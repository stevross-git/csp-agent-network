import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class ChromaVectorStore:
    def __init__(self, collection_name="csp_memory", persist_directory="./chroma_data"):
        embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.client = chromadb.Client(chromadb.config.Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

    def store(self, key, content, metadata=None):
        self.collection.add(
            documents=[content],
            ids=[key],
            metadatas=[metadata or {}]
        )
        print(f"[ChromaDB] Stored {key}")

    def search(self, query_text, top_k=3):
        result = self.collection.query(query_texts=[query_text], n_results=top_k)
        print(f"[ChromaDB] Query result for '{query_text}': {result}")
        return result