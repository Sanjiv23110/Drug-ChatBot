import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import AzureOpenAI
from app.core.config import settings
from typing import List, Dict, Any
import time

class VectorStoreService:
    def __init__(self):
        # Initialize Azure OpenAI client for embeddings
        self.openai_client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
        )
        
        # Initialize ChromaDB with PersistentClient (saves to disk!)
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_DIR
        )
        
        # Get or create collection with new name for Azure OpenAI
        self.collection = self.chroma_client.get_or_create_collection(
            name="drug_monographs_azure",
            metadata={"hnsw:space": "cosine"}
        )
    
    def _embed_text(self, text: str):
        """Generate embedding using Azure OpenAI"""
        response = self.openai_client.embeddings.create(
            input=text,
            model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )
        return response.data[0].embedding

    def add_documents(self, documents: list):
        """
        documents: List of dicts with 'text', 'id', 'metadata'
        """
        if not documents:
            return
            
        import time
        # Process in batches
        batch_size = 10
        total_docs = len(documents)
        
        print(f"Adding {total_docs} documents in batches of {batch_size}...")
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc['text'] for doc in batch]
            ids = [doc['id'] for doc in batch]
            metadatas = [doc['metadata'] for doc in batch]
            
            try:
                # Generate embeddings using hash function
                embeddings = [self._embed_text(text) for text in texts]
                
                self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Processed batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}")
                time.sleep(1.0)  # Wait 1 second between batches for Cohere rate limits (100/min)
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                time.sleep(5.0)  # Wait longer on error before retrying
                
        print(f"Finished adding documents to ChromaDB.")

    def query(self, query_text: str, n_results: int = 5):
        """Query the vector store"""
        # Embed the query using Azure OpenAI
        query_embedding = self._embed_text(query_text)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
