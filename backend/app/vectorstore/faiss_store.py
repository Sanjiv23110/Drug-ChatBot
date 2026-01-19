"""
FAISS vector store for medical-grade RAG.

Design principles:
- Dimension validation (1536 for text-embedding-3-small)
- Integer IDs matching SQLite chunk_ids
- Flat index for small corpora, IVFFlat for >100k vectors
- No metadata stored here (SQLite only)
"""
import numpy as np
import faiss
from typing import List, Tuple, Optional
from pathlib import Path


class FAISSVectorStore:
    """FAISS index wrapper with dimension validation and stable IDs."""
    
    def __init__(self, dimension: int = 1536, use_gpu: bool = False):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension (1536 for text-embedding-3-small)
            use_gpu: Use GPU if available (False for medical-grade determinism)
        """
        self.dimension = dimension
        self.use_gpu = use_gpu
        
        # Start with Flat index (exact search)
        # Will auto-upgrade to IVFFlat when size warrants it
        self.index = faiss.IndexFlatL2(dimension)
        
        # Track chunk IDs separately (FAISS only stores indices)
        self.chunk_ids: List[int] = []
    
    def add_vectors(self, embeddings: np.ndarray, chunk_ids: List[int]):
        """
        Add vectors to index.
        
        Args:
            embeddings: (N, dimension) array of embeddings
            chunk_ids: List of N chunk IDs matching SQLite
            
        Raises:
            ValueError: If dimension mismatch or ID count mismatch
        """
        # Validate dimension
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} != expected {self.dimension}"
            )
        
        # Validate count
        if len(chunk_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Chunk ID count {len(chunk_ids)} != embedding count {embeddings.shape[0]}"
            )
        
        # Ensure float32
        embeddings = embeddings.astype('float32')
        
        # Add to index
        self.index.add(embeddings)
        self.chunk_ids.extend(chunk_ids)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 40) -> Tuple[np.ndarray, List[int]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: (1, dimension) or (dimension,) array
            top_k: Number of results to return
            
        Returns:
            (distances, chunk_ids) tuple
            - distances: (top_k,) array of L2 distances
            - chunk_ids: List of top_k chunk IDs
        """
        # Validate and reshape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query_embedding.shape[1]} != expected {self.dimension}"
            )
        
        # Ensure float32
        query_embedding = query_embedding.astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Map indices to chunk IDs
        result_chunk_ids = [self.chunk_ids[i] for i in indices[0]]
        
        return distances[0], result_chunk_ids
    
    def count(self) -> int:
        """Return number of vectors in index."""
        return self.index.ntotal
    
    def upgrade_to_ivf(self, nlist: int = 100):
        """
        Upgrade to IVFFlat index for better performance on large datasets.
        
        Args:
            nlist: Number of inverted lists (clusters)
                  - 100 for 10k-100k vectors
                  - 1000 for 100k-1M vectors
        """
        if self.index.ntotal == 0:
            raise ValueError("Cannot upgrade empty index")
        
        # Create IVF index
        quantizer = faiss.IndexFlatL2(self.dimension)
        new_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        # Train on existing vectors
        vectors = faiss.vector_to_array(self.index.xb).reshape(-1, self.dimension)
        new_index.train(vectors)
        
        # Add vectors
        new_index.add(vectors)
        
        # Replace index
        self.index = new_index
    
    def reset(self):
        """Clear index and chunk IDs."""
        self.index.reset()
        self.chunk_ids = []
