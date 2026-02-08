"""
Qdrant Vector Database Manager
Handles hybrid search, metadata filtering, and indexing for regulatory QA system
"""

from typing import List, Dict, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, 
    PayloadSchemaType, 
    SparseVectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    SearchRequest,
    SparseVector
)
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantManager:
    """
    Manage Qdrant vector database for SPL chunks
    Supports hybrid search (dense + sparse) with metadata filtering
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "spl_chunks"
    ):
        """
        Initialize Qdrant client
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of collection to use
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        
        logger.info(f"Connected to Qdrant at {host}:{port}")
    
    def create_collection(
        self,
        dense_vector_size: int = 768,
        recreate: bool = False
    ):
        """
        Create Qdrant collection with hybrid search support
        
        Args:
            dense_vector_size: Dimension of dense embeddings (e.g., 768 for S-PubMedBert)
            recreate: If True, delete existing collection first
        """
        if recreate:
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception as e:
                logger.warning(f"Could not delete collection: {e}")
        
        # Create collection with dense + sparse vectors
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=dense_vector_size,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            }
        )
        
        logger.info(f"Created collection: {self.collection_name}")
        
        # Create payload indices for fast filtering
        self._create_indices()
    
    def _create_indices(self):
        """Create indices on metadata fields for fast filtering"""
        
        indices = [
            ("metadata.drug_name", PayloadSchemaType.KEYWORD),
            ("metadata.rxcui", PayloadSchemaType.KEYWORD),
            ("metadata.loinc_code", PayloadSchemaType.KEYWORD),
            ("metadata.loinc_section", PayloadSchemaType.KEYWORD),
            ("metadata.set_id", PayloadSchemaType.KEYWORD),
            ("metadata.version", PayloadSchemaType.KEYWORD),
            ("metadata.is_table", PayloadSchemaType.BOOL),
        ]
        
        for field_name, field_type in indices:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.info(f"Created index on {field_name}")
            except Exception as e:
                logger.warning(f"Could not create index on {field_name}: {e}")
    
    def upsert_chunks(
        self,
        chunks: List[Dict],
        dense_embeddings: np.ndarray,
        sparse_embeddings: List[SparseVector]
    ):
        """
        Insert or update chunks in Qdrant
        
        Args:
            chunks: List of chunk dictionaries (from DocumentChunk.to_dict())
            dense_embeddings: Dense vectors (shape: [n_chunks, embedding_dim])
            sparse_embeddings: Sparse vectors (list of SparseVector objects)
        """
        points = []
        
        for i, chunk in enumerate(chunks):
            point = PointStruct(
                id=hash(chunk['chunk_id']),  # Use hash of chunk_id as numeric ID
                vector={
                    "dense": dense_embeddings[i].tolist(),
                    "sparse": sparse_embeddings[i]
                },
                payload={
                    "chunk_id": chunk['chunk_id'],
                    "semantic_text": chunk['semantic_text'],
                    "raw_text": chunk['raw_text'],
                    "metadata": chunk['metadata']
                }
            )
            points.append(point)
        
        # Batch upsert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Upserted {len(points)} chunks")
    
    def hybrid_search(
        self,
        query_dense: np.ndarray,
        query_sparse: SparseVector,
        filter_conditions: Optional[Dict] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Perform hybrid search (dense + sparse)
        
        Args:
            query_dense: Dense query embedding
            query_sparse: Sparse query embedding
            filter_conditions: Metadata filters (applied BEFORE search)
            limit: Max results to return
            
        Returns:
            List of search results with scores
        """
        # Build filter
        query_filter = None
        if filter_conditions:
            query_filter = self._build_filter(filter_conditions)
        
        # Search with dense + sparse vectors using query_points (v1.16+ API)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_dense.tolist(),
            using="dense",  # CRITICAL: Specify vector name for named vector collections
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False  # Don't return vectors (saves bandwidth)
        ).points
        
        # Format results - handle both old (spl_chunks) and new (spl_children) schema
        formatted_results = []
        for result in results:
            payload = result.payload
            # New hierarchical schema (spl_children collection)
            if 'child_id' in payload:
                formatted_results.append({
                    "chunk_id": payload.get('child_id'),
                    "semantic_text": payload.get('sentence_text', ''),
                    "raw_text": payload.get('sentence_text', ''),  # Children don't have raw_text
                    "metadata": {
                        "parent_id": payload.get('parent_id'),
                        "drug_name": payload.get('drug_name'),
                        "rxcui": payload.get('rxcui'),
                        "loinc_code": payload.get('loinc_code'),
                        "loinc_section": payload.get('loinc_section')
                    },
                    "score": result.score
                })
            # Old flat schema (spl_chunks collection)
            else:
                formatted_results.append({
                    "chunk_id": payload.get('chunk_id'),
                    "semantic_text": payload.get('semantic_text', ''),
                    "raw_text": payload.get('raw_text', ''),
                    "metadata": payload.get('metadata', {}),
                    "score": result.score
                })
        
        return formatted_results
    
    def _build_filter(self, conditions: Dict) -> Filter:
        """
        Build Qdrant filter from conditions
        
        For spl_children collection, fields are at top level (no metadata prefix)
        For old spl_chunks collection, fields were nested under metadata.
        """
        must_conditions = []
        
        for key, value in conditions.items():
            # For hierarchical schema (spl_children), fields are at top level
            # No metadata prefix needed
            field_name = key
            
            if isinstance(value, list):
                # Match any of multiple values
                condition = FieldCondition(
                    key=field_name,
                    match=MatchAny(any=value)
                )
            else:
                # Match single value
                condition = FieldCondition(
                    key=field_name,
                    match=MatchValue(value=value)
                )
            
            must_conditions.append(condition)
        
        return Filter(must=must_conditions)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Retrieve specific chunk by ID"""
        chunk_hash = hash(chunk_id)
        
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[chunk_hash],
            with_payload=True,
            with_vectors=False
        )
        
        if result:
            return {
                "chunk_id": result[0].payload['chunk_id'],
                "semantic_text": result[0].payload['semantic_text'],
                "raw_text": result[0].payload['raw_text'],
                "metadata": result[0].payload['metadata']
            }
        
        return None
    
    def count_chunks(self, filter_conditions: Optional[Dict] = None) -> int:
        """Count chunks matching filter"""
        query_filter = None
        if filter_conditions:
            query_filter = self._build_filter(filter_conditions)
        
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=query_filter
        )
        
        return result.count
    
    def get_collection_info(self) -> Dict:
        """Get collection statistics"""
        info = self.client.get_collection(self.collection_name)
        
        return {
            "name": info.config.params.vectors['dense'].size,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status
        }


class ReciprocalRankFusion:
    """
    Combine results from dense and sparse search using RRF
    """
    
    @staticmethod
    def fuse(
        dense_results: List[Dict],
        sparse_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        Apply Reciprocal Rank Fusion
        
        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            k: Constant (typically 60)
            
        Returns:
            Fused results sorted by RRF score
        """
        scores = {}
        chunk_map = {}
        
        # Score dense results
        for rank, result in enumerate(dense_results):
            chunk_id = result['chunk_id']
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            chunk_map[chunk_id] = result
        
        # Score sparse results
        for rank, result in enumerate(sparse_results):
            chunk_id = result['chunk_id']
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = result
        
        # Sort by fused score
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return chunks with RRF scores
        fused_results = []
        for chunk_id, rrf_score in sorted_chunks:
            chunk = chunk_map[chunk_id]
            chunk['rrf_score'] = rrf_score
            fused_results.append(chunk)
        
        return fused_results


# Example usage
if __name__ == "__main__":
    # Initialize Qdrant
    qm = QdrantManager(host="localhost", port=6333)
    
    # Create collection (only once)
    qm.create_collection(dense_vector_size=768, recreate=False)
    
    # Get collection info
    info = qm.get_collection_info()
    print(f"Collection info: {info}")
    
    # Example: Count chunks for a specific drug
    count = qm.count_chunks(filter_conditions={"rxcui": "203644"})
    print(f"Chunks for Lisinopril: {count}")
