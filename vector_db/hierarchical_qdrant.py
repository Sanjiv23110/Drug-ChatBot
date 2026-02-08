"""
Qdrant Manager for HIERARCHICAL PARENT-CHILD Storage
MANDATORY: Separate collections for parents and children
"""

from typing import List, Dict, Optional
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
    NamedVector,
    SparseVector
)
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalQdrantManager:
    """
    MANDATORY: Separate collections for parent and child chunks
    
    - CHILD COLLECTION: Used for search (sentence-level indexing)
    - PARENT COLLECTION: Used for display (paragraph-level SOURCE OF TRUTH)
    
    Search flow:
    1. Search child collection → get child IDs
    2. Extract parent_ids from children
    3. Retrieve parent chunks from parent collection
    4. Display PARENT text (VERBATIM)
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        child_collection: str = "spl_children",
        parent_collection: str = "spl_parents"
    ):
        """
        Initialize Qdrant client with HIERARCHICAL collections
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            child_collection: Collection for child chunks (SEARCH)
            parent_collection: Collection for parent chunks (DISPLAY)
        """
        self.client = QdrantClient(host=host, port=port)
        self.child_collection = child_collection
        self.parent_collection = parent_collection
        
        logger.info(f"Connected to Qdrant at {host}:{port}")
        logger.info(f"Child collection: {child_collection}")
        logger.info(f"Parent collection: {parent_collection}")
    
    def create_collections(
        self,
        dense_vector_size: int = 768,
        recreate: bool = False
    ):
        """
        Create BOTH parent and child collections
        
        CHILD COLLECTION:
        - Has dense + sparse vectors (for search)
        - Stores sentence text + parent_id reference
        
        PARENT COLLECTION:
        - NO vectors (pure key-value store)
        - Stores full paragraph text (SOURCE OF TRUTH)
        """
        if recreate:
            try:
                self.client.delete_collection(self.child_collection)
                self.client.delete_collection(self.parent_collection)
                logger.info("Deleted existing collections")
            except Exception as e:
                logger.warning(f"Could not delete collections: {e}")
        
        # Create CHILD collection (with vectors for search)
        self.client.create_collection(
            collection_name=self.child_collection,
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
        logger.info(f"Created child collection: {self.child_collection}")
        
        # Create PARENT collection (NO vectors - pure storage)
        # Parents are retrieved by ID, not searched
        self.client.create_collection(
            collection_name=self.parent_collection,
            vectors_config={
                "dummy": VectorParams(size=1, distance=Distance.COSINE)
            }  # Qdrant requires at least one vector config
        )
        logger.info(f"Created parent collection: {self.parent_collection}")
        
        # Create indices
        self._create_child_indices()
        self._create_parent_indices()
    
    def _create_child_indices(self):
        """Create indices on child collection for fast filtering"""
        indices = [
            ("drug_name", PayloadSchemaType.KEYWORD),
            ("rxcui", PayloadSchemaType.KEYWORD),
            ("loinc_code", PayloadSchemaType.KEYWORD),
            ("loinc_section", PayloadSchemaType.KEYWORD),
            ("parent_id", PayloadSchemaType.KEYWORD),  # CRITICAL for parent lookup
        ]
        
        for field_name, field_type in indices:
            try:
                self.client.create_payload_index(
                    collection_name=self.child_collection,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.info(f"Created child index on {field_name}")
            except Exception as e:
                logger.warning(f"Could not create child index on {field_name}: {e}")
    
    def _create_parent_indices(self):
        """Create indices on parent collection"""
        indices = [
            ("parent_id", PayloadSchemaType.KEYWORD),
            ("drug_name", PayloadSchemaType.KEYWORD),
            ("set_id", PayloadSchemaType.KEYWORD),
        ]
        
        for field_name, field_type in indices:
            try:
                self.client.create_payload_index(
                    collection_name=self.parent_collection,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.info(f"Created parent index on {field_name}")
            except Exception as e:
                logger.warning(f"Could not create parent index on {field_name}: {e}")
    
    def upsert_children(
        self,
        children: List[Dict],
        dense_embeddings: np.ndarray,
        sparse_embeddings: List[SparseVector]
    ):
        """
        Insert child chunks (for search indexing)
        
        Args:
            children: List of child chunk dictionaries
            dense_embeddings: Dense vectors for semantic search
            sparse_embeddings: Sparse vectors for keyword search
        """
        points = []
        
        for i, child in enumerate(children):
            point = PointStruct(
                id=abs(hash(child['child_id'])),  # Use abs() to ensure unsigned int for Qdrant v1.16.0
                vector={
                    "dense": dense_embeddings[i].tolist(),
                    "sparse": sparse_embeddings[i]
                },
                payload={
                    "child_id": child['child_id'],
                    "sentence_text": child['sentence_text'],
                    "parent_id": child['parent_id'],  # MANDATORY REFERENCE
                    "drug_name": child['drug_name'],
                    "rxcui": child['rxcui'],
                    "loinc_code": child['loinc_code'],
                    "loinc_section": child['loinc_section']
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.child_collection,
            points=points
        )
        
        logger.info(f"Upserted {len(points)} child chunks")
    
    def upsert_parents(self, parents: List[Dict]):
        """
        Insert parent chunks (SOURCE OF TRUTH for display)
        NO vectors - pure key-value storage
        
        Args:
            parents: List of parent chunk dictionaries
        """
        points = []
        
        for parent in parents:
            point = PointStruct(
                id=abs(hash(parent['parent_id'])),  # Use abs() to ensure unsigned int for Qdrant v1.16.0
                vector={"dummy": [0.0]},  # Dummy vector (not used)
                payload={
                    "parent_id": parent['parent_id'],
                    "raw_text": parent['raw_text'],  # IMMUTABLE SOURCE OF TRUTH
                    "drug_name": parent['drug_name'],
                    "rxcui": parent['rxcui'],
                    "set_id": parent['set_id'],
                    "root_id": parent['root_id'],
                    "version": parent['version'],
                    "effective_date": parent['effective_date'],
                    "loinc_code": parent['loinc_code'],
                    "loinc_section": parent['loinc_section'],
                    "is_table": parent['is_table'],
                    "ndc": parent['ndc']
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.parent_collection,
            points=points
        )
        
        logger.info(f"Upserted {len(points)} parent chunks")
    
    def search_children(
        self,
        query_dense: np.ndarray,
        query_sparse: SparseVector,
        filter_conditions: Optional[Dict] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Search CHILD collection (sentence-level)
        Returns child chunks with parent_id references
        """
        query_filter = None
        if filter_conditions:
            query_filter = self._build_filter(filter_conditions)
        
        results = self.client.search(
            collection_name=self.child_collection,
            query_vector=NamedVector(
                name="dense",
                vector=query_dense.tolist()
            ),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "child_id": result.payload['child_id'],
                "sentence_text": result.payload['sentence_text'],
                "parent_id": result.payload['parent_id'],  # CRITICAL
                "drug_name": result.payload['drug_name'],
                "loinc_section": result.payload['loinc_section'],
                "score": result.score
            })
        
        return formatted_results
    
    def get_parents_by_ids(self, parent_ids: List[str]) -> List[Dict]:
        """
        Retrieve PARENT chunks by IDs
        This is the SOURCE OF TRUTH text displayed to user
        
        Args:
            parent_ids: List of parent chunk IDs
            
        Returns:
            List of parent chunks with VERBATIM text
        """
        # Convert parent IDs to hashes
        id_hashes = [hash(pid) for pid in parent_ids]
        
        results = self.client.retrieve(
            collection_name=self.parent_collection,
            ids=id_hashes,
            with_payload=True,
            with_vectors=False
        )
        
        parents = []
        for result in results:
            parents.append({
                "parent_id": result.payload['parent_id'],
                "raw_text": result.payload['raw_text'],  # VERBATIM
                "drug_name": result.payload['drug_name'],
                "rxcui": result.payload['rxcui'],
                "set_id": result.payload['set_id'],
                "root_id": result.payload['root_id'],
                "version": result.payload['version'],
                "effective_date": result.payload['effective_date'],
                "loinc_code": result.payload['loinc_code'],
                "loinc_section": result.payload['loinc_section'],
                "is_table": result.payload['is_table']
            })
        
        return parents
    
    def _build_filter(self, conditions: Dict) -> Filter:
        """Build Qdrant filter from conditions"""
        must_conditions = []
        
        for key, value in conditions.items():
            if isinstance(value, list):
                condition = FieldCondition(
                    key=key,
                    match=MatchAny(any=value)
                )
            else:
                condition = FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            
            must_conditions.append(condition)
        
        return Filter(must=must_conditions)


# Example usage
if __name__ == "__main__":
    qm = HierarchicalQdrantManager(host="localhost", port=6333)
    
    # Create collections
    qm.create_collections(dense_vector_size=768, recreate=False)
    
    # Example: Search children, then retrieve parents
    import numpy as np
    
    query_embedding = np.random.rand(768)
    query_sparse = SparseVector(indices=[], values=[])
    
    # Step 1: Search children
    child_results = qm.search_children(
        query_dense=query_embedding,
        query_sparse=query_sparse,
        filter_conditions={"rxcui": "203644", "loinc_code": "34084-4"},
        limit=10
    )
    
    print(f"Found {len(child_results)} child chunks")
    
    # Step 2: Extract parent IDs
    parent_ids = list(set([child['parent_id'] for child in child_results]))
    
    # Step 3: Retrieve parents (SOURCE OF TRUTH)
    parents = qm.get_parents_by_ids(parent_ids)
    
    print(f"Retrieved {len(parents)} parent chunks (VERBATIM TEXT)")
    
    if parents:
        print(f"\nFirst parent text:")
        print(parents[0]['raw_text'][:200])
