"""
Entity Validation Layer - Pre-Retrieval Drug Name Validation
Prevents unfiltered retrieval by enforcing explicit drug entity presence in queries.
"""

from typing import Optional, Set, Dict
import logging

logger = logging.getLogger(__name__)


class EntityValidator:
    """
    Pre-retrieval entity validation to ensure queries contain explicit drug references.
    
    Design:
    - Maintains cached set of indexed drug names from vector DB metadata
    - Performs case-insensitive substring matching
    - Requires EXACT entity presence (no LLM inference, no approximation)
    - O(1) lookup performance via set-based matching
    """
    
    def __init__(self, vector_db_manager):
        """
        Args:
            vector_db_manager: QdrantManager instance to extract drug names from metadata
        """
        self.vector_db = vector_db_manager
        # Mapping: lowercase_name -> original_cased_name (e.g., "dantrium" -> "Dantrium")
        self.drug_names: Dict[str, str] = {}
        self._load_drug_names()
    
    def _load_drug_names(self):
        """
        Load unique drug names from vector DB metadata.
        Called once at startup for O(1) lookup performance.
        Stores mapping to handle case-sensitive retrieval requirements.
        """
        try:
            # Determine collection name (handle both QdrantManager and HierarchicalQdrantManager)
            if hasattr(self.vector_db, 'child_collection'):
                collection_name = self.vector_db.child_collection
            else:
                collection_name = self.vector_db.collection_name
                
            # Scroll through all points in collection using pagination
            # Scalability: Handles 20k+ drugs by iterating in batches
            next_offset = None
            total_loaded = 0
            
            while True:
                points, next_offset = self.vector_db.client.scroll(
                    collection_name=collection_name,
                    limit=1000,  # Batch size
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not points:
                    break
                
                for point in points:
                    drug_name = point.payload.get('drug_name')
                    if drug_name:
                        # Store mapping: lower -> original
                        self.drug_names[drug_name.lower()] = drug_name
                
                total_loaded += len(points)
                
                # If no next page, stop
                if next_offset is None:
                    break
            
            logger.info(f"EntityValidator: Loaded {len(self.drug_names)} unique drug names from {total_loaded} chunks")
            
        except Exception as e:
            logger.error(f"Failed to load drug names from vector DB: {e}")
            self.drug_names = {}
    
    def validate(self, query: str) -> Dict[str, any]:
        """
        Validate that query contains an explicit drug entity.
        
        Args:
            query: User query string
            
        Returns:
            {
                "valid": bool,
                "drug_name": Optional[str],  # Matched drug name (ORIGINAL CASING) if valid
                "reason": Optional[str]       # Refusal reason if invalid
            }
        """
        if not self.drug_names:
            # Fail-safe: if drug names not loaded, refuse query
            return {
                "valid": False,
                "drug_name": None,
                "reason": "System not ready. Unable to validate drug entity."
            }
        
        query_lower = query.lower()
        
        # Match against known drug names (substring match, case-insensitive)
        for drug_lower, drug_original in self.drug_names.items():
            if drug_lower in query_lower:
                return {
                    "valid": True,
                    "drug_name": drug_original, # Return Correct Casing for Retrieval
                    "reason": None
                }
        
        # No explicit drug found
        return {
            "valid": False,
            "drug_name": None,
            "reason": "No drug specified. Please provide the drug name."
        }
    
    def refresh_drug_names(self):
        """
        Refresh drug name cache from vector DB.
        Call after ingestion to update cached drug list.
        """
        logger.info("Refreshing drug name cache...")
        self._load_drug_names()
