"""
SQL-First Retrieval Engine for Medical Drug Monographs.

Routing Strategy (STRICT ORDER):
1. Path A: SQL Exact Match (PREFERRED) - if drug AND section known
2. Path A+: SQL Fuzzy Match - if exact match fails, try fuzzy
3. Path B: Image Lookup - if needs_image=TRUE
4. Path C: Scoped Vector Fallback (LAST RESORT) - only if SQL returns nothing

Philosophy: Embeddings are OPTIONAL and strictly a fallback mechanism.
Vector search MUST be scoped to drug_name - global vector search is FORBIDDEN.

DESIGN: NO HARDCODED SECTIONS - uses dynamic section matching with pg_trgm
"""
import os
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy import select, or_
from sqlalchemy import text

from app.db.models import MonographSection, SectionMapping, find_similar_sections
from app.db.session import get_session
from app.retrieval.intent_classifier import QueryIntent, IntentClassifier
from app.ingestion.embedder import AzureEmbedder  # Reuse existing embedder

logger = logging.getLogger(__name__)


class RetrievalPath(str, Enum):
    """Which retrieval strategy was used."""
    SQL_EXACT = "SQL_EXACT"           # Path A: SQL exact match
    SQL_FUZZY = "SQL_FUZZY"           # Path A+: SQL fuzzy match
    IMAGE_LOOKUP = "IMAGE_LOOKUP"     # Path B: Image retrieval
    VECTOR_SCOPED = "VECTOR_SCOPED"   # Path C: Scoped vector search
    NO_RESULT = "NO_RESULT"           # Nothing found


@dataclass
class RetrievalResult:
    """Result from the retrieval engine."""
    # Content
    sections: List[Dict[str, Any]] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    
    # Metadata
    path_used: RetrievalPath = RetrievalPath.NO_RESULT
    drug_name: Optional[str] = None
    section_name: Optional[str] = None  # DYNAMIC section name
    
    # Query info
    original_query: str = ""
    intent: Optional[QueryIntent] = None
    
    # Statistics
    total_results: int = 0
    sql_executed: Optional[str] = None


class RetrievalEngine:
    """
    Production retrieval engine with SQL-first strategy.
    
    Routing:
    - Path A: SQL exact match when drug+section are known
    - Path A+: SQL fuzzy match using pg_trgm
    - Path B: Image lookup when structure is requested
    - Path C: Vector fallback ONLY when SQL returns nothing
    
    Vector search is ALWAYS scoped to drug_name.
    Global vector search is FORBIDDEN.
    
    DESIGN: NO HARDCODED SECTIONS - uses dynamic section lookup
    """
    
    def __init__(
        self,
        enable_vector_fallback: bool = True,
        vector_top_k: int = 10,
        use_fuzzy_matching: bool = True
    ):
        """
        Initialize the retrieval engine.
        
        Args:
            enable_vector_fallback: Allow vector search as fallback
            vector_top_k: Number of results for vector search
            use_fuzzy_matching: Enable pg_trgm fuzzy matching
        """
        self.enable_vector_fallback = enable_vector_fallback
        self.vector_top_k = vector_top_k
        self.use_fuzzy_matching = use_fuzzy_matching
        
        self.intent_classifier = IntentClassifier()
        
        # Initialize embedder for vector fallback
        if enable_vector_fallback:
            try:
                self.embedder = AzureEmbedder()
            except Exception as e:
                logger.warning(f"Embedder init failed, vector fallback disabled: {e}")
                self.enable_vector_fallback = False
        
        logger.info(
            f"RetrievalEngine initialized (vector_fallback: {enable_vector_fallback})"
        )
    
    async def retrieve(self, query: str) -> RetrievalResult:
        """
        Main retrieval method.
        
        Args:
            query: User's natural language query
            
        Returns:
            RetrievalResult with sections and/or images
        """
        # Step 1: Classify intent
        intent = self.intent_classifier.classify(query)
        
        result = RetrievalResult(
            original_query=query,
            intent=intent,
            drug_name=intent.target_drug,
            section_name=intent.target_section  # DYNAMIC - string, not enum
        )
        
        # Step 2: Route to appropriate path
        
        # Path B: Image lookup (check first - specific request)
        if intent.needs_image and intent.target_drug:
            logger.info(f"Routing to Path B: Image Lookup for {intent.target_drug}")
            return await self._path_b_image_lookup(intent, result)
        
        # Path A: SQL Exact Match (preferred)
        if intent.target_drug and intent.target_section:
            logger.info(
                f"Routing to Path A: SQL Match "
                f"({intent.target_drug}, {intent.target_section})"
            )
            result = await self._path_a_sql_match(intent, result)
            
            # If SQL found results, return them
            if result.sections:
                return result
            
            # Otherwise fall through to Path C
        
        # Path A (partial): Drug known, section unknown
        if intent.target_drug and not intent.target_section:
            logger.info(f"Routing to Path A (partial): All sections for {intent.target_drug}")
            result = await self._path_a_sql_drug_only(intent, result)
            
            if result.sections:
                return result
        
        # Path C: Vector Fallback (LAST RESORT)
        if self.enable_vector_fallback and intent.target_drug:
            logger.info(f"Routing to Path C: Vector Fallback for {intent.target_drug}")
            return await self._path_c_vector_scoped(intent, result)
        
        # No drug identified - cannot proceed
        if not intent.target_drug:
            logger.warning("No drug identified in query, cannot retrieve")
            result.path_used = RetrievalPath.NO_RESULT
            return result
        
        return result
    
    async def _path_a_sql_match(
        self,
        intent: QueryIntent,
        result: RetrievalResult
    ) -> RetrievalResult:
        """
        Path A: SQL match when drug AND section are known.
        
        Strategy:
        1. Try exact section_name match
        2. If no results, try fuzzy match using pg_ trgm
        """
        async with get_session() as session:
            # First try: Exact match
            # IMPORTANT: Check brand_name, generic_name, AND drug_name for flexible matching
            stmt = (
                select(MonographSection)
                .where(MonographSection.drug_name == intent.target_drug)
                .where(MonographSection.section_name == intent.target_section)
                .order_by(MonographSection.page_start)
            )
            
            result.sql_executed = str(stmt)
            
            db_result = await session.execute(stmt)
            sections = db_result.scalars().all()
            
            if sections:
                result.sections = [self._section_to_dict(s) for s in sections]
                result.path_used = RetrievalPath.SQL_EXACT
                result.total_results = len(sections)
                
                # Include images if section has them
                for section in sections:
                    if section.image_paths:
                        result.image_paths.extend(section.image_paths)
                
                logger.info(f"Path A (exact) returned {len(sections)} sections")
                return result
            
            # Second try: Fuzzy match using pg_trgm
            if self.use_fuzzy_matching:
                fuzzy_stmt = text("""
                    SELECT * FROM monograph_sections
                    WHERE drug_name = :drug_name
                    AND similarity(section_name, :section_query) > 0.3
                    ORDER BY similarity(section_name, :section_query) DESC
                    LIMIT 10
                """)
                
                fuzzy_result = await session.execute(
                    fuzzy_stmt,
                    {
                        "drug_name": intent.target_drug,
                        "section_query": intent.target_section
                    }
                )
                
                rows = fuzzy_result.fetchall()
                
                if rows:
                    result.sections = [
                        {
                            "id": row.id,
                            "drug_name": row.drug_name,
                            "section_name": row.section_name,
                            "original_header": row.original_header,
                            "content_text": row.content_text,
                            "image_paths": row.image_paths or [],
                            "has_chemical_structure": row.has_chemical_structure
                        }
                        for row in rows
                    ]
                    result.path_used = RetrievalPath.SQL_FUZZY
                    result.total_results = len(rows)
                    
                    logger.info(f"Path A (fuzzy) returned {len(rows)} sections")
        
        return result
    
    async def _path_a_sql_drug_only(
        self,
        intent: QueryIntent,
        result: RetrievalResult
    ) -> RetrievalResult:
        """
        Path A (partial): Get all sections for a drug when section is unknown.
        """
        async with get_session() as session:
            stmt = (
                select(MonographSection)
                .where(MonographSection.drug_name == intent.target_drug)
                .order_by(MonographSection.section_name, MonographSection.page_start)
            )
            
            result.sql_executed = str(stmt)
            
            db_result = await session.execute(stmt)
            sections = db_result.scalars().all()
            
            if sections:
                result.sections = [self._section_to_dict(s) for s in sections]
                result.path_used = RetrievalPath.SQL_EXACT
                result.total_results = len(sections)
            
            logger.info(f"Path A (drug only) returned {len(sections)} sections")
        
        return result
    
    async def _path_b_image_lookup(
        self,
        intent: QueryIntent,
        result: RetrievalResult
    ) -> RetrievalResult:
        """
        Path B: Image lookup for chemical structure requests.
        
        Uses fuzzy matching on section_name to find structure-related sections.
        """
        async with get_session() as session:
            # Find sections with chemical structures using fuzzy match
            stmt = text("""
                SELECT * FROM monograph_sections
                WHERE drug_name = :drug_name
                AND has_chemical_structure = TRUE
                ORDER BY 
                    CASE WHEN section_name ILIKE '%structure%' THEN 1
                         WHEN section_name ILIKE '%chemical%' THEN 2
                         WHEN section_name ILIKE '%description%' THEN 3
                         ELSE 4
                    END
            """)
            
            result.sql_executed = str(stmt)
            
            db_result = await session.execute(
                stmt,
                {"drug_name": intent.target_drug}
            )
            rows = db_result.fetchall()
            
            if rows:
                result.sections = [
                    {
                        "id": row.id,
                        "drug_name": row.drug_name,
                        "section_name": row.section_name,
                        "original_header": row.original_header,
                        "content_text": row.content_text,
                        "image_paths": row.image_paths or [],
                        "has_chemical_structure": row.has_chemical_structure
                    }
                    for row in rows
                ]
                
                for row in rows:
                    if row.image_paths:
                        result.image_paths.extend(row.image_paths)
                
                result.path_used = RetrievalPath.IMAGE_LOOKUP
                result.total_results = len(rows)
            
            logger.info(f"Path B returned {len(result.image_paths)} images")
        
        return result
    
    async def _path_c_vector_scoped(
        self,
        intent: QueryIntent,
        result: RetrievalResult
    ) -> RetrievalResult:
        """
        Path C: Vector fallback - ONLY when SQL returns nothing.
        
        CRITICAL: Vector search MUST be scoped to drug_name.
        Global vector search is FORBIDDEN.
        """
        if not intent.target_drug:
            logger.error("Path C called without drug_name - this is FORBIDDEN")
            result.path_used = RetrievalPath.NO_RESULT
            return result
        
        try:
            # Get query embedding
            query_embedding = self.embedder.embed_batch([intent.original_query])[0]
            
            # Search in sections table using pgvector
            async with get_session() as session:
                # Vector search SCOPED to drug (check all name fields)
                # Using pgvector's <-> operator for L2 distance  
                stmt = text("""
                    SELECT 
                        id, drug_name, section_name, original_header,
                        content_text, image_paths, has_chemical_structure,
                        embedding <-> :query_vector AS distance
                    FROM monograph_sections
                    WHERE drug_name = :drug_name
                    AND embedding IS NOT NULL
                    ORDER BY embedding <-> :query_vector
                    LIMIT :top_k
                """)
                
                result.sql_executed = str(stmt)
                
                db_result = await session.execute(
                    stmt,
                    {
                        "query_vector": str(query_embedding.tolist()),
                        "drug_name": intent.target_drug,
                        "top_k": self.vector_top_k
                    }
                )
                
                rows = db_result.fetchall()
                
                if rows:
                    result.sections = [
                        {
                            "id": row.id,
                            "drug_name": row.drug_name,
                            "section_name": row.section_name,
                            "original_header": row.original_header,
                            "content_text": row.content_text,
                            "image_paths": row.image_paths or [],
                            "has_chemical_structure": row.has_chemical_structure,
                            "distance": row.distance
                        }
                        for row in rows
                    ]
                    result.path_used = RetrievalPath.VECTOR_SCOPED
                    result.total_results = len(rows)
                
                logger.info(f"Path C returned {len(result.sections)} sections")
                
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            result.path_used = RetrievalPath.NO_RESULT
        
        return result
    
    def _section_to_dict(self, section: MonographSection) -> Dict[str, Any]:
        """Convert MonographSection to dict."""
        return {
            "id": section.id,
            "drug_name": section.drug_name,
            "brand_name": section.brand_name,
            "generic_name": section.generic_name,
            "section_name": section.section_name,  # DYNAMIC
            "original_header": section.original_header,
            "content_text": section.content_text,
            "content_markdown": section.content_markdown,
            "image_paths": section.image_paths or [],
            "has_chemical_structure": section.has_chemical_structure,
            "page_start": section.page_start,
            "page_end": section.page_end,
            "char_count": section.char_count
        }
    
    async def get_available_drugs(self) -> List[str]:
        """Get list of all drugs in database."""
        async with get_session() as session:
            stmt = select(MonographSection.drug_name).distinct()
            result = await session.execute(stmt)
            return [row[0] for row in result.fetchall()]
    
    async def get_drug_sections(self, drug_name: str) -> List[str]:
        """Get available sections for a drug (DYNAMIC)."""
        async with get_session() as session:
            stmt = (
                select(MonographSection.section_name)
                .where(MonographSection.drug_name == drug_name)
                .distinct()
            )
            result = await session.execute(stmt)
            return [row[0] for row in result.fetchall()]
    
    async def get_all_section_types(self) -> List[Dict[str, Any]]:
        """
        Get all known section types with usage stats.
        
        Returns:
            List of section mappings with counts
        """
        async with get_session() as session:
            stmt = (
                select(SectionMapping)
                .order_by(SectionMapping.usage_count.desc())
            )
            result = await session.execute(stmt)
            mappings = result.scalars().all()
            
            return [
                {
                    "normalized_name": m.normalized_name,
                    "display_name": m.display_name,
                    "usage_count": m.usage_count,
                    "is_common": m.is_common
                }
                for m in mappings
            ]
    
    async def search_sections(
        self,
        drug_name: str,
        section_query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for sections using fuzzy matching.
        
        Args:
            drug_name: Drug to search within
            section_query: Section name to search for
            limit: Max results
            
        Returns:
            List of matching sections
        """
        async with get_session() as session:
            stmt = text("""
                SELECT *, similarity(section_name, :query) as sim
                FROM monograph_sections
                WHERE drug_name = :drug_name
                AND similarity(section_name, :query) > 0.2
                ORDER BY sim DESC
                LIMIT :limit
            """)
            
            result = await session.execute(
                stmt,
                {
                    "drug_name": drug_name,
                    "query": section_query,
                    "limit": limit
                }
            )
            
            rows = result.fetchall()
            return [
                {
                    "section_name": row.section_name,
                    "original_header": row.original_header,
                    "similarity": row.sim,
                    "content_preview": row.content_text[:200] + "..."
                }
                for row in rows
            ]


# Convenience function
async def retrieve(query: str) -> RetrievalResult:
    """
    Convenience function to retrieve content.
    
    Args:
        query: User query
        
    Returns:
        RetrievalResult with sections and/or images
    """
    engine = RetrievalEngine()
    return await engine.retrieve(query)
