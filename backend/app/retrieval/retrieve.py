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
from app.db.fact_span_model import FactSpan
from app.retrieval.attribute_aggregator import AttributeAggregator, CandidateSentence
from app.retrieval.query_planner import QueryPlanner, RetrievalPlan
from sqlalchemy import func, desc

logger = logging.getLogger(__name__)


class RetrievalPath(str, Enum):
    """Which retrieval strategy was used."""
    SQL_EXACT = "SQL_EXACT"           # Path A: SQL exact match
    SQL_FUZZY = "SQL_FUZZY"           # Path A+: SQL fuzzy match
    SECTION_LOOKUP = "SECTION_LOOKUP"   # Path A: SQL exact/fuzzy
    ATTRIBUTE_LOOKUP = "ATTRIBUTE_LOOKUP" # Path A++: Attribute scoped
    BM25_FACTSPAN = "BM25_FACTSPAN"       # Path D: BM25 FactSpan
    GLOBAL_FACTSPAN_SCAN = "GLOBAL_FACTSPAN_SCAN" # Path E: Global Scan
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
    attribute_name: Optional[str] = None # NEW
    
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
    - Path A++: Attribute lookup (NEW) - maps attribute to section(s)
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
        self.planner = QueryPlanner()
        
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

    def _is_junk_section(self, content: str) -> bool:
        """
        Detect junk/TOC sections.
        
        Criteria:
        - Length < 50 chars
        - Contains TOC indicators ("......")
        """
        import re
        content = content.strip()
        
        # 1. Too short to be useful section
        if len(content) < 50:
            # But allow if it's purely a table reference or "See X" 
            # actually, strictly filtering short content is safer for "Dosage" which is usually long.
            # Use regex to detect navigation junk specifically
            if re.search(r'\.{3,}\s*\d+', content): # "...... 6"
                return True
            if re.match(r'^[\d\s\.]+$', content): # Just numbers and dots
                return True
            
            # If it's just "Dosage......6", it's junk.
            # If it's "Capsules 150mg", it's distinct content.
            # Let's rely mainly on the TOC pattern for now to be safe.
            return False
            
        return False
    
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
            section_name=intent.target_section,
            attribute_name=intent.target_attribute
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
            
            # Otherwise fall through to Path A++ or C
            
        # Path A++: Attribute Lookup (NEW)
        # Only if drug is known and specific attribute detected
        if intent.target_drug and intent.target_attribute:
            logger.info(
                f"Routing to Path A++: Attribute Lookup "
                f"({intent.target_drug}, {intent.target_attribute})"
            )
            result = await self._path_a_attribute_lookup(intent, result)
            
            if result.sections:
                return result

        # Path A (partial): Drug known, section unknown
        if intent.target_drug and not intent.target_section and not intent.target_attribute:
            logger.info(f"Routing to Path A (partial): All sections for {intent.target_drug}")
            result = await self._path_a_sql_drug_only(intent, result)
            
            if result.sections:
                return result

        # Path A++: Attribute Lookup (if attribute identified)
        if intent.target_attribute:
            result = await self._path_a_attribute_lookup(intent, result)
            if result.sections:
                return result

        # NEW: Recall Amplification Paths (BM25 / Global Scan)
        # Only activated if exact/fuzzy/attribute lookups failed
        if not result.sections:
            logger.info("Primary paths empty. Invoking QueryPlanner for recall amplification...")
            try:
                plan = await self.planner.plan(query)

                # ISSUE 4 FIX: Deterministic Override
                # If planner suggests core sections, force SQL lookup (Path A) instead of BM25.
                # This fixes "dosage forms" being processed as a keyword search instead of section lookup.
                core_overrides = {"dosage", "administration", "description", "indications", "composition", "contraindications", "warnings"}
                candidates = [s.lower() for s in (plan.candidate_sections or [])]
                
                # Check intersection
                force_sql = False
                for cand in candidates:
                    if any(core in cand for core in core_overrides):
                        force_sql = True
                        break
                
                if force_sql:
                    logger.info(f"Deterministic Override: Enforcing SQL lookup for planner sections: {candidates}")
                    
                    found_any = False
                    for sec in candidates:
                        # Reuse Path A logic for each suggested section
                        temp_intent = QueryIntent(target_drug=plan.drug, target_section=sec)
                        # We pass a fresh result object to avoid pollution, then merge
                        sub_result = await self._path_a_sql_match(temp_intent, RetrievalResult())
                        
                        if sub_result.sections:
                            for s in sub_result.sections:
                                # Simple dedup by ID
                                if not any(existing['id'] == s['id'] for existing in result.sections):
                                    result.sections.append(s)
                            found_any = True
                    
                    if found_any:
                        result.path_used = RetrievalPath.SECTION_LOOKUP
                        return result

                # Path D: BM25 FactSpan
                result = await self._path_d_bm25_factspan(plan, result)
                if result.sections:
                    return result
                    
                # Path E: Global FactSpan Scan fallback
                result = await self._path_e_global_scan(plan, result)
                if result.sections:
                    return result
                    
            except Exception as e:
                logger.error(f"Planning/Recall paths failed: {e}")

        # Path C: Vector Fallback (Last Resort)
        if self.enable_vector_fallback and not result.sections and intent.target_drug:
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
            # Check ALL name fields: drug_name, brand_name, generic_name
            stmt = (
                select(MonographSection)
                .where(
                    or_(
                        MonographSection.drug_name == intent.target_drug,
                        MonographSection.brand_name == intent.target_drug,
                        MonographSection.generic_name == intent.target_drug
                    )
                )
                .where(MonographSection.section_name == intent.target_section)
                .order_by(MonographSection.page_start)
            )
            
            result.sql_executed = str(stmt)
            
            db_result = await session.execute(stmt)
            sections = db_result.scalars().all()
            
            if sections:
                # Filter out junk/TOC sections
                valid_sections = [
                    s for s in sections 
                    if not self._is_junk_section(s.content_text or "")
                ]
                
                if valid_sections:
                    result.sections = [self._section_to_dict(s) for s in valid_sections]
                    result.path_used = RetrievalPath.SQL_EXACT
                    result.total_results = len(valid_sections)
                    
                    # Include images if section has them
                    for section in valid_sections:
                        if section.image_paths:
                            result.image_paths.extend(section.image_paths)
                    
                    logger.info(f"Path A (exact) returned {len(valid_sections)} sections (original: {len(sections)})")
                    return result
            
            # Second try: Enhanced fuzzy match using pg_trgm + keyword fallback
            if self.use_fuzzy_matching:
                # Try fuzzy with LOWER threshold for better recall
                fuzzy_stmt = text("""
                    SELECT *, similarity(section_name, :section_query) as sim_score
                    FROM monograph_sections
                    WHERE (
                        drug_name = :drug_name 
                        OR brand_name = :drug_name 
                        OR generic_name = :drug_name
                    )
                    AND (
                        -- Fuzzy similarity (lowered threshold for 19k PDFs)
                        similarity(section_name, :section_query) > 0.2
                        OR 
                        -- Keyword fallback for common patterns
                        (
                            -- "indications" matches "what is X used for"
                            (:section_query = 'indications' AND (
                                section_name ILIKE '%used%for%' OR 
                                section_name ILIKE '%indication%' OR
                                section_name ILIKE '%therapeutic%'
                            ))
                            OR
                            -- "contraindications" matches "when not to use"
                            (:section_query = 'contraindications' AND (
                                section_name ILIKE '%contraindication%' OR
                                section_name ILIKE '%not%use%' OR
                                section_name ILIKE '%should not%'
                            ))
                            OR
                            -- "dosage" matches "how to take"
                            (:section_query = 'dosage' AND (
                                section_name ILIKE '%dosage%' OR
                                section_name ILIKE '%how%take%' OR
                                section_name ILIKE '%administration%'
                            ))
                            OR
                            -- "side_effects" matches various forms
                            (:section_query = 'side_effects' AND (
                                section_name ILIKE '%side%effect%' OR
                                section_name ILIKE '%adverse%' OR
                                section_name ILIKE '%unwanted%'
                            ))
                        )
                    )
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
                    
                    logger.info(f"Path A (enhanced fuzzy) returned {len(rows)} sections")
        
        
        return result

    async def _path_a_attribute_lookup(
        self,
        intent: QueryIntent,
        result: RetrievalResult
    ) -> RetrievalResult:
        """
        Path A++: Attribute-Scoped Section Retrieval.
        
        Behavior:
        1. Map attribute to allowed sections (from IntentClassifier.ATTRIBUTE_MAP)
        2. Retrieve ALL matching sections for the drug using SQL (ILIKE)
        3. No vector search, no inferencing.
        """
        if not intent.target_attribute:
            return result

        # Get allowed sections for this attribute
        attr_data = IntentClassifier.ATTRIBUTE_MAP.get(intent.target_attribute)
        if not attr_data:
            logger.warning(f"Attribute {intent.target_attribute} not found in map")
            return result
        
        allowed_sections = attr_data["sections"]
        logger.info(f"Path A++ looking for attribute '{intent.target_attribute}' in sections: {allowed_sections}")
        
        async with get_session() as session:
            # NEW: Attribute Evidence Aggregation (FactSpan + LLM Filter)
            
            # RETRIEVAL INVARIANT (PRIORITY 1)
            # IF authoritative FactSpans exist, return them ALL verbatim.
            try:
                invariant_stmt = select(FactSpan).where(
                    FactSpan.drug_name == intent.target_drug,
                    FactSpan.assertion_type.in_(['FACT', 'CONDITIONAL']),
                    or_(*[FactSpan.section_enum.ilike(f"%{s}%") for s in allowed_sections])
                )
                inv_result = await session.execute(invariant_stmt)
                inv_spans = inv_result.scalars().all()

                if inv_spans:
                    logger.info(f"Retrieval Invariant Triggered: Found {len(inv_spans)} authoritative spans for {intent.target_attribute}.")
                    # Return ALL spans verbatim. never "not found".
                    formatted_content = "\n".join([f"- {s.sentence_text}" for s in inv_spans])
                    
                    synthetic_section = {
                        "section_name": f"Authoritative Facts: {intent.target_attribute}",
                        "content_text": formatted_content,
                        "drug_name": intent.target_drug,
                        "attribute_provenance": True,
                        "is_aggregated": True
                    }
                    
                    result.sections = [synthetic_section]
                    result.path_used = RetrievalPath.ATTRIBUTE_LOOKUP 
                    result.attribute_name = intent.target_attribute
                    return result
            except Exception as e:
                logger.error(f"Invariant check failed: {e}")

            try:
                # 1. Deterministically retrieve candidate FactSpans
                # We want sentences that contain attribute keywords AND are in allowed sections
                
                # Derive keywords from attribute name (simple heuristic)
                keywords = intent.target_attribute.replace('_', ' ').split()
                
                # Build FactSpan query
                fact_stmt = select(FactSpan).where(
                    FactSpan.drug_name == intent.target_drug,
                    FactSpan.source_type == 'sentence',
                    or_(*[FactSpan.section_enum.ilike(f"%{s}%") for s in allowed_sections])
                )
                
                # Apply keyword filter if keywords exist
                if keywords:
                    keyword_filters = [FactSpan.sentence_text.ilike(f"%{kw}%") for kw in keywords]
                    fact_stmt = fact_stmt.where(or_(*keyword_filters))
                
                # Execute query
                span_result = await session.execute(fact_stmt)
                spans = span_result.scalars().all()
                
                logger.info(f"Path A++ found {len(spans)} candidate FactSpans for aggregation.")
                
                if spans:
                    # 2. Prepare candidates
                    candidates = [
                        CandidateSentence(
                            index=i+1,
                            text=span.sentence_text,
                            section_name=span.section_enum,
                            ids=span.fact_span_id
                        )
                        for i, span in enumerate(spans)
                    ]
                    
                    # 3. Aggregation via LLM
                    aggregator = AttributeAggregator()
                    selected = await aggregator.filter_sentences(
                        attribute=intent.target_attribute, 
                        question=result.original_query,
                        candidates=candidates
                    )
                    
                    logger.info(f"Path A++ selected {len(selected)} sentences after LLM filtering.")
                    
                    if selected:
                        # 4. Return Verbatim Result
                        formatted_content = aggregator.format_verbatim_response(selected)
                        
                        synthetic_section = {
                            "section_name": f"Attribute Evidence: {intent.target_attribute}",
                            "content_text": formatted_content,
                            "drug_name": intent.target_drug,
                            "attribute_provenance": True
                        }
                        
                        result.sections = [synthetic_section]
                        result.path_used = RetrievalPath.ATTRIBUTE_LOOKUP
                        result.total_results = len(selected)
                        result.attribute_name = intent.target_attribute
                        return result
                    else:
                        # LLM found candidates irrelevant -> Fallback or NO_RESULT
                        logger.info("Path A++ candidates rejected by LLM. Falling back to section lookup.")
                
            except Exception as e:
                logger.error(f"Attribute Aggregation failed: {e}")
                # Continue to fallback
            
            # FALLBACK: Original Section-Based Lookup
            # Build query to meaningful sections
            stmt = (
                select(MonographSection)
                .where(
                    or_(
                        MonographSection.drug_name == intent.target_drug,
                        MonographSection.brand_name == intent.target_drug,
                        MonographSection.generic_name == intent.target_drug
                    )
                )
                .where(
                    or_(*[MonographSection.section_name.ilike(f"%{s}%") for s in allowed_sections])
                )
                .order_by(MonographSection.page_start)
            )
            
            result.sql_executed = str(stmt)
            
            db_result = await session.execute(stmt)
            sections = db_result.scalars().all()
            
            if sections:
                result.sections = [self._section_to_dict(s) for s in sections]
                result.path_used = RetrievalPath.ATTRIBUTE_LOOKUP
                result.total_results = len(sections)
                result.attribute_name = intent.target_attribute
                
                logger.info(f"Path A++ returned {len(sections)} sections for attribute '{intent.target_attribute}'")
            else:
                logger.info(f"Path A++ found NO sections for attribute '{intent.target_attribute}'")
        
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
                .where(
                    or_(
                        MonographSection.drug_name == intent.target_drug,
                        MonographSection.brand_name == intent.target_drug,
                        MonographSection.generic_name == intent.target_drug
                    )
                )
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
                WHERE (
                    drug_name = :drug_name 
                    OR brand_name = :drug_name 
                    OR generic_name = :drug_name
                )
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
                    WHERE (
                        drug_name = :drug_name 
                        OR brand_name = :drug_name 
                        OR generic_name = :drug_name
                    )
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

    async def _path_d_bm25_factspan(
        self,
        plan: RetrievalPlan,
        result: RetrievalResult
    ) -> RetrievalResult:
        """
        Path D: BM25 FactSpan Retrieval.
        Uses PostgreSQL ts_rank to find relevant fact spans.
        """
        if not plan.search_phrases:
            return result
            
        logger.info(f"Routing to Path D: BM25 Search. Phrases: {plan.search_phrases}")
        
        async with get_session() as session:
            try:
                # Construct safe tsquery string (phrase1 | phrase2 | ...)
                # Remove special characters that might break tsquery syntax
                import re
                sanitized = [re.sub(r"[^\w\s\-]", "", p).strip() for p in plan.search_phrases if p.strip()]
                query_str = " ".join(sanitized)
                
                if not query_str:
                    return result
                
                # BM25 Ranking Query
                ts_query = func.plainto_tsquery('english', query_str)
                stmt = (
                    select(FactSpan, func.ts_rank(FactSpan.search_vector, ts_query).label('rank'))
                    .where(
                        FactSpan.drug_name == plan.drug,
                        FactSpan.search_vector.op('@@')(ts_query)
                    )
                    .order_by(desc('rank'))
                    .limit(20)
                )
                
                db_result = await session.execute(stmt)
                rows = db_result.all()  # List of (FactSpan, rank)
                
                if rows:
                    # Create synthetic section from fact spans
                    spans_content = []
                    valid_rows = []
                    
                    for span, rank in rows:
                        # FILTER: Skip TOC-like spans using strict regex
                        if self._is_junk_section(span.sentence_text):
                            continue
                        spans_content.append(f"[{span.section_enum}] {span.sentence_text}")
                        valid_rows.append(span)
                    
                    if not valid_rows:
                        logger.info("Path D (BM25) returned results but all were filtered as junk.")
                        return result

                    content_text = "\n\n".join(spans_content)
                    
                    synthetic_section = {
                        "section_name": "Relevant Facts (BM25)",
                        "content_text": content_text,
                        "drug_name": plan.drug,
                        "is_aggregated": True,
                        "path_provenance": "BM25"
                    }
                    
                    result.sections = [synthetic_section]
                    result.path_used = RetrievalPath.BM25_FACTSPAN
                    result.total_results = len(valid_rows)
                    logger.info(f"Path D (BM25) returned {len(valid_rows)} fact spans")
                else:
                    logger.info("Path D (BM25) returned 0 results")
                    
            except Exception as e:
                logger.error(f"Path D (BM25) failed: {e}")
                
        return result

    async def _path_e_global_scan(
        self,
        plan: RetrievalPlan,
        result: RetrievalResult
    ) -> RetrievalResult:
        """
        Path E: Global FactSpan Scan (Safety Net).
        Retrieves top matches using substring search if full-text fails.
        """
        logger.info("Routing to Path E: Global FactSpan Scan")
        
        async with get_session() as session:
            try:
                # Use first 5 search phrases for ILIKE check to avoid query explosion
                phrases = plan.search_phrases[:5]
                if not phrases:
                    return result
                
                filters = [FactSpan.sentence_text.ilike(f"%{p}%") for p in phrases]
                
                stmt = (
                    select(FactSpan)
                    .where(
                        FactSpan.drug_name == plan.drug,
                        or_(*filters)
                    )
                    .limit(50)  # Cap at 50 spans
                )
                
                db_result = await session.execute(stmt)
                spans = db_result.scalars().all()
                
                if spans:
                    spans_content = []
                    valid_spans = []
                    for span in spans:
                        # FILTER: Skip TOC-like spans using strict regex
                        if self._is_junk_section(span.sentence_text):
                            continue
                        spans_content.append(f"[{span.section_enum}] {span.sentence_text}")
                        valid_spans.append(span)
                    
                    if not valid_spans:
                        logger.info("Path E (Global Scan) returned results but all were filtered as junk.")
                        return result
                    
                    content_text = "\n\n".join(spans_content)
                    
                    synthetic_section = {
                        "section_name": "Extended Search Results",
                        "content_text": content_text,
                        "drug_name": plan.drug,
                        "is_aggregated": True,
                        "path_provenance": "Global Scan"
                    }
                    
                    result.sections = [synthetic_section]
                    result.path_used = RetrievalPath.GLOBAL_FACTSPAN_SCAN
                    result.total_results = len(valid_spans)
                    logger.info(f"Path E (Global Scan) returned {len(valid_spans)} spans")
                else:
                    logger.info("Path E (Global Scan) returned 0 results")
                    
            except Exception as e:
                logger.error(f"Path E (Global Scan) failed: {e}")
                
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
