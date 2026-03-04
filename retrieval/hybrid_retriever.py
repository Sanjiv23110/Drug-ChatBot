"""
Hybrid Retrieval System
Combines dense (semantic) and sparse (lexical) search with cross-encoder reranking
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client.models import SparseVector
import logging
import json

from orchestrator.section_intent_normalizer import SectionIntentNormalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenseEmbedder:
    """
    Generate dense embeddings using biomedical models
    """
    
    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"):
        """
        Args:
            model_name: HuggingFace model identifier
        """
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading dense embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of shape [len(texts), embedding_dim]
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # For cosine similarity
            show_progress_bar=False
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed single query"""
        return self.embed([query])[0]


class SparseEmbedder:
    """
    Generate sparse embeddings using BM25
    """
    
    def __init__(self, corpus: Optional[List[str]] = None):
        """
        Args:
            corpus: List of documents to build BM25 index
        """
        self.corpus = corpus
        self.bm25 = None
        self.tokenized_corpus = None
        
        if corpus:
            self._build_bm25()
    
    def _build_bm25(self):
        """Build BM25 index from corpus"""
        from rank_bm25 import BM25Okapi
        
        logger.info(f"Building BM25 index for {len(self.corpus)} documents")
        # Simple whitespace tokenization
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index built")

    def _hash_token(self, token: str) -> int:
        """
        Generate stable hash for token
        Using zlib.crc32 which is stable across runs (unlike python hash())
        """
        import zlib
        return zlib.crc32(token.encode('utf-8'))

    def embed_query(self, query: str) -> SparseVector:
        """
        Generate sparse embedding for query
        For Qdrant, we use the same hashing for query terms
        """
        tokenized_query = query.lower().split()
        
        # If we have BM25 model, use IDF weights? 
        # For hybrid search in Qdrant, typically query is just 1.0 for terms
        # UNLESS we want to do BM25 expansion.
        # Simple approach: Term Frequency in Query (usually just 1s) mapped to hashes
        
        term_freq = {}
        for token in tokenized_query:
            term_freq[token] = term_freq.get(token, 0) + 1
            
        indices = []
        values = []
        
        for token, count in term_freq.items():
            indices.append(self._hash_token(token))
            values.append(float(count)) # Or use BM25 IDF if available
            
        return SparseVector(indices=indices, values=values)

    def embed_documents(self, documents: List[str]) -> List[SparseVector]:
        """
        Generate sparse embeddings for documents
        Uses Term Frequency and Stable Hashing
        """
        sparse_vectors = []
        
        for doc in documents:
            tokens = doc.lower().split()
            term_freq = {}
            
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            
            indices = []
            values = []
            
            for token, count in term_freq.items():
                indices.append(self._hash_token(token))
                values.append(float(count))
            
            sparse_vectors.append(SparseVector(indices=indices, values=values))
        
        return sparse_vectors


class CrossEncoderReranker:
    """
    Re-rank retrieved candidates using cross-encoder
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Args:
            model_name: Cross-encoder model
            
        Note: For biomedical domain, consider:
        - ncbi/MedCPT-Cross-Encoder (if available)
        - cross-encoder/ms-marco-MiniLM-L-12-v2 (general)
        """
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading cross-encoder: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device)
        logger.info("Cross-encoder loaded")
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Re-rank candidates using cross-encoder
        
        Args:
            query: User query
            candidates: List of candidate chunks
            top_k: Number of top results to return
            
        Returns:
            Top-k candidates sorted by cross-encoder score
        """
        if not candidates:
            return []
        
        # Create query-document pairs
        # Handle both child chunks (sentence_text) and parent chunks (raw_text)
        pairs = []
        for chunk in candidates:
            text = chunk.get('sentence_text', chunk.get('raw_text', ''))
            pairs.append([query, text])
        
        # Score pairs
        scores = self.model.predict(pairs)
        
        # Add scores to candidates
        for i, chunk in enumerate(candidates):
            chunk['rerank_score'] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        # Add rank information for conflict resolution
        for rank, chunk in enumerate(reranked[:top_k]):
            chunk['rank'] = rank
        
        return reranked[:top_k]


class HybridRetriever:
    """
    Complete hybrid retrieval pipeline
    """
    
    def __init__(
        self,
        dense_embedder: DenseEmbedder,
        sparse_embedder: Optional[SparseEmbedder],
        reranker: CrossEncoderReranker,
        vector_db_manager
    ):
        """
        Args:
            dense_embedder: Dense embedding model
            sparse_embedder: Sparse embedding model (optional)
            reranker: Cross-encoder reranker
            vector_db_manager: QdrantManager instance
        """
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.reranker = reranker
        self.vector_db = vector_db_manager
        self.section_normalizer = SectionIntentNormalizer()
    
    def retrieve(
        self,
        query: str,
        filter_conditions: Optional[Dict] = None,
        retrieval_limit: int = 50,
        rerank_top_k: int = 5,
        target_loinc: Optional[str] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Full retrieval pipeline with SECTION-FIRST RERANKING STRATEGY
        
        Args:
            query: User query
            filter_conditions: Metadata filters (should NOT include loinc_code if target_loinc is set)
            retrieval_limit: Max candidates to retrieve (Recommended: 75+)
            rerank_top_k: Final number of results after reranking
            target_loinc: Detected LOINC code for section filtering (NEW)
            
        Returns:
            (top_k_chunks, metadata)
        """
        # Step 1: Generate embeddings
        query_dense = self.dense_embedder.embed_query(query)
        
        query_sparse = None
        if self.sparse_embedder:
            try:
                query_sparse = self.sparse_embedder.embed_query(query)
            except Exception as e:
                logger.warning(f"Sparse embedding failed: {e}")
        
        # Step 2: Hybrid search (with fallback logic)
        def do_search(filters):
            if query_sparse:
                return self.vector_db.hybrid_search(
                    query_dense=query_dense,
                    query_sparse=query_sparse,
                    filter_conditions=filters,
                    limit=retrieval_limit
                )
            else:
                return self.vector_db.hybrid_search(
                    query_dense=query_dense,
                    query_sparse=SparseVector(indices=[], values=[]),
                    filter_conditions=filters,
                    limit=retrieval_limit
                )
        
        # Try with full filter first
        candidates = do_search(filter_conditions)
        used_fallback = False
        
        # FALLBACK 1: If LOINC was in filter and got 0 results, drop LOINC only
        # (legacy path — orchestrator no longer passes loinc in filter_conditions)
        if len(candidates) == 0 and filter_conditions and 'loinc_code' in filter_conditions:
            logger.warning("No results with LOINC filter, retrying without LOINC")
            fallback_filters = {k: v for k, v in filter_conditions.items() if k != 'loinc_code'}
            if fallback_filters:
                candidates = do_search(fallback_filters)
                used_fallback = True

        # FALLBACK 2: If rxcui resolved incorrectly and got 0 results, drop rxcui
        # but KEEP drug_name — the drug lock must NEVER be released.
        if len(candidates) == 0 and filter_conditions and 'rxcui' in filter_conditions and 'drug_name' in filter_conditions:
            logger.warning("0 results with drug_name+rxcui filter. Retrying with drug_name only (rxcui may be wrong).")
            drug_only_filter = {"drug_name": filter_conditions["drug_name"]}
            candidates = do_search(drug_only_filter)
            used_fallback = True

        # HARD STOP — never go fully unfiltered.
        # Cross-drug retrieval would violate the safety contract of this system.
        if len(candidates) == 0:
            logger.warning(
                f"No results found for filter: {filter_conditions}. "
                "Drug not in database or query mismatch. NOT falling back to cross-drug search."
            )


        
        total_retrieved = len(candidates)
        logger.info(f"Retrieved {total_retrieved} candidates")
        
        # Step 3: Section intent normalization (boost)
        # We still apply boost, as it might help within the subset or fallbacks
        candidates, intent_audit = self.section_normalizer.normalize_and_boost(
            query=query,
            candidates=candidates
        )
        
        # Step 4: SECTION-FIRST RERANKING
        # Strategy (drug already filtered by Qdrant above):
        #   Tier 1 — exact LOINC code match (modern XMLs with correct LOINC codes)
        #   Tier 2 — section NAME synonym match (older XMLs using 42229-5/unclassified
        #             with free-text titles like "ACTION" or "CLINICAL PHARMACOLOGY")
        #   Tier 3 — full set (no section intent detected or no matches in either tier)
        rerank_set = candidates
        section_subset = []
        detected_intent = intent_audit.get('detected_intent') if intent_audit else None

        if target_loinc:
            # Tier 1: exact LOINC code match
            section_subset = [
                c for c in candidates
                if c.get('metadata', {}).get('loinc_code') == target_loinc
            ]

            if section_subset:
                rerank_set = section_subset
                logger.info(
                    f"Section-First Tier 1 (LOINC match): {len(section_subset)} chunks for {target_loinc}"
                )
            else:
                # Tier 2: section NAME synonym match using SECTION_INTENT_MAP
                # Handles XMLs where clinical pharmacology is stored as LOINC 42229-5
                # with an English title like "ACTION", "PHARMACOLOGICAL ACTIONS", etc.
                if detected_intent:
                    try:
                        from config.section_intent_map import SECTION_INTENT_MAP
                        synonyms = {s.lower() for s in SECTION_INTENT_MAP.get(detected_intent, [])}
                        if synonyms:
                            section_subset = [
                                c for c in candidates
                                if (c.get('metadata', {}).get('loinc_section') or '').lower()
                                in synonyms
                            ]
                    except Exception as e:
                        logger.warning(f"Section-First Tier 2: SECTION_INTENT_MAP lookup failed: {e}")

                if section_subset:
                    rerank_set = section_subset
                    logger.info(
                        f"Section-First Tier 2 (name match): {len(section_subset)} chunks "
                        f"for intent '{detected_intent}'"
                    )
                else:
                    # Tier 3: no section filter — rerank everything
                    logger.warning(
                        f"Section-First Tier 3 (full fallback): no chunks matched "
                        f"LOINC {target_loinc} or intent '{detected_intent}' "
                        f"in {total_retrieved} candidates"
                    )
                    rerank_set = candidates
        else:
            logger.info("No target LOINC detected. Reranking full candidate set.")


        # Perform Reranking
        if rerank_set:
            reranked = self.reranker.rerank(
                query=query,
                candidates=rerank_set,
                top_k=rerank_top_k
            )
        else:
            reranked = []

        # ── PARENT FETCH (SOURCE OF TRUTH) ──────────────────────────────────
        #
        # FIX 1 — STABLE PARENT SCORING
        #   Instead of best child rank, accumulate the SUM of all matched
        #   children's reranker scores per parent. Score aggregation over
        #   multiple evidence points is stable across paraphrase variants —
        #   "what is the generic name" vs "whats the generic name" produce the
        #   same ranked parents.
        #
        # FIX 2 — POST-FETCH SECTION GUARD
        #   After fetching parents, drop any parent whose LOINC section is not
        #   compatible with the detected section intent. Prevents adjacent
        #   sections (e.g. CLINICAL PHARMACOLOGY appearing before INDICATIONS
        #   AND USAGE) from bleeding through because their children say
        #   "Dantrium is indicated...".
        #   Uses SECTION_INTENT_MAP synonyms — no hardcoding.
        #
        # FIX 3 — CONDITIONAL PARENT FETCH
        #   section_specific query (target_loinc detected) → fetch full parents
        #   fact query (Tier 3, no section LOINC) → return top-K child sentences
        #   directly. Prevents returning a 400-word paragraph when the user
        #   asked for a single value (e.g. "what is the elimination half-life").
        # ─────────────────────────────────────────────────────────────────────

        # Decide if parent-fetch is appropriate.
        # ONLY run parent-fetch when target_loinc is explicitly set by the orchestrator
        # (SectionClassifier recognised a section-level intent like "adverse reactions",
        # "indications and usage", etc.). section_subset being non-empty means Tier 2
        # narrowed the RERANK SCOPE but the user still asked a fact question, not a
        # full-section dump (e.g. "what is the elimination half-life" → Tier 2 uses
        # PHARMACOKINETICS synonyms to scope reranking but returns child sentences).
        is_section_query = (target_loinc is not None)

        if reranked and hasattr(self.vector_db, 'get_parents_by_ids') and is_section_query:

            # STEP 1 — Accumulate score SUM per parent (stable across paraphrases).
            # Same parent wins for "overdose" and "overdosage" because the
            # treatment-sentence children always outscore the animal-tox children.
            parent_score_map: dict = {}   # parent_id → cumulative score
            parent_child_count: dict = {} # parent_id → number of matched children
            for child in reranked:
                pid = child.get('metadata', {}).get('parent_id')
                if not pid:
                    continue
                score = child.get('rerank_score', child.get('score', 0.0))
                parent_score_map[pid] = parent_score_map.get(pid, 0.0) + score
                parent_child_count[pid] = parent_child_count.get(pid, 0) + 1

            unique_parent_ids = list(parent_score_map.keys())
            logger.info(
                f"Parent-fetch: {len(reranked)} children → "
                f"{len(unique_parent_ids)} unique parent_ids"
            )

            parent_docs = self.vector_db.get_parents_by_ids(unique_parent_ids)

            if parent_docs:
                # Sort by: 1) child-count desc (stable across paraphrases),
                #          2) score sum desc (tiebreaker for same count),
                #          3) parent_id asc (alphabetic stability)
                # Child count is deterministic: the treatment paragraph always has
                # MORE sentences than the animal-tox paragraph, so both "overdose"
                # and "overdosage" queries select the same parent.
                parent_docs.sort(
                    key=lambda p: (
                        -parent_child_count.get(p.get('parent_id'), 0),
                        -parent_score_map.get(p.get('parent_id'), 0.0),
                        p.get('parent_id', '')
                    )
                )

                # STEP 2 — Post-fetch section guard (LOINC + synonym check)
                allowed_loincs: set = set()
                allowed_section_names: set = set()
                if target_loinc:
                    allowed_loincs.add(target_loinc)
                if detected_intent:
                    try:
                        from config.section_intent_map import SECTION_INTENT_MAP
                        allowed_section_names = {
                            s.lower()
                            for s in SECTION_INTENT_MAP.get(detected_intent, [])
                        }
                    except Exception:
                        pass

                seen_fingerprints: set = set()
                filtered_parents = []

                for rank_idx, p in enumerate(parent_docs):
                    raw = p.get("raw_text", "")

                    if allowed_loincs or allowed_section_names:
                        p_loinc   = p.get("loinc_code", "")
                        p_section = (p.get("loinc_section") or "").lower()
                        if p_loinc not in allowed_loincs and p_section not in allowed_section_names:
                            logger.debug(
                                f"Section guard: dropped loinc='{p_loinc}' "
                                f"section='{p_section}'"
                            )
                            continue

                    fp = " ".join(raw.lower().split())[:300]
                    if fp in seen_fingerprints:
                        continue
                    seen_fingerprints.add(fp)
                    filtered_parents.append((rank_idx, p))

                logger.info(
                    f"Section guard: {len(filtered_parents)} parents survived "
                    f"({len(parent_docs) - len(filtered_parents)} removed)"
                )

                # ── STEP 3: BEST-PARENT → DIRECT RAW_TEXT ─────────────────────
                # Pick the top-ranked parent (already sorted by child-count,
                # then score sum, then parent_id for determinism).
                # Use the parent's raw_text DIRECTLY — it's the canonical
                # paragraph from ingestion (no prefix, no reconstruction needed).
                # ─────────────────────────────────────────────────────────────
                if filtered_parents:
                    _best_rank, best_parent = filtered_parents[0]
                    best_pid   = best_parent.get("parent_id")
                    best_score = parent_score_map.get(best_pid, 0.0)
                    best_count = parent_child_count.get(best_pid, 0)

                    logger.info(
                        f"Best-parent: pid={best_pid} "
                        f"children={best_count} score={best_score:.4f} "
                        f"section='{best_parent.get('loinc_section')}'"
                    )

                    # Use the parent's raw_text directly — canonical paragraph
                    reranked = [{
                        "raw_text":     best_parent.get("raw_text", ""),
                        "rerank_score": best_score,
                        "metadata": {
                            "parent_id":     best_pid,
                            "drug_name":     best_parent.get("drug_name"),
                            "rxcui":         best_parent.get("rxcui"),
                            "loinc_code":    best_parent.get("loinc_code"),
                            "loinc_section": best_parent.get("loinc_section"),
                            "set_id":        best_parent.get("set_id"),
                            "root_id":       best_parent.get("root_id"),
                            "version":       best_parent.get("version"),
                            "effective_date":best_parent.get("effective_date"),
                        },
                        "rank": 0
                    }]

                else:
                    logger.warning(
                        "Parent-fetch: no parents survived section guard"
                    )
                    reranked = []

            else:
                logger.warning(
                    "Parent-fetch: no parent docs fetched from Qdrant"
                )
                reranked = []
        else:
            # FIX 3: Fact query — skip parent-fetch, keep child sentences.
            # The reranker already scored children at sentence level; returning
            # the top-K children IS the verbatim answer for specific facts.
            #
            # Child deduplication: the same sentence can be stored as multiple
            # child chunks (e.g. it appears in two ancestor paragraphs). Filter
            # by sentence text fingerprint so the LLM extractor never sees
            # the same sentence twice (prevents identical bullet points in output).
            if reranked:
                seen_child_fps: set = set()
                deduped_children = []
                for c in reranked:
                    text = (c.get('sentence_text') or c.get('raw_text') or '').strip()
                    fp = " ".join(text.lower().split())[:200]
                    if fp and fp in seen_child_fps:
                        logger.debug(
                            f"Fact dedup: dropped duplicate child sentence: {text[:60]}..."
                        )
                        continue
                    if fp:
                        seen_child_fps.add(fp)
                    deduped_children.append(c)
                n_dropped = len(reranked) - len(deduped_children)
                reranked = deduped_children
                logger.info(
                    f"Fact query (no section LOINC): returning {len(reranked)} "
                    f"unique child sentences ({n_dropped} duplicates removed)."
                )

        # ─────────────────────────────────────────────────────────────────────


        # Mandatory Structured Logging
        logger.info(json.dumps({
            "event": "retrieval_stats",
            "detected_loinc": target_loinc,
            "total_candidates_retrieved": total_retrieved,
            "section_subset_count": len(section_subset),
            "rerank_scope_size": len(rerank_set),
            "final_candidate_count": len(reranked)
        }))

        # Metadata about retrieval
        retrieval_metadata = {
            "num_candidates": len(candidates),
            "num_reranked": len(reranked),
            "filter_conditions": filter_conditions,
            "retrieval_method": "hybrid" if query_sparse else "dense_only",
            "used_fallback": used_fallback,
            "section_intent": intent_audit
        }

        return reranked, retrieval_metadata
    
    def retrieve_for_section_query(
        self,
        query: str,
        drug_name: str,
        rxcui: str,
        loinc_code: str,
        retrieval_limit: int = 50,
        rerank_top_k: int = 5
    ) -> Tuple[List[Dict], Dict]:
        """
        Retrieve for section-specific query
        e.g., "What are adverse reactions of Lisinopril?"
        """
        filter_conditions = {
            "rxcui": rxcui,
            "loinc_code": loinc_code
        }
        
        return self.retrieve(
            query=query,
            filter_conditions=filter_conditions,
            retrieval_limit=retrieval_limit,
            rerank_top_k=rerank_top_k
        )
    
    def retrieve_for_class_query(
        self,
        query: str,
        rxcui_list: List[str],
        loinc_code: str,
        retrieval_limit: int = 100,
        rerank_top_k: int = 10
    ) -> Tuple[List[Dict], Dict]:
        """
        Retrieve for class-based query
        e.g., "What are adverse reactions of ACE inhibitors?"
        """
        filter_conditions = {
            "rxcui": rxcui_list,  # Match any of these RxCUIs
            "loinc_code": loinc_code
        }
        
        return self.retrieve(
            query=query,
            filter_conditions=filter_conditions,
            retrieval_limit=retrieval_limit,
            rerank_top_k=rerank_top_k
        )


# Example usage
if __name__ == "__main__":
    from vector_db.qdrant_manager import QdrantManager
    
    # Initialize components
    dense_embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
    vector_db = QdrantManager(host="localhost", port=6333)
    
    # Sparse embedder requires corpus (build during ingestion)
    # sparse_embedder = SparseEmbedder(corpus=all_documents)
    
    # Create retriever
    retriever = HybridRetriever(
        dense_embedder=dense_embedder,
        sparse_embedder=None,  # Optional
        reranker=reranker,
        vector_db_manager=vector_db
    )
    
    # Example query
    query = "What are the adverse reactions of Lisinopril?"
    
    results, metadata = retriever.retrieve_for_section_query(
        query=query,
        drug_name="Lisinopril",
        rxcui="203644",
        loinc_code="34084-4",  # ADVERSE REACTIONS
        retrieval_limit=50,
        rerank_top_k=5
    )
    
    print(f"Retrieved {len(results)} results")
    print(f"Metadata: {metadata}")
    
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Score: {result['rerank_score']:.3f}")
        print(f"Text: {result['raw_text'][:200]}...")
