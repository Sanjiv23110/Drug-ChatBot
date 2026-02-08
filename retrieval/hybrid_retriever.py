"""
Hybrid Retrieval System
Combines dense (semantic) and sparse (lexical) search with cross-encoder reranking
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client.models import SparseVector
import logging

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
        logger.info(f"Loading dense embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
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
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
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
    
    def retrieve(
        self,
        query: str,
        filter_conditions: Optional[Dict] = None,
        retrieval_limit: int = 50,
        rerank_top_k: int = 5
    ) -> Tuple[List[Dict], Dict]:
        """
        Full retrieval pipeline with FALLBACK for empty results
        
        Args:
            query: User query
            filter_conditions: Metadata filters
            retrieval_limit: Max candidates to retrieve
            rerank_top_k: Final number of results after reranking
            
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
        
        # FALLBACK: If 0 results and we have a loinc_code filter, retry without it
        # This handles cases where XML sections don't match expected LOINC codes
        if len(candidates) == 0 and filter_conditions and 'loinc_code' in filter_conditions:
            logger.warning(f"No results with LOINC filter, falling back to RxCUI-only filter")
            fallback_filters = {k: v for k, v in filter_conditions.items() if k != 'loinc_code'}
            if fallback_filters:
                candidates = do_search(fallback_filters)
                used_fallback = True
        
        # FALLBACK 2: If still 0 results, try without any filter (semantic search only)
        if len(candidates) == 0 and filter_conditions:
            logger.warning(f"No results with filters, falling back to unfiltered search")
            candidates = do_search(None)
            used_fallback = True
        
        logger.info(f"Retrieved {len(candidates)} candidates")
        
        # Step 3: Cross-encoder reranking
        if candidates:
            reranked = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=rerank_top_k
            )
        else:
            reranked = []
        
        # Metadata about retrieval
        retrieval_metadata = {
            "num_candidates": len(candidates),
            "num_reranked": len(reranked),
            "filter_conditions": filter_conditions,
            "retrieval_method": "hybrid" if query_sparse else "dense_only",
            "used_fallback": used_fallback
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
