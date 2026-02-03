"""
Cross-Encoder Reranker Module.

Purpose:
    Re-scores candidate sentences based on their relevance to the user's query.
    Uses a Cross-Encoder model (ms-marco-MiniLM-L-6-v2) which outputs a similarity score
    for a pair (query, document). This is significantly more precise than BM25 or
    Bi-Encoder vectors for niche/specific queries.

Design:
    - Input: Query (str), Candidates (list of text/objects)
    - Output: Re-ordered candidates with scores
    - Failure Mode: Returns original list if model fails or library missing.
"""
import logging
from typing import List, Any, Tuple, Dict

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Reranks candidates using a Cross-Encoder model.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the Cross-Encoder model.
        
        Args:
            model_name: HuggingFace model identifier.
                        Default is efficient/accurate implementation.
        """
        self.model = None
        self.model_name = model_name
        self.enabled = False
        
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading Cross-Encoder model: {model_name}...")
            # max_length=512 is standard for this model, but our sentences are shorter.
            self.model = CrossEncoder(model_name, max_length=512)
            self.enabled = True
            logger.info("Cross-Encoder model loaded successfully.")
        except ImportError:
            logger.error("sentence-transformers not installed. Reranking disabled.")
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder model: {e}")

    def rerank_fact_spans(
        self, 
        query: str, 
        candidates: List[Any], 
        top_k: int = 10
    ) -> List[Any]:
        """
        Rerank a list of FactSpan objects (or similar objects with a 'sentence_text' attr).
        
        Args:
            query: User query string
            candidates: List of objects. MUST have attribute 'sentence_text'.
            top_k: Number of results to return after reranking.
            
        Returns:
            List of objects, sorted by relevance vs query.
        """
        if not self.enabled or not candidates or not query:
            return candidates[:top_k]
            
        try:
            # Prepare pairs for the model: [(query, doc1), (query, doc2), ...]
            # We use 'sentence_text' from the FactSpan object
            pairs = []
            valid_candidates = []
            
            for cand in candidates:
                text = getattr(cand, 'sentence_text', None)
                if text:
                    pairs.append((query, text))
                    valid_candidates.append(cand)
            
            if not pairs:
                return candidates[:top_k]
                
            # Predict scores
            # scores will be a numpy array of floats
            scores = self.model.predict(pairs)
            
            # Combine candidates with scores
            scored_candidates = []
            for i, score in enumerate(scores):
                scored_candidates.append((score, valid_candidates[i]))
            
            # Sort by score descending
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Extract top_k objects
            final_results = [item[1] for item in scored_candidates[:top_k]]
            
            logger.info(f"Reranked {len(candidates)} candidates -> kept Top-{len(final_results)}")
            return final_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Returning original order.")
            # Fallback to original order
            return candidates[:top_k]
