"""
MMR-based reranker with explicit similarity definitions.

Critical guarantees:
- Cosine similarity only (explicitly defined)
- L2 normalization enforced
- Deterministic (stable sorting on ties)
- No external dependencies
"""
import numpy as np
from typing import List, Dict


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalized vectors.
    
    CRITICAL: Both vectors MUST be L2-normalized before calling.
    
    Args:
        vec1: Normalized vector (1536,)
        vec2: Normalized vector (1536,)
        
    Returns:
        float in [0, 1] where 1 = identical, 0 = orthogonal
    """
    return float(np.dot(vec1, vec2))


def mmr_rerank(
    candidates: List[Dict],
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int = 8,
    lambda_param: float = 0.7
) -> List[Dict]:
    """
    Maximal Marginal Relevance reranking for diversity.
    
    Balances relevance to query with diversity from already-selected chunks.
    Prevents retrieving 8 redundant chunks from same page.
    
    Args:
        candidates: List of candidate chunks (must have distance, file_path, page_num)
        query_embedding: Query embedding vector (1536,)
        candidate_embeddings: Candidate embedding vectors (N, 1536)
        top_k: Number of results to return (default 8)
        lambda_param: Relevance vs diversity balance
                     - 1.0 = pure relevance (may be redundant)
                     - 0.0 = pure diversity (may miss key info)
                     - 0.7 = balanced (recommended for medical)
    
    Formula:
        MMR(c) = λ × cos_sim(c, query) - (1-λ) × max(cos_sim(c, selected))
    
    Returns:
        Reranked list of candidates (length = min(top_k, len(candidates)))
        
    Guarantees:
        - Deterministic: same input → same output
        - Uses only cosine similarity
        - All vectors normalized
        - Stable sorting on ties
    """
    if not candidates:
        return []
    
    # Ensure L2 normalization
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    candidate_embeddings = candidate_embeddings / np.linalg.norm(
        candidate_embeddings, axis=1, keepdims=True
    )
    
    # Pre-sort candidates for determinism on ties
    # Sort by: distance (ascending), file_path (lex), page_num (ascending)
    candidates_sorted = sorted(
        enumerate(candidates),
        key=lambda x: (x[1]['distance'], x[1].get('file_path', ''), x[1].get('page_num', 0))
    )
    
    selected_indices = []
    selected_embeddings = []
    
    for _ in range(min(top_k, len(candidates))):
        best_score = -float('inf')
        best_idx = None
        
        for idx, (orig_idx, candidate) in enumerate(candidates_sorted):
            if orig_idx in selected_indices:
                continue
            
            # Relevance: similarity to query
            relevance = cosine_similarity(
                query_embedding,
                candidate_embeddings[orig_idx]
            )
            
            # Diversity: max similarity to already selected
            if selected_embeddings:
                diversity_penalty = max(
                    cosine_similarity(
                        candidate_embeddings[orig_idx],
                        sel_emb
                    )
                    for sel_emb in selected_embeddings
                )
            else:
                diversity_penalty = 0.0
            
            # MMR score
            score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            
            if score > best_score:
                best_score = score
                best_idx = orig_idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            selected_embeddings.append(candidate_embeddings[best_idx])
    
    return [candidates[i] for i in selected_indices]
