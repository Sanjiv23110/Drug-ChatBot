"""
Hierarchical Conflict Resolution - Post-Rerank Deduplication
Eliminates duplicate answers caused by parent/child chunk overlap.
"""

from typing import List, Dict
import logging
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


class HierarchicalConflictResolver:
    """
    Post-reranking deduplication to remove hierarchically overlapping chunks.
    
    Design:
    - Operates on top-K ranked results (K ≤ 50)
    - Compares only chunks from the SAME DRUG
    - Uses RapidFuzz for fast text similarity
    - Removes chunk B if similarity(A, B) ≥ 95% AND rank(A) < rank(B)
    - Preserves ranking order
    - O(K²) performance (acceptable for small K)
    """
    
    def __init__(self, similarity_threshold: float = 95.0):
        """
        Args:
            similarity_threshold: Minimum similarity (0-100) to consider chunks duplicates
        """
        self.similarity_threshold = similarity_threshold
    
    def resolve(self, ranked_chunks: List[Dict]) -> List[Dict]:
        """
        Remove hierarchically overlapping chunks from ranked results.
        
        Args:
            ranked_chunks: List of chunk dicts with 'text', 'rxcui', 'rank' keys
                          Assumes list is already sorted by rank (best first)
        
        Returns:
            Filtered list of chunks with duplicates removed
        """
        if not ranked_chunks:
            return []
        
        # Ensure chunks have rank information
        for i, chunk in enumerate(ranked_chunks):
            if 'rank' not in chunk:
                chunk['rank'] = i
        
        # Group chunks by drug (rxcui) for same-drug comparison
        drug_groups: Dict[str, List[Dict]] = {}
        for chunk in ranked_chunks:
            rxcui = str(chunk.get('rxcui', 'unknown'))
            if rxcui not in drug_groups:
                drug_groups[rxcui] = []
            drug_groups[rxcui].append(chunk)
        
        # Process each drug group independently
        filtered_chunks = []
        for rxcui, chunks in drug_groups.items():
            filtered_chunks.extend(self._resolve_drug_group(chunks))
        
        # Re-sort by rank to preserve ordering
        filtered_chunks.sort(key=lambda x: x['rank'])
        
        initial_count = len(ranked_chunks)
        final_count = len(filtered_chunks)
        removed_count = initial_count - final_count
        
        if removed_count > 0:
            logger.info(f"HierarchicalConflictResolver: Removed {removed_count} duplicate chunks "
                       f"({initial_count} → {final_count})")
        
        return filtered_chunks
    
    def _resolve_drug_group(self, chunks: List[Dict]) -> List[Dict]:
        """
        Resolve conflicts within a single drug group.
        
        Args:
            chunks: List of chunks from the same drug
        
        Returns:
            Filtered list with duplicates removed
        """
        if len(chunks) <= 1:
            return chunks
        
        # Sort by rank (best first)
        chunks_sorted = sorted(chunks, key=lambda x: x['rank'])
        
        # Track which chunks to keep
        keep_indices = set(range(len(chunks_sorted)))
        
        # Compare all pairs (i, j) where i < j
        for i in range(len(chunks_sorted)):
            if i not in keep_indices:
                continue
                
            chunk_a = chunks_sorted[i]
            # Prioritize raw_text (parent) > sentence_text (child) > text (fallback)
            text_a = chunk_a.get('raw_text') or chunk_a.get('sentence_text') or chunk_a.get('text', '')
            
            for j in range(i + 1, len(chunks_sorted)):
                if j not in keep_indices:
                    continue
                
                chunk_b = chunks_sorted[j]
                text_b = chunk_b.get('raw_text') or chunk_b.get('sentence_text') or chunk_b.get('text', '')
                
                # Calculate similarity
                similarity = fuzz.ratio(text_a, text_b)
                
                if similarity >= self.similarity_threshold:
                    # chunk_a has better rank (lower rank value), remove chunk_b
                    keep_indices.discard(j)
                    logger.debug(f"Removed duplicate chunk (similarity={similarity:.1f}%): "
                               f"rank {chunk_a['rank']} kept, rank {chunk_b['rank']} removed")
        
        # Return only kept chunks
        return [chunks_sorted[i] for i in sorted(keep_indices)]
