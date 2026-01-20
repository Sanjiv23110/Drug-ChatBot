"""
Deduplication utilities for retrieval results.

Removes duplicate chunks from multi-query retrieval to ensure
each piece of information appears only once in the final context.
"""
import logging
from typing import List, Dict, Set

def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Remove duplicate chunks from retrieval results.
    
    Chunks are considered duplicates if they come from the same
    file and have the same chunk index.
    
    Args:
        chunks: List of retrieved chunks with metadata
        
    Returns:
        List of unique chunks (preserves original order)
        
    Example:
        Input: [chunk_A, chunk_B, chunk_A, chunk_C]
        Output: [chunk_A, chunk_B, chunk_C]
    """
    seen: Set[str] = set()
    unique_chunks: List[Dict] = []
    
    for chunk in chunks:
        # Create unique identifier for chunk
        chunk_id = _get_chunk_id(chunk)
        
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique_chunks.append(chunk)
    
    duplicates_removed = len(chunks) - len(unique_chunks)
    if duplicates_removed > 0:
        logging.info(f"Removed {duplicates_removed} duplicate chunks ({len(unique_chunks)} unique)")
    
    return unique_chunks


def _get_chunk_id(chunk: Dict) -> str:
    """
    Generate unique identifier for a chunk.
    
    Uses file_path + chunk_index as the unique key.
    
    Args:
        chunk: Chunk dictionary with metadata
        
    Returns:
        Unique string identifier
    """
    file_path = chunk.get('file_path', 'unknown')
    chunk_index = chunk.get('chunk_index', -1)
    
    # Create identifier
    chunk_id = f"{file_path}_{chunk_index}"
    
    return chunk_id


def deduplicate_and_merge_scores(chunks: List[Dict]) -> List[Dict]:
    """
    Deduplicate chunks and merge scores if duplicates found.
    
    For duplicates, keeps the chunk with the highest score.
    
    Args:
        chunks: List of chunks with 'score' field
        
    Returns:
        Deduplicated chunks with best scores
    """
    chunk_map: Dict[str, Dict] = {}
    
    for chunk in chunks:
        chunk_id = _get_chunk_id(chunk)
        
        if chunk_id not in chunk_map:
            # First occurrence
            chunk_map[chunk_id] = chunk
        else:
            # Duplicate found - keep higher score
            existing_score = chunk_map[chunk_id].get('score', 0)
            new_score = chunk.get('score', 0)
            
            if new_score > existing_score:
                chunk_map[chunk_id] = chunk
    
    unique_chunks = list(chunk_map.values())
    
    duplicates_removed = len(chunks) - len(unique_chunks)
    if duplicates_removed > 0:
        logging.info(f"Deduplicated {duplicates_removed} chunks (kept highest scores)")
    
    return unique_chunks
