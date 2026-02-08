"""Check ranking of Renese adverse reactions chunk"""
import sys
import os

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.hybrid_retriever import DenseEmbedder, CrossEncoderReranker
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
import numpy as np

def check_ranking():
    # Load models
    embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
    
    # DB connection
    client = QdrantClient(host="localhost", port=6333)
    
    query = "What are the gastrointestinal adverse reactions of Renese?"
    query_vector = embedder.embed_query(query)
    
    # Get all Renese chunks (unfiltered by section, just drug)
    # Using RxCUI 224931 (Renese)
    # We simulate what the fallback retriever does
    
    # 1. Search with vector (using query_points for v1.16+)
    search_result = client.query_points(
        collection_name="spl_children",
        query=query_vector.tolist(),
        using="dense",
        query_filter={
            "must": [
                {"key": "rxcui", "match": {"value": "224931"}}
            ]
        },
        limit=100, # Start with 100
        with_payload=True
    ).points
    
    print(f"Query: {query}")
    print(f"Retrieved {len(search_result)} candidates (limit=100)")
    
    found_anorexia = False
    anorexia_rank = -1
    
    for i, point in enumerate(search_result):
        text = point.payload.get('sentence_text', '').lower()
        if "anorexia" in text and "gastric" in text:
            print(f"\n[RANK {i+1}] Found Target Chunk!")
            print(f"Score: {point.score}")
            print(f"Text: {point.payload.get('sentence_text')[:200]}...")
            found_anorexia = True
            anorexia_rank = i
            break
            
    if not found_anorexia:
        print("\nTarget chunk NOT found in top 100 dense results!")
    
    # Check Reranking impact if found
    if found_anorexia:
        # Prepare candidates format for reranker
        candidates = []
        for point in search_result:
            candidates.append({
                "chunk_id": point.id,
                "sentence_text": point.payload.get('sentence_text'),
                "score": point.score
            })
            
        # Rerank
        reranked = reranker.rerank(query, candidates, top_k=50)
        
        print(f"\nRe-ranking top 50...")
        for i, cand in enumerate(reranked):
            text = cand['sentence_text'].lower()
            if "anorexia" in text and "gastric" in text:
                print(f"[NEW RANK {i+1}] Target Chunk after Rerank")
                print(f"New Score: {cand['rerank_score']}")
                return

        print("Target chunk was present in dense retrieval but LOST in reranking top 50!")

if __name__ == "__main__":
    check_ranking()
