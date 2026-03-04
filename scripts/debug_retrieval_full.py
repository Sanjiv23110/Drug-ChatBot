"""Debug retrieval for failing queries"""
import sys
import os

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.hybrid_retriever import HybridRetriever, DenseEmbedder, SparseEmbedder, CrossEncoderReranker
from qdrant_client import QdrantClient
from vector_db.qdrant_manager import QdrantManager

def debug_query(query, drug_name, expected_loinc):
    print(f"\n\n=== DEBUGGING QUERY: '{query}' ===")
    
    embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
    # Backend uses QdrantManager with spl_children collection
    qm = QdrantManager(host="localhost", port=6333, collection_name="spl_children")
    
    retriever = HybridRetriever(embedder, None, reranker, qm)
    
    # 1. Simulate what Orchestrator does (Intent + Section)
    # Orchestrator would map "contraindications" -> 34070-3
    filter_conditions = {
        "drug_name": drug_name, # In reality it uses RxCUI, but lets test with name if possible or assume RxCUI is correct
        # We need RxCUI for the filter actually. 
    }
    
    # Let's get RxCUI first
    if drug_name == "Tolinase":
        rxcui = "202479" # Tolinase (approx) or need to lookup. Tolinase -> 202479 (brand) or 10642 (ingredient)
        # From previous logs: Tolinase -> RxCUI 220338
        rxcui = "220338"
    elif drug_name == "Renese":
        rxcui = "224931" # From logs
    
    filter_conditions = {
        "rxcui": rxcui,
        "loinc_code": expected_loinc
    }
    
    print(f"Filters: {filter_conditions}")
    
    # Run retrieval
    chunks, meta = retriever.retrieve(
        query=query,
        filter_conditions={k:v for k,v in filter_conditions.items() if v}, # remove None
        retrieval_limit=50,
        rerank_top_k=15
    )
    
    print(f"Retrieved {len(chunks)} chunks.")
    
    found_relevant = False
    for i, chunk in enumerate(chunks):
        text = chunk['raw_text'].lower()
        print(f"\n[Rank {i+1}] Score: {chunk.get('rerank_score', 'N/A')}")
        meta = chunk.get('metadata', {})
        print(f"Section: {meta.get('loinc_section')} ({meta.get('loinc_code')})")
        print(f"Text: {text[:200]}...")
        
        # Check relevance
        if "strength" in query:
            if "mg" in text or "tablet" in text:
                print("-> POTENTIAL MATCH (Strengths)")
        elif "contraindication" in query:
            if "hypersensitivity" in text or "contraindicat" in text:
                print("-> POTENTIAL MATCH (Contraindications)")

if __name__ == "__main__":
    # 1. Contraindications for Renese
    debug_query("What are the contraindications for Renese?", "Renese", "34070-3")
    
    # 2. Contraindications for Tolinase
    debug_query("What are the contraindications for Tolinase?", "Tolinase", "34070-3")
    
    # 3. Strengths for Tolinase
    # Note: "Strengths" usually Maps to HOW SUPPLIED (34069-5) or DOSAGE (34068-7) or DESCRIPTION (34089-3)
    # Let's try WITHOUT LOINC first (just semantic) to see where it lands
    debug_query("What are the available strengths of Tolinase tablets?", "Tolinase", None)
