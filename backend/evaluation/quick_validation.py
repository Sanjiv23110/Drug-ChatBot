"""
Quick 5-Question Spot Check for 50 PDFs

Tests 5 critical queries to validate RAG quality without running full RAGas evaluation.
Fast validation before scaling to 100+ PDFs.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vectorstore.faiss_store import FAISSVectorStore
from app.vectorstore.index_manager import IndexManager
from app.metadata.sqlite_store import SQLiteMetadataStore
from app.ingestion.embedder import AzureEmbedder
from app.retrieval.retriever import retrieve
from app.generation.answer_generator import AnswerGenerator

def test_rag_system():
    """Quick spot-check of 5 critical queries."""
    
    print("="*70)
    print("QUICK 50-PDF VALIDATION (5 Critical Queries)")
    print("="*70)
    
    # Initialize RAG
    print("\nInitializing RAG system...")
    faiss_store = FAISSVectorStore(dimension=1536)
    index_manager = IndexManager("data/faiss/medical_index")
    
    loaded = index_manager.load(dimension=1536)
    if loaded:
        faiss_store.index, faiss_store.chunk_ids, _ = loaded
        print(f"✓ Loaded FAISS index: {faiss_store.count()} vectors")
    else:
        print("❌ Failed to load index")
        return
    
    metadata_store = SQLiteMetadataStore("data/metadata.db")
    embedder = AzureEmbedder()
    generator = AnswerGenerator()
    
    # Test questions
    test_queries = [
        ("What are the contraindications for CeeNU?", "Critical safety query"),
        ("What is CeeNU indicated for?", "Indications query"),
        ("What are the common side effects of CeeNU?", "Side effects query"),
        ("How should CeeNU be stored?", "Storage query"),
        ("Is CeeNU safe during pregnancy?", "Pregnancy safety query"),
    ]
    
    print(f"\nRunning {len(test_queries)} test queries...\n")
    print("-"*70)
    
    results = []
    for i, (query, description) in enumerate(test_queries, 1):
        print(f"\n{i}. {description}")
        print(f"   Q: {query}")
        
        # Retrieve
        chunks = retrieve(
            query=query,
            faiss_store=faiss_store,
            metadata_store=metadata_store,
            embedder=embedder
        )
        
        # Generate
        result = generator.generate(query=query, context_chunks=chunks)
        answer = result['answer']
        
        # Evaluate
        is_valid = len(answer) > 20 and "not found" not in answer.lower()
        
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"   A: {answer[:150]}{'...' if len(answer) > 150 else ''}")
        print(f"   {status} ({len(chunks)} chunks, {len(answer)} chars)")
        
        results.append({
            'query': query,
            'answer': answer,
            'chunks': len(chunks),
            'chars': len(answer),
            'passed': is_valid
        })
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    pass_rate = (passed / total) * 100
    
    print(f"\nQueries Passed: {passed}/{total} ({pass_rate:.0f}%)")
    print(f"Average Chunks Retrieved: {sum(r['chunks'] for r in results) / total:.1f}")
    print(f"Average Answer Length: {sum(r['chars'] for r in results) / total:.0f} chars")
    
    print("\n" + "-"*70)
    
    if pass_rate >= 80:
        print("✅ VALIDATION PASSED - System ready for scaling!")
        print("   → Recommend adding more PDFs")
    elif pass_rate >= 60:
        print("⚠️  VALIDATION MARGINAL - System works but needs improvement")
        print("   → Can proceed but monitor quality")
    else:
        print("❌ VALIDATION FAILED - Fix issues before scaling")
        print("   → Review failed queries")
    
    print("="*70)
    
    return pass_rate >= 60

if __name__ == "__main__":
    success = test_rag_system()
    sys.exit(0 if success else 1)
