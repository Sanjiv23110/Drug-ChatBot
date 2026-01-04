from app.services.rag_service import RagService

try:
    rag = RagService()
    result = rag.ask("what is aspirin?")
    print("✅ RAG works!")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result.get('metadatas', [])}")
except Exception as e:
    print(f"❌ RAG failed: {e}")
    import traceback
    traceback.print_exc()
