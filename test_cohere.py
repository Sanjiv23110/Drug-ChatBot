import sys
sys.path.insert(0, 'C:/G/Maclens chatbot w api/backend')

from app.services.vector_store import VectorStoreService
from app.services.rag_service import RagService

print("Testing VectorStore...")
vs = VectorStoreService()
print(f"Collection count: {vs.collection.count()}")

print("\nTesting query...")
try:
    result = vs.query('gravol', 2)
    print(f"Query OK! Found {len(result['documents'][0])} results")
    print(f"First result preview: {result['documents'][0][0][:100]}...")
except Exception as e:
    print(f"Query FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting RAG Service...")
try:
    rag = RagService()
    response = rag.ask("What is Gravol?")
    print(f"RAG OK! Answer: {response['answer'][:100]}...")
except Exception as e:
    print(f"RAG FAILED: {e}")
    import traceback
    traceback.print_exc()
