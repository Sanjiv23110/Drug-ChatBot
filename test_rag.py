import os
import sys
sys.path.insert(0, 'C:/G/Maclens chatbot w api/backend')
os.chdir('C:/G/Maclens chatbot w api/backend')

print("Testing RAG Service...")
try:
    from app.services.rag_service import RagService
    rag = RagService()
    print("RAG Service initialized OK")
    
    print("\nAsking question...")
    result = rag.ask("What is Gravol?")
    
    print("\n=== SUCCESS! ===")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['metadatas']}")
    
except Exception as e:
    print(f"\n=== ERROR ===")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
