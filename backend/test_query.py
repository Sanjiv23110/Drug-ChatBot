from app.services.vector_store import VectorStoreService
import traceback

try:
    print("Creating VectorStoreService...")
    vs = VectorStoreService()
    print(f"Collection name: {vs.collection.name}")
    
    print("\nTesting query (without counting)...")
    results = vs.query("aspirin", n_results=3)
    
    print(f"✅ Query successful!")
    print(f"Documents returned: {len(results.get('documents', [[]])[0])}")
    
    if results.get('documents') and results['documents'][0]:
        print(f"Sample text: {results['documents'][0][0][:150]}...")
    
except Exception as e:
    print(f"❌ Error: {e}")
    traceback.print_exc()
