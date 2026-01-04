import sys
import traceback

print("Testing RAG System...")
print("-" * 60)

try:
    print("1. Importing modules...")
    from app.services.vector_store import VectorStoreService
    from openai import AzureOpenAI
    from app.core.config import settings
    print("   ✓ Modules imported")
    
    print("\n2. Creating VectorStoreService...")
    vs = VectorStoreService()
    print(f"   ✓ Collection: {vs.collection.name}")
    print(f"   ✓ Count: {vs.collection.count()}")
    
    print("\n3. Testing vector query...")
    results = vs.query("aspirin", n_results=3)
    print(f"   ✓ Query returned")
    print(f"   ✓ Documents found: {len(results.get('documents', [[]])[0])}")
    
    if results.get('documents'):
        print(f"   ✓ Sample: {results['documents'][0][0][:100]}...")
    
    print("\n4. Testing Azure OpenAI chat...")
    client = AzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
    )
    
    response = client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hello"}
        ],
        max_tokens=20
    )
    print(f"   ✓ Chat response: {response.choices[0].message.content}")
    
    print("\n5. Full RAG test...")
    from app.services.rag_service import RagService
    rag = RagService()
    result = rag.ask("what is aspirin?")
    print(f"   ✓ Answer: {result['answer'][:150]}...")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print(f"\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
