"""
Diagnostic script to test each component of the Solomind chatbot
"""
import sys
import traceback

print("="*60)
print("SOLOMIND CHATBOT DIAGNOSTIC TEST")
print("="*60)

# Test 1: Check environment variables
print("\n1. Testing environment variables...")
try:
    from app.core.config import settings
    print(f"   ✓ AZURE_OPENAI_ENDPOINT: {settings.AZURE_OPENAI_ENDPOINT[:30]}...")
    print(f"   ✓ AZURE_OPENAI_API_KEY: {settings.AZURE_OPENAI_API_KEY[:10]}...")
    print(f"   ✓ CHROMA_DB_DIR: {settings.CHROMA_DB_DIR}")
    print(f"   ✓ EMBEDDING_DEPLOYMENT: {settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")
    print(f"   ✓ CHAT_DEPLOYMENT: {settings.AZURE_OPENAI_CHAT_DEPLOYMENT}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    sys.exit(1)

# Test 2: Check ChromaDB connection
print("\n2. Testing ChromaDB connection...")
try:
    import chromadb
    client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
    collection = client.get_collection("drug_monographs_azure")
    count = collection.count()
    print(f"   ✓ ChromaDB connected")
    print(f"   ✓ Collection found: drug_monographs_azure")
    print(f"   ✓ Documents in database: {count}")
    
    if count == 0:
        print(f"   ✗ WARNING: Database is empty!")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test Azure OpenAI embedding
print("\n3. Testing Azure OpenAI Embeddings...")
try:
    from openai import AzureOpenAI
    
    client = AzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
    )
    
    response = client.embeddings.create(
        input="test query",
        model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    )
    
    embedding = response.data[0].embedding
    print(f"   ✓ Embedding API working")
    print(f"   ✓ Embedding dimension: {len(embedding)}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test vector search
print("\n4. Testing vector search...")
try:
    from app.services.vector_store import VectorStoreService
    
    vs = VectorStoreService()
    results = vs.query("gravol", n_results=3)
    
    print(f"   ✓ Vector search working")
    print(f"   ✓ Found {len(results['documents'][0]) if results.get('documents') else 0} results")
    
    if results.get('documents') and results['documents'][0]:
        print(f"   ✓ Sample result: {results['documents'][0][0][:100]}...")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test Azure OpenAI chat completion
print("\n5. Testing Azure OpenAI Chat...")
try:
    response = client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'test successful'"}
        ],
        max_tokens=50
    )
    
    answer = response.choices[0].message.content
    print(f"   ✓ Chat API working")
    print(f"   ✓ Response: {answer}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test full RAG pipeline
print("\n6. Testing full RAG pipeline...")
try:
    from app.services.rag_service import RagService
    
    rag = RagService()
    result = rag.ask("what is gravol?")
    
    print(f"   ✓ RAG service working")
    print(f"   ✓ Answer: {result['answer'][:200]}...")
    print(f"   ✓ Sources: {result.get('metadatas', [])}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
