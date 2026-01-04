from openai import AzureOpenAI
from app.core.config import settings
import chromadb
import traceback

try:
    print("1. Testing Azure OpenAI embedding...")
    client = AzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
    )
    
    response = client.embeddings.create(
        input="test query about aspirin",
        model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    )
    
    query_embedding = response.data[0].embedding
    print(f"   ✓ Embedding generated: {len(query_embedding)} dimensions")
    
    print("\n2. Testing ChromaDB query with pre-generated embedding...")
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
    collection = chroma_client.get_collection("drug_monographs_azure")
    
    print(f"   ✓ Collection loaded: {collection.name}")
    
    print("\n3. Querying collection...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    print(f"   ✓ Query successful!")
    print(f"   ✓ Documents found: {len(results['documents'][0])}")
    
    if results['documents'][0]:
        print(f"   ✓ Sample: {results['documents'][0][0][:100]}...")

except Exception as e:
    print(f"\n❌ Error: {e}")
    traceback.print_exc()
