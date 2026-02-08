from qdrant_client import QdrantClient

# Check available methods
c = QdrantClient(host="localhost", port=6333)
methods = [m for m in dir(c) if not m.startswith('_')]
search_methods = [m for m in methods if 'search' in m.lower() or 'query' in m.lower()]

print("All search/query related methods:")
for m in search_methods:
    print(f"  - {m}")

# Check if search exists
print(f"\nHas 'search' method: {hasattr(c, 'search')}")
print(f"Has 'query' method: {hasattr(c, 'query')}")
print(f"Has 'search_batch' method: {hasattr(c, 'search_batch')}")
