import sys
sys.path.insert(0, r"C:\G\solomindUS")
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
client = QdrantClient(host="127.0.0.1", port=6333)
r, _ = client.scroll("spl_parents",
    scroll_filter=Filter(must=[FieldCondition(key="drug_name", match=MatchValue(value="Dantrium"))]),
    limit=50, with_payload=True, with_vectors=False)
print(f"Dantrium parents: {len(r)}")
for p in r:
    pay = p.payload
    print(f"  loinc={pay.get('loinc_code'):<12} section=\"{pay.get('loinc_section')}\"")
