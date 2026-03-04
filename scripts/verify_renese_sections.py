import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(host="127.0.0.1", port=6333)
result, _ = client.scroll(
    collection_name="spl_parents",
    scroll_filter=Filter(must=[FieldCondition(key="drug_name", match=MatchValue(value="Renese"))]),
    limit=30,
    with_payload=True,
    with_vectors=False
)
print(f"Renese parent chunks: {len(result)}\n")
for p in result:
    pay = p.payload
    print(f"LOINC={pay.get('loinc_code'):<12} section='{pay.get('loinc_section')}'")
