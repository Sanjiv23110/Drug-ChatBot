"""
Quick live test: verify drug_name filter works at DB level.
Simulates what the orchestrator now sends to hybrid_search.
"""
import sys, os
sys.path.insert(0, r"C:\G\solomindUS")

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(host="127.0.0.1", port=6333)

# Test 1: drug_name filter alone (new primary filter)
result_drug_name, _ = client.scroll(
    collection_name="spl_children",
    scroll_filter=Filter(must=[
        FieldCondition(key="drug_name", match=MatchValue(value="Renese"))
    ]),
    limit=5,
    with_payload=True,
    with_vectors=False
)

print(f"Test 1 — drug_name='Renese' filter: {len(result_drug_name)} results")
for r in result_drug_name[:3]:
    p = r.payload
    print(f"  drug_name='{p.get('drug_name')}' loinc_section='{p.get('loinc_section')}' | {p.get('sentence_text','')[:60]}...")

# Test 2: confirm 'mechanism of action' as drug_name returns 0 (old wrong path)
result_wrong, _ = client.scroll(
    collection_name="spl_children",
    scroll_filter=Filter(must=[
        FieldCondition(key="drug_name", match=MatchValue(value="mechanism of action"))
    ]),
    limit=5,
    with_payload=True,
    with_vectors=False
)
print(f"\nTest 2 — drug_name='mechanism of action' filter (should be 0): {len(result_wrong)} results")

# Test 3: ACTION section of Renese specifically
result_action, _ = client.scroll(
    collection_name="spl_children",
    scroll_filter=Filter(must=[
        FieldCondition(key="drug_name", match=MatchValue(value="Renese")),
        FieldCondition(key="loinc_section", match=MatchValue(value="ACTION"))
    ]),
    limit=10,
    with_payload=True,
    with_vectors=False
)
print(f"\nTest 3 — Renese ACTION section children: {len(result_action)} chunks")
for r in result_action:
    print(f"  | {r.payload.get('sentence_text','')[:80]}...")

print("\n=== ALL TESTS PASSED ===" if len(result_drug_name) > 0 and len(result_wrong) == 0 and len(result_action) > 0 else "\n=== ISSUES FOUND ===")
