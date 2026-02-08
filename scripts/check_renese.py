"""Check the actual data for Renese adverse reactions"""
from qdrant_client import QdrantClient

c = QdrantClient(host='localhost', port=6333)

# Get ALL data and filter manually
result = c.scroll(collection_name='spl_children', limit=200, with_payload=True)

print("=== ALL RENESE CHUNKS ===")
renese_chunks = []
for point in result[0]:
    p = point.payload
    if p.get('drug_name', '').lower() == 'renese':
        renese_chunks.append(p)
        
print(f"Total Renese chunks: {len(renese_chunks)}")

print("\n=== RENESE LOINC DISTRIBUTION ===")
from collections import Counter
loinc_counter = Counter()
for chunk in renese_chunks:
    loinc = chunk.get('loinc_code', 'MISSING')
    loinc_counter[loinc] += 1

for loinc, count in sorted(loinc_counter.items()):
    print(f"  {loinc}: {count} chunks")

print("\n=== CHECKING LOINC 34084-4 (ADVERSE REACTIONS) ===")
adverse_chunks = [c for c in renese_chunks if c.get('loinc_code') == '34084-4']
print(f"Renese + LOINC 34084-4 chunks: {len(adverse_chunks)}")

if adverse_chunks:
    for chunk in adverse_chunks:
        print(f"\nRxCUI: {chunk.get('rxcui')}")
        print(f"Text: {chunk.get('sentence_text', '')[:200]}...")
else:
    print("NO CHUNKS FOUND! Checking why...")
    # Check what LOINC codes Renese HAS for adverse reactions
    for chunk in renese_chunks:
        section = chunk.get('loinc_section', '').upper()
        if 'ADVERSE' in section or 'REACTION' in section:
            print(f"\n  LOINC: {chunk.get('loinc_code')} Section: {section}")
            print(f"  Text: {chunk.get('sentence_text', '')[:150]}...")

print("\n=== CHECKING RXCUI FOR RENESE ===")
rxcuis = set()
for chunk in renese_chunks:
    rxcuis.add(chunk.get('rxcui', 'MISSING'))
print(f"RxCUIs in DB for Renese: {rxcuis}")
