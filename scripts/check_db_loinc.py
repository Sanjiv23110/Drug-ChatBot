"""Check what LOINC codes are actually in the database"""
from qdrant_client import QdrantClient
from collections import Counter

c = QdrantClient(host='localhost', port=6333)

# Get ALL data from children collection  
result = c.scroll(collection_name='spl_children', limit=200, with_payload=True)

print("=== COUNTING LOINC CODES IN DATABASE ===\n")
loinc_counter = Counter()

for point in result[0]:
    p = point.payload
    loinc_code = p.get('loinc_code', 'MISSING')
    loinc_counter[loinc_code] += 1

print("LOINC Code Distribution:")
for code, count in sorted(loinc_counter.items()):
    print(f"  {code}: {count} chunks")

# Check if rxcui filter works
print("\n=== SAMPLE RXCUI VALUES ===")
rxcui_counter = Counter()
for point in result[0][:20]:
    rxcui = point.payload.get('rxcui', 'MISSING')
    drug = point.payload.get('drug_name', '?')
    rxcui_counter[(drug, rxcui)] += 1

for (drug, rxcui), count in rxcui_counter.items():
    print(f"  Drug: {drug} | RxCUI: {rxcui}")

# Show sample text
print("\n=== SAMPLE SENTENCE TEXT ===")
for point in result[0][:3]:
    text = point.payload.get('sentence_text', '')[:200]
    loinc = point.payload.get('loinc_code', 'MISSING')
    section = point.payload.get('loinc_section', 'MISSING')
    print(f"\n[{loinc} - {section}]")
    print(f"  {text}...")
