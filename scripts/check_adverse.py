"""Check what Renese adverse reactions text exists in DB"""
from qdrant_client import QdrantClient

c = QdrantClient(host='localhost', port=6333)

# Get ALL Renese data
result = c.scroll(collection_name='spl_children', limit=200, with_payload=True)

print("=== SEARCHING FOR ADVERSE REACTIONS TEXT FOR RENESE ===\n")

adverse_keywords = ['anorexia', 'nausea', 'vomiting', 'cramping', 'diarrhea', 'constipation', 
                    'jaundice', 'pancreatitis', 'gastrointestinal', 'adverse']

for point in result[0]:
    p = point.payload
    if p.get('drug_name', '').lower() != 'renese':
        continue
        
    text = p.get('sentence_text', '').lower()
    for keyword in adverse_keywords:
        if keyword in text:
            print(f"\n[LOINC: {p.get('loinc_code')} | Section: {p.get('loinc_section')}]")
            print(f"Text: {p.get('sentence_text', '')[:500]}")
            print("-" * 80)
            break

print("\n=== ALL RENESE SECTION TEXTS (first 200 chars) ===")
for point in result[0][:10]:
    p = point.payload
    if p.get('drug_name', '').lower() != 'renese':
        continue
    print(f"\n{p.get('loinc_section')}:")
    print(f"  {p.get('sentence_text', '')[:200]}")
