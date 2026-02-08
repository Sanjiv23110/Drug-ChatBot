"""Check if Contraindications text exists in DB for Tolinase and Renese"""
from qdrant_client import QdrantClient

c = QdrantClient(host='localhost', port=6333)

def check_section(drug_name, section_loinc):
    print(f"\n=== CHECKING {drug_name} (LOINC {section_loinc}) ===")
    
    # Simple scroll search for chunks matching drug and section
    # Using 'should' to catch slight variations or just filtering manually
    
    # Scroll and filter manually to be sure
    result = c.scroll(
        collection_name='spl_children', 
        limit=500, 
        with_payload=True,
        scroll_filter={
            "must": [
                {"key": "drug_name", "match": {"value": drug_name}}
            ]
        }
    )
    
    found = False
    for point in result[0]:
        p = point.payload
        if p.get('loinc_code') == section_loinc:
            print(f"[FOUND CHUNK] LOINC: {p.get('loinc_code')} | Section: {p.get('loinc_section')}")
            print(f"Text prefix: {p.get('sentence_text', '')[:100]}...")
            found = True
            
    if not found:
        print(f"NO CHUNKS FOUND with LOINC {section_loinc} for {drug_name}")
        # Let's list what sections ARE found
        print(f"Available sections for {drug_name}:")
        sections = set()
        for point in result[0]:
            sections.add(f"{point.payload.get('loinc_section')} ({point.payload.get('loinc_code')})")
        for s in sorted(sections):
            print(f"  - {s}")

if __name__ == "__main__":
    # Tolinase (Drug B)
    check_section("Tolinase", "34070-3")
    
    # Renese (Drug C)
    check_section("Renese", "34070-3")
