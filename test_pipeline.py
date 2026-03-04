"""
Diagnostic script to test parsing and chunking pipeline
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.spl_xml_parser import SPLXMLParser
from ingestion.hierarchical_chunking import HierarchicalChunker

# Test file
XML_FILE = r"C:\G\solomindUS\data\xml\drug(b) (1).xml"
XSLT_PATH = r"C:\G\solomindUS\data\spl.xsl"

print("="*80)
print("DIAGNOSTIC: Testing Parsing - Chunking Pipeline")
print("="*80)

# 1. Test Parsing
print("\n[STEP 1] Initializing Parser...")
parser = SPLXMLParser(xsl_path=XSLT_PATH if os.path.exists(XSLT_PATH) else None)
print(f"[OK] Parser initialized")

print(f"\n[STEP 2] Parsing XML file: {os.path.basename(XML_FILE)}")
try:
    metadata, sections = parser.parse_document(XML_FILE)
    print(f"[OK] Parsing successful!")
    print(f"  - Drug name: {metadata.drug_name}")
    print(f"  - Set ID: {metadata.set_id}")
    print(f"  - NDC codes: {metadata.ndc_codes}")
    print(f"  - Sections extracted: {len(sections)}")
    
    if sections:
        print(f"\n  Sample sections (first 3):")
        for i, section in enumerate(sections[:3]):
            print(f"    [{i+1}] LOINC: {section.loinc_code} | {section.section_name}")
            print(f"        Text length: {len(section.text_content)} chars")
            print(f"        Preview: {section.text_content[:100]}...")
    else:
        print("  [WARNING]  WARNING: No sections extracted!")
        
except Exception as e:
    print(f"[FAIL] Parsing failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. Test Chunking
print(f"\n[STEP 3] Initializing Chunker...")
chunker = HierarchicalChunker()
print(f"[OK] Chunker initialized")

print(f"\n[STEP 4] Chunking document...")
try:
    parents, children = chunker.chunk_document(metadata, sections)
    print(f"[OK] Chunking successful!")
    print(f"  - Parent chunks: {len(parents)}")
    print(f"  - Child chunks: {len(children)}")
    
    if parents:
        print(f"\n  Sample parent (first 1):")
        p = parents[0]
        print(f"    Parent ID: {p.parent_id}")
        print(f"    LOINC: {p.loinc_code} | {p.loinc_section}")
        print(f"    Text length: {len(p.raw_text)} chars")
        print(f"    Preview: {p.raw_text[:100]}...")
    else:
        print("  [WARNING]  WARNING: No parent chunks generated!")
    
    if children:
        print(f"\n  Sample children (first 3):")
        for i, c in enumerate(children[:3]):
            print(f"    [{i+1}] Child ID: {c.child_id}")
            print(f"        Parent ID: {c.parent_id}")
            print(f"        Text: {c.sentence_text[:80]}...")
    else:
        print("  [WARNING]  WARNING: No child chunks generated!")
        
except Exception as e:
    print(f"[FAIL] Chunking failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
