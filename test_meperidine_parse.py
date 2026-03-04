import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.spl_xml_parser import SPLXMLParser

XSLT_PATH = r"C:\G\solomindUS\data\spl.xsl"
MEPERIDINE_FILE = r"C:\G\solomindUS\data\xml\9C4F12E5-5E69-44D4-81A5-BD72C375EEF5 (1).xml"

print(f"Attempting to parse: {MEPERIDINE_FILE}")
print(f"File exists: {os.path.exists(MEPERIDINE_FILE)}")

try:
    parser = SPLXMLParser(xsl_path=XSLT_PATH if os.path.exists(XSLT_PATH) else None)
    print("Parser initialized")
    
    metadata, sections = parser.parse_document(MEPERIDINE_FILE)
    print(f"SUCCESS: Parsed {len(sections)} sections")
    print(f"Drug name: {metadata.drug_name}")
    print(f"Set ID: {metadata.set_id}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
