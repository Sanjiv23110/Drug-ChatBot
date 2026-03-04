import sys
import os
import logging
from lxml import etree

# Add root to path
sys.path.append(os.getcwd())

from ingestion.spl_xml_parser import SPLXMLParser, NAMESPACES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_parsing():
    xml_path = r"c:\G\solomindUS\data\xml\drug(c) (1).xml"
    if not os.path.exists(xml_path):
        print(f"File not found: {xml_path}")
        return

    print(f"Parsing {xml_path}...")
    parser = SPLXMLParser() # No XSLT for now, testing raw extraction fallback
    
    metadata, sections = parser.parse_document(xml_path)
    
    print(f"Drug Name: {metadata.drug_name}")
    print(f"Found {len(sections)} sections.")
    
    found_anorexia = False
    
    for i, section in enumerate(sections):
        # recursive check for subsections
        check_section(section, 0)

def check_section(section, depth):
    indent = "  " * depth
    # print(f"{indent}Section: {section.section_name} ({section.loinc_code}) - Length: {len(section.text_content)}")
    
    if "anorexia" in section.text_content:
        print(f"\n{indent}[FOUND 'anorexia'] in Section: {section.section_name} ({section.loinc_code})")
        print(f"{indent}Text content prefix: {section.text_content[:200]}")
    
    if section.subsections:
        for sub in section.subsections:
            check_section(sub, depth + 1)

if __name__ == "__main__":
    debug_parsing()
