"""
SPL XML Parser - FDA Structured Product Labeling
Extracts structured data from HL7 v3 XML documents with full namespace support
"""

from lxml import etree
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
from ingestion.table_preserver import TablePreserver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HL7 v3 Namespaces
NAMESPACES = {
    'hl7': 'urn:hl7-org:v3',
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
}

# LOINC Code Mapping for FDA Sections
LOINC_SECTION_MAPPING = {
    '34084-4': 'ADVERSE REACTIONS',
    '34090-1': 'CONTRAINDICATIONS',
    '43685-7': 'WARNINGS AND PRECAUTIONS',
    '34068-7': 'DOSAGE AND ADMINISTRATION',
    '34073-7': 'DRUG INTERACTIONS',
    '34070-3': 'CONTRAINDICATIONS',
    '34067-9': 'INDICATIONS AND USAGE',
    '34071-1': 'WARNINGS',
    '34066-1': 'BOXED WARNING',
    '42229-5': 'SPL UNCLASSIFIED SECTION',
    '34088-5': 'OVERDOSAGE',
    '34069-5': 'HOW SUPPLIED',
    '43678-2': 'MECHANISM OF ACTION',
    '34090-1': 'CLINICAL PHARMACOLOGY',
    '34092-7': 'CLINICAL STUDIES'
}

@dataclass
class SPLMetadata:
    """Core metadata from SPL document"""
    set_id: str
    version_number: str
    root_id: str
    effective_time: str
    drug_name: str
    ndc_codes: List[str]
    rxcui: Optional[str] = None

@dataclass
class SPLSection:
    """Structured section from SPL document"""
    loinc_code: str
    section_name: str
    text_content: str
    html_content: str
    is_table: bool
    parent_section: Optional[str] = None
    subsections: List['SPLSection'] = None

class SPLXMLParser:
    """
    Parse FDA SPL XML documents with full structure preservation
    """
    
    def __init__(self, xsl_path: Optional[str] = None):
        """
        Args:
            xsl_path: Path to FDA's official spl.xsl stylesheet
        """
        self.xsl_path = xsl_path
        self.table_preserver = None
        
        if xsl_path:
            try:
                # Initialize TablePreserver for XSLT transformations
                self.table_preserver = TablePreserver(xsl_path)
                logger.info("TablePreserver initialized successfully")
            except Exception as e:
                logger.warning(f"Could not load TablePreserver: {e}")
    
    def parse_document(self, xml_path: str) -> Tuple[SPLMetadata, List[SPLSection]]:
        """
        Parse complete SPL document
        
        Returns:
            (metadata, sections)
        """
        tree = etree.parse(xml_path)
        root = tree.getroot()
        
        metadata = self._extract_metadata(root)
        sections = self._extract_sections(root)
        
        logger.info(f"Parsed {metadata.drug_name}: {len(sections)} sections extracted")
        return metadata, sections
    
    def _extract_metadata(self, root) -> SPLMetadata:
        """Extract core document metadata"""
        
        # SetID (family identifier - same across versions)
        set_id_elem = root.find('.//hl7:setId', NAMESPACES)
        set_id = set_id_elem.get('root') if set_id_elem is not None else None
        
        # RootID (specific version identifier)
        root_id_elem = root.find('.//hl7:id', NAMESPACES)
        root_id = root_id_elem.get('root') if root_id_elem is not None else None
        
        # Version number
        version_elem = root.find('.//hl7:versionNumber', NAMESPACES)
        version = version_elem.get('value') if version_elem is not None else '1'
        
        # Effective time
        effective_elem = root.find('.//hl7:effectiveTime', NAMESPACES)
        effective_time = effective_elem.get('value') if effective_elem is not None else None
        
        # Drug name
        drug_name_elem = root.find('.//hl7:manufacturedProduct//hl7:name', NAMESPACES)
        drug_name = drug_name_elem.text if drug_name_elem is not None else 'Unknown'
        
        # NDC codes
        ndc_codes = []
        for code_elem in root.findall('.//hl7:code[@codeSystem="2.16.840.1.113883.6.69"]', NAMESPACES):
            ndc = code_elem.get('code')
            if ndc:
                ndc_codes.append(ndc)
        
        return SPLMetadata(
            set_id=set_id,
            version_number=version,
            root_id=root_id,
            effective_time=effective_time,
            drug_name=drug_name,
            ndc_codes=ndc_codes
        )
    
    def _extract_sections(self, root) -> List[SPLSection]:
        """Extract all sections with LOINC codes"""
        sections = []
        
        # Find all section elements
        for section_elem in root.findall('.//hl7:section', NAMESPACES):
            section = self._parse_section(section_elem)
            if section:
                sections.append(section)
        
        return sections
    
    def _parse_section(self, section_elem) -> Optional[SPLSection]:
        """Parse individual section element"""
        
        # Extract LOINC code
        code_elem = section_elem.find('.//hl7:code', NAMESPACES)
        if code_elem is None:
            return None
        
        loinc_code = code_elem.get('code')
        mapped_name = LOINC_SECTION_MAPPING.get(loinc_code, 'UNCLASSIFIED')
        
        # Extract title (preferred name)
        title_elem = section_elem.find('.//hl7:title', NAMESPACES)
        title_text = ''
        if title_elem is not None and title_elem.text:
            title_text = title_elem.text.strip()
            
        # Use title if available, otherwise mapped name
        # If mapped name is UNCLASSIFIED and we have a title, definitely use title
        if title_text:
            section_name = title_text
        else:
            section_name = mapped_name
        
        # Extract text content
        text_elem = section_elem.find('.//hl7:text', NAMESPACES)
        if text_elem is None:
            return None
        
        # Detect tables
        is_table = text_elem.find('.//hl7:table', NAMESPACES) is not None
        
        # Parse content using TablePreserver if available and it's a table
        if self.table_preserver:
             # Use XSLT to preserve structure
             text_content, is_table_verified = self.table_preserver.preserve_table_structure(text_elem)
             # If XSLT worked, text_content is markdown with table structure
             # If it failed or wasn't a table, it falls back to text extraction
             if is_table_verified:
                 is_table = True
        else:
            # Fallback for when no XSLT is provided
            text_content = self._extract_text_recursive(text_elem)
        
        # Get HTML (for record)
        html_content = etree.tostring(text_elem, encoding='unicode')
        
        # Parse subsections
        subsections = []
        for subsection_elem in section_elem.findall('.//hl7:section', NAMESPACES):
            subsection = self._parse_section(subsection_elem)
            if subsection:
                subsections.append(subsection)
        
        return SPLSection(
            loinc_code=loinc_code,
            section_name=section_name,
            text_content=text_content,
            html_content=html_content,
            is_table=is_table,
            subsections=subsections if subsections else None
        )
    
        return ' '.join(filter(None, parts))

    def _extract_text_recursive(self, elem) -> str:
        """
        Extract all text from element, preserving mixed content
        Handles text + tail from nested elements
        Adds formatting for list items
        """
        parts = []
        
        # Get element's direct text
        if elem.text:
            cleaned = elem.text.strip()
            if cleaned:
                # Add bullet for list items if not present
                if elem.tag.endswith('item') and not cleaned.startswith(('•', '-', '1.', 'a.')):
                    parts.append(f"• {cleaned}")
                else:
                    parts.append(cleaned)
        
        # Recursively get text from children
        for child in elem:
            child_text = self._extract_text_recursive(child)
            if child_text:
                parts.append(child_text)
                
            # SPECIAL HANDLING: Add newline after list items
            if child.tag.endswith('item') or child.tag.endswith('paragraph'):
                parts.append('\n')

            # Get tail text (text after child element)
            if child.tail:
                parts.append(child.tail.strip())
        
        # Join with space usually, but respect explicit newlines we added
        # This is a bit tricky with ' '.join. 
        # Instead, let's join and then cleanup.
        full_text = ' '.join(filter(None, parts))
        
        # Fix the "newline space" issue
        full_text = full_text.replace(' \n ', '\n').replace(' \n', '\n').replace('\n ', '\n')
        
        return full_text
    
    def apply_xslt_transformation(self, xml_path: str) -> str:
        """
        Apply FDA's official spl.xsl transformation
        Returns HTML string
        """
        if not self.xslt_transform:
            raise ValueError("XSLT stylesheet not loaded")
        
        tree = etree.parse(xml_path)
        result = self.xslt_transform(tree)
        
        return str(result)


class TablePreserver:
    """
    MANDATORY: Preserve table structure through XML → XSLT → HTML → Markdown pipeline
    Uses FDA's official spl.xsl stylesheet
    """
    
    def __init__(self, xsl_path: str):
        """
        Args:
            xsl_path: Path to FDA's official spl.xsl stylesheet (REQUIRED)
        """
        if not xsl_path:
            raise ValueError("XSLT stylesheet path is REQUIRED for table preservation")
        
        try:
            xslt_tree = etree.parse(xsl_path)
            self.xslt_transform = etree.XSLT(xslt_tree)
            logger.info(f"Loaded FDA XSLT stylesheet: {xsl_path}")
        except Exception as e:
            raise ValueError(f"CRITICAL: Failed to load FDA XSLT stylesheet: {e}")
    
    def apply_xslt_to_section(self, section_elem) -> str:
        """
        Apply FDA's official XSLT transformation to section
        Returns HTML string with preserved table structure
        """
        try:
            html_result = self.xslt_transform(section_elem)
            return str(html_result)
        except Exception as e:
            logger.error(f"XSLT transformation failed: {e}")
            return ""
    
    def html_to_markdown(self, html_content: str) -> str:
        """
        Convert HTML tables to Markdown tables
        PRESERVES row/column semantics (MANDATORY)
        """
        import html2text
        
        h = html2text.HTML2Text()
        h.body_width = 0  # No line wrapping - preserve structure
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_emphasis = False
        h.bypass_tables = False  # MUST process tables
        
        markdown = h.handle(html_content)
        return markdown.strip()
    
    def detect_table_in_text(self, text: str) -> bool:
        """Check if text contains markdown table"""
        lines = text.split('\n')
        
        # Look for markdown table separator (|---|---|)
        for line in lines:
            if line.strip().startswith('|') and '---' in line:
                return True
        
        return False
    
    def preserve_table_structure(self, section_elem) -> Tuple[str, bool]:
        """
        Complete table preservation pipeline:
        XML → XSLT → HTML → Markdown
        
        Returns:
            (markdown_text, is_table)
        """
        # Step 1: Apply FDA XSLT
        html_content = self.apply_xslt_to_section(section_elem)
        
        if not html_content:
            # Fallback to raw text extraction
            text = self._extract_text_recursive(section_elem)
            return text, False
        
        # Step 2: HTML → Markdown
        markdown_text = self.html_to_markdown(html_content)
        
        # Step 3: Verify table preservation
        is_table = self.detect_table_in_text(markdown_text)
        
        return markdown_text, is_table
    
    def _extract_text_recursive(self, elem) -> str:
        """Fallback text extraction if XSLT fails"""
        parts = []
        if elem.text:
            parts.append(elem.text.strip())
        for child in elem:
            parts.append(self._extract_text_recursive(child))
            if child.tail:
                parts.append(child.tail.strip())
        return ' '.join(filter(None, parts))


# Example usage
if __name__ == "__main__":
    # Example: Parse SPL XML file
    parser = SPLXMLParser(xsl_path='path/to/spl.xsl')  # Optional
    
    xml_file = 'path/to/lisinopril_spl.xml'
    metadata, sections = parser.parse_document(xml_file)
    
    print(f"Drug: {metadata.drug_name}")
    print(f"SetID: {metadata.set_id}")
    print(f"Version: {metadata.version_number}")
    print(f"NDC Codes: {', '.join(metadata.ndc_codes)}")
    print(f"\nSections extracted: {len(sections)}")
    
    for section in sections:
        print(f"  - {section.section_name} ({section.loinc_code})")
        print(f"    Text length: {len(section.text_content)} chars")
        print(f"    Contains table: {section.is_table}")
