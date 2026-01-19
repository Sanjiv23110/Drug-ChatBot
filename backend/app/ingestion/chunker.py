"""
Section-aware chunker with adaptive sizing.

Medical documents have structured sections - preserve them.
"""
import re
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Medical document section headers (Health Canada monograph standard)
SECTION_HEADERS = {
    "INDICATIONS AND CLINICAL USE",
    "CONTRAINDICATIONS",
    "WARNINGS AND PRECAUTIONS", 
    "ADVERSE REACTIONS",
    "DRUG INTERACTIONS",
    "DOSAGE AND ADMINISTRATION",
    "OVERDOSAGE",
    "ACTION AND CLINICAL PHARMACOLOGY",
    "PHARMACOKINETICS",
    "STORAGE AND STABILITY",
    "SPECIAL HANDLING INSTRUCTIONS",
    "DOSAGE FORMS",
    "PHARMACEUTICAL INFORMATION"
}

# Subsection patterns for special populations and conditions
# These are often nested within main sections
SUBSECTION_PATTERNS = {
    'PREGNANCY': [
        'use in pregnancy',
        'pregnant women',
        'pregnancy and lactation',
        'teratogenic effects',
        'reproductive toxicity',
        'use during pregnancy'
    ],
    'GERIATRIC': [
        'geriatric use',
        'use in the elderly',  
        'elderly patients',
        'geriatrics',
        'use in elderly',
        'aged patients'
    ],
    'PEDIATRIC': [
        'pediatric use',
        'use in children',
        'pediatric patients',
        'paediatric use',
        'children and adolescents'
    ]
}

# Sections that need larger chunks to keep tables intact
LARGE_CHUNK_SECTIONS = {
    "DOSAGE AND ADMINISTRATION",
    "PHARMACOKINETICS",
    "DOSAGE FORMS"
}

# Configuration
DEFAULT_CHUNK_SIZE = 1200  # Conservative for better retrieval
LARGE_CHUNK_SIZE = 2000    # For tables/structured data
CHUNK_OVERLAP = 200


def is_section_header(line: str) -> Optional[str]:
    """
    Detect if a line is a section header.
    
    Rules:
    1. All uppercase AND in SECTION_HEADERS
    2. OR ends with colon and matches known header
    3. OR surrounded by  blank lines
    
    Args:
        line: Text line to check
        
    Returns:
        Section name if header detected, None otherwise
    """
    line = line.strip()
    
    if not line:
        return None
    
    # Rule 1: Exact match (all caps)
    if line.upper() in SECTION_HEADERS:
        return line.upper()
    
    # Rule 2: Ends with colon
    if line.endswith(':'):
        line_no_colon = line[:-1].strip().upper()
        if line_no_colon in SECTION_HEADERS:
            return line_no_colon
    
    # Rule 3: Fuzzy match (80% similarity)
    for known_header in SECTION_HEADERS:
        if known_header in line.upper() or line.upper() in known_header:
            return known_header
    
    return None


def detect_subsection(text: str) -> Optional[str]:
    """
    Detect subsections (pregnancy, geriatric, pediatric).
    
    These are often nested within main sections and use title case
    or mixed formatting rather than all caps.
    """
    text_lower = text.lower()
    
    # Check for each subsection type
    for subsection_type, patterns in SUBSECTION_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                return subsection_type
    
    return None


def detect_section(text: str) -> Optional[str]:
    """
    Detect current section from page text.
    
    Scans for section headers and returns the last one found.
    Also checks for subsections (pregnancy, geriatric, etc.).
    """
    lines = text.split('\n')
    current_section = None
    
    for line in lines:
        header = is_section_header(line)
        if header:
            current_section = header
    
    # Also check for subsections
    subsection = detect_subsection(text)
    if subsection:
        # Combine main section with subsection
        if current_section:
            return f"{current_section} > {subsection}"
        return subsection
    
    return current_section


def chunk_text(
    text: str,
    page_num: int,
    section_name: Optional[str] = None,
    default_chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict]:
    """
    Split text into chunks with adaptive sizing.
    
    Args:
        text: Page text to chunk
        page_num: Page number (0-indexed)
        section_name: Current section (for adaptive sizing)
        default_chunk_size: Default chunk size (1200)
        chunk_overlap: Overlap between chunks (200)
        
    Returns:
        List of chunk dictionaries with:
        - text: Chunk text
        - page_num: Page number
        - section_name: Section name (or None)
        - char_start: Character offset in page
        - char_end: End offset in page
    """
    # Adaptive chunk sizing based on section
    if section_name and section_name in LARGE_CHUNK_SECTIONS:
        chunk_size = LARGE_CHUNK_SIZE
    else:
        chunk_size = default_chunk_size
    
    # Create text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )
    
    # Split text
    chunks = splitter.split_text(text)
    
    # Build chunk metadata
    result = []
    char_offset = 0
    
    for chunk_text in chunks:
        # Detect section within chunk (may change mid-page)
        chunk_section = detect_section(chunk_text) or section_name
        
        result.append({
            'text': chunk_text,
            'page_num': page_num,
            'section_name': chunk_section,
            'char_start': char_offset,
            'char_end': char_offset + len(chunk_text)
        })
        
        char_offset += len(chunk_text) - chunk_overlap
    
    return result
