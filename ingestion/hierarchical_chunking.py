"""
HIERARCHICAL PARENT-CHILD CHUNKING (MANDATORY)
This is NOT optional. This is the CORE requirement for WORD-TO-WORD accuracy.

CHILD CHUNKS: Individual sentences - used ONLY for search indexing
PARENT CHUNKS: Full paragraphs/subsections - SOURCE OF TRUTH for display

Every child MUST reference its parent chunk ID.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
import re
from ingestion.spl_xml_parser import SPLMetadata, SPLSection

@dataclass
class ParentChunk:
    """
    PARENT CHUNK: Full paragraph or XML subsection
    This is the SOURCE OF TRUTH - displayed to user
    IMMUTABLE - never modified after creation
    """
    parent_id: str
    raw_text: str  # VERBATIM from XML - IMMUTABLE
    loinc_code: str
    loinc_section: str
    drug_name: str
    rxcui: Optional[str]
    set_id: str
    root_id: str
    version: str
    effective_date: str
    is_table: bool
    ndc: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ChildChunk:
    """
    CHILD CHUNK: Individual sentence
    Used ONLY for search indexing
    MUST reference parent_id
    """
    child_id: str
    sentence_text: str  # Single sentence for indexing
    parent_id: str  # MANDATORY reference to parent
    loinc_code: str
    loinc_section: str
    drug_name: str
    rxcui: Optional[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class HierarchicalChunker:
    """
    MANDATORY HIERARCHICAL CHUNKING IMPLEMENTATION
    
    Creates parent-child structure:
    - Parents: Full paragraphs (SOURCE OF TRUTH)
    - Children: Individual sentences (SEARCH INDEX ONLY)
    
    This is NON-NEGOTIABLE for regulatory compliance.
    """
    
    def __init__(self):
        """Initialize hierarchical chunker"""
        self.sentence_splitter = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    def chunk_document(
        self,
        metadata: SPLMetadata,
        sections: List[SPLSection]
    ) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        """
        Process entire SPL document into hierarchical chunks
        
        Returns:
            (parent_chunks, child_chunks)
        """
        all_parents = []
        all_children = []
        
        for sec_idx, section in enumerate(sections):
            parents, children = self._chunk_section(metadata, section, sec_idx)
            all_parents.extend(parents)
            all_children.extend(children)
        
        return all_parents, all_children
    
    def _chunk_section(
        self,
        metadata: SPLMetadata,
        section: SPLSection,
        section_index: int
    ) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        """
        Chunk individual section into parent-child hierarchy
        """
        parents = []
        children = []
        
        # For tables: entire table is ONE parent, no children
        if section.is_table:
            parent = self._create_parent_chunk(
                metadata=metadata,
                section=section,
                text=section.text_content,
                parent_index=0,
                section_index=section_index
            )
            parents.append(parent)
            # Tables have NO child chunks - they are atomic
            return parents, children
        
        # For regular text: split into paragraphs
        paragraphs = self._split_into_paragraphs(section.text_content)
        
        for para_idx, paragraph in enumerate(paragraphs):
            # Create parent chunk (full paragraph)
            parent = self._create_parent_chunk(
                metadata=metadata,
                section=section,
                text=paragraph,
                parent_index=para_idx,
                section_index=section_index
            )
            parents.append(parent)
            
            # Create child chunks (individual sentences)
            sentences = self._split_into_sentences(paragraph)
            
            for sent_idx, sentence in enumerate(sentences):
                child = self._create_child_chunk(
                    metadata=metadata,
                    section=section,
                    sentence=sentence,
                    parent_id=parent.parent_id,
                    sentence_index=sent_idx
                )
                children.append(child)
        
        return parents, children
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs
        Paragraphs separated by double newlines or section breaks
        """
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean and filter
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If no paragraph breaks, treat entire text as one paragraph
        if not paragraphs:
            paragraphs = [text.strip()]
        
        return paragraphs
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split paragraph into individual sentences
        Used for child chunk creation
        """
        # Simple sentence splitting (can be improved with spaCy)
        sentences = self.sentence_splitter.split(text)
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Minimum sentence length (avoid fragments)
        sentences = [s for s in sentences if len(s) > 10]
        
        return sentences
    
    def _create_parent_chunk(
        self,
        metadata: SPLMetadata,
        section: SPLSection,
        text: str,
        parent_index: int,
        section_index: int
    ) -> ParentChunk:
        """
        Create PARENT chunk (SOURCE OF TRUTH)
        """
        # Generate deterministic parent ID
        parent_id = self._generate_parent_id(
            metadata.drug_name,
            metadata.version_number,
            section.loinc_code,
            parent_index,
            section_index
        )
        
        return ParentChunk(
            parent_id=parent_id,
            raw_text=text,  # IMMUTABLE
            loinc_code=section.loinc_code,
            loinc_section=section.section_name,
            drug_name=metadata.drug_name,
            rxcui=metadata.rxcui,
            set_id=metadata.set_id,
            root_id=metadata.root_id,
            version=metadata.version_number,
            effective_date=metadata.effective_time,
            is_table=section.is_table,
            ndc=metadata.ndc_codes
        )
    
    def _create_child_chunk(
        self,
        metadata: SPLMetadata,
        section: SPLSection,
        sentence: str,
        parent_id: str,
        sentence_index: int
    ) -> ChildChunk:
        """
        Create CHILD chunk (SEARCH INDEX ONLY)
        MUST reference parent_id
        """
        # Generate deterministic child ID
        child_id = f"{parent_id}_sent_{sentence_index:03d}"
        
        return ChildChunk(
            child_id=child_id,
            sentence_text=f"Drug: {metadata.drug_name}. Section: {section.section_name}. {sentence}",
            parent_id=parent_id,  # MANDATORY REFERENCE
            loinc_code=section.loinc_code,
            loinc_section=section.section_name,
            drug_name=metadata.drug_name,
            rxcui=metadata.rxcui
        )
    
    def _generate_parent_id(
        self,
        drug_name: str,
        version: str,
        loinc_code: str,
        parent_index: int,
        section_index: int
    ) -> str:
        """Generate deterministic parent chunk ID"""
        drug_normalized = drug_name.upper().replace(' ', '_')
        parent_id = f"{drug_normalized}_v{version}_{loinc_code}_sec_{section_index:03d}_para_{parent_index:03d}"
        return parent_id


# Example usage
if __name__ == "__main__":
    from spl_xml_parser import SPLXMLParser
    
    # Parse document
    parser = SPLXMLParser()
    metadata, sections = parser.parse_document('lisinopril_spl.xml')
    
    # Create hierarchical chunks
    chunker = HierarchicalChunker()
    parents, children = chunker.chunk_document(metadata, sections)
    
    print(f"Created {len(parents)} parent chunks (SOURCE OF TRUTH)")
    print(f"Created {len(children)} child chunks (SEARCH INDEX)")
    
    # Verify parent-child relationship
    if children:
        first_child = children[0]
        print(f"\nChild ID: {first_child.child_id}")
        print(f"Parent ID: {first_child.parent_id}")
        print(f"Sentence: {first_child.sentence_text[:100]}...")
        
        # Find parent
        parent = next((p for p in parents if p.parent_id == first_child.parent_id), None)
        if parent:
            print(f"\nParent text (first 200 chars):")
            print(parent.raw_text[:200])
