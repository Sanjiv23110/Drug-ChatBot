"""
Dual-Chunking Strategy for Regulatory QA System
- Semantic chunks: for retrieval (enriched with metadata)
- Raw narrative blocks: for display (immutable ground truth)
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import hashlib
from spl_xml_parser import SPLMetadata, SPLSection

@dataclass
class ChunkMetadata:
    """Metadata attached to each chunk"""
    drug_name: str
    rxcui: Optional[str]
    set_id: str
    root_id: str
    version: str
    effective_date: str
    loinc_code: str
    loinc_section: str
    is_table: bool
    ndc: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class DocumentChunk:
    """
    Represents a dual-format chunk
    """
    chunk_id: str
    semantic_text: str  # For embedding/retrieval (can be enriched)
    raw_text: str  # For display (IMMUTABLE - verbatim from source)
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict:
        return {
            'chunk_id': self.chunk_id,
            'semantic_text': self.semantic_text,
            'raw_text': self.raw_text,
            'metadata': self.metadata.to_dict()
        }


class DualChunker:
    """
    Creates semantic + raw chunks with proper overlap
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """
        Args:
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size (avoid tiny chunks)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_document(
        self,
        metadata: SPLMetadata,
        sections: List[SPLSection]
    ) -> List[DocumentChunk]:
        """
        Process entire SPL document into chunks
        
        Returns:
            List of DocumentChunk objects
        """
        all_chunks = []
        
        for section in sections:
            section_chunks = self._chunk_section(metadata, section)
            all_chunks.extend(section_chunks)
        
        return all_chunks
    
    def _chunk_section(
        self,
        metadata: SPLMetadata,
        section: SPLSection
    ) -> List[DocumentChunk]:
        """Chunk individual section"""
        
        chunks = []
        text = section.text_content
        
        # Special handling for tables - don't split
        if section.is_table:
            chunk = self._create_chunk(
                metadata=metadata,
                section=section,
                text=text,
                chunk_index=0
            )
            chunks.append(chunk)
            return chunks
        
        # Regular text chunking with overlap
        words = text.split()
        
        if len(words) <= self.chunk_size:
            # Section fits in one chunk
            chunk = self._create_chunk(
                metadata=metadata,
                section=section,
                text=text,
                chunk_index=0
            )
            chunks.append(chunk)
        else:
            # Split into overlapping chunks
            chunk_index = 0
            start = 0
            
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk_words = words[start:end]
                
                # Skip if too small (unless it's the last chunk)
                if len(chunk_words) < self.min_chunk_size and end < len(words):
                    start += self.chunk_size - self.overlap
                    continue
                
                chunk_text = ' '.join(chunk_words)
                
                chunk = self._create_chunk(
                    metadata=metadata,
                    section=section,
                    text=chunk_text,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                
                chunk_index += 1
                start += self.chunk_size - self.overlap
        
        return chunks
    
    def _create_chunk(
        self,
        metadata: SPLMetadata,
        section: SPLSection,
        text: str,
        chunk_index: int
    ) -> DocumentChunk:
        """Create a DocumentChunk with both semantic and raw text"""
        
        # Generate deterministic chunk ID
        chunk_id = self._generate_chunk_id(
            metadata.drug_name,
            metadata.version_number,
            section.loinc_code,
            chunk_index
        )
        
        # Semantic text: enriched for retrieval
        semantic_text = self._enrich_for_retrieval(text, metadata, section)
        
        # Raw text: verbatim (IMMUTABLE)
        raw_text = text
        
        # Metadata
        chunk_metadata = ChunkMetadata(
            drug_name=metadata.drug_name,
            rxcui=metadata.rxcui,
            set_id=metadata.set_id,
            root_id=metadata.root_id,
            version=metadata.version_number,
            effective_date=metadata.effective_time,
            loinc_code=section.loinc_code,
            loinc_section=section.section_name,
            is_table=section.is_table,
            ndc=metadata.ndc_codes
        )
        
        return DocumentChunk(
            chunk_id=chunk_id,
            semantic_text=semantic_text,
            raw_text=raw_text,
            metadata=chunk_metadata
        )
    
    def _enrich_for_retrieval(
        self,
        text: str,
        metadata: SPLMetadata,
        section: SPLSection
    ) -> str:
        """
        Enrich text with context for better retrieval
        This is ONLY for embedding, NOT for display
        """
        prefix = f"Drug: {metadata.drug_name}. Section: {section.section_name}. "
        return prefix + text
    
    def _generate_chunk_id(
        self,
        drug_name: str,
        version: str,
        loinc_code: str,
        chunk_index: int
    ) -> str:
        """Generate deterministic chunk ID"""
        # Normalize drug name
        drug_normalized = drug_name.upper().replace(' ', '_')
        
        # Format: DRUGNAME_vVERSION_LOINC_chunkINDEX
        chunk_id = f"{drug_normalized}_v{version}_{loinc_code}_chunk_{chunk_index:03d}"
        
        return chunk_id


class SemanticChunkEnricher:
    """
    Optional: Add drug class information to semantic chunks
    Requires RxClass integration
    """
    
    def __init__(self, rxclass_client=None):
        self.rxclass_client = rxclass_client
    
    def enrich_with_drug_classes(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        Add pharmacologic/therapeutic classes to semantic text
        """
        if not self.rxclass_client or not chunk.metadata.rxcui:
            return chunk
        
        # Get drug classes from RxClass
        classes = self.rxclass_client.get_classes(chunk.metadata.rxcui)
        
        if classes:
            class_text = ", ".join([c['class_name'] for c in classes])
            chunk.semantic_text = f"Drug classes: {class_text}. " + chunk.semantic_text
        
        return chunk


# Example usage
if __name__ == "__main__":
    from spl_xml_parser import SPLXMLParser
    
    # Parse document
    parser = SPLXMLParser()
    metadata, sections = parser.parse_document('lisinopril_spl.xml')
    
    # Chunk document
    chunker = DualChunker(chunk_size=512, overlap=50)
    chunks = chunker.chunk_document(metadata, sections)
    
    print(f"Generated {len(chunks)} chunks")
    
    # Inspect first chunk
    first_chunk = chunks[0]
    print(f"\nChunk ID: {first_chunk.chunk_id}")
    print(f"Semantic text (first 200 chars): {first_chunk.semantic_text[:200]}")
    print(f"Raw text (first 200 chars): {first_chunk.raw_text[:200]}")
    print(f"Metadata: {first_chunk.metadata.to_dict()}")
