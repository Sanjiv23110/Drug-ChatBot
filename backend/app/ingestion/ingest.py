"""
Production Ingestion Pipeline for Medical Drug Monographs.

This module orchestrates the complete ingestion flow:
1. Parse PDF with docling → Markdown + images
2. Extract drug metadata (name) from first page via LLM
3. Split content by Markdown headers (no character-based splitting)
4. Store sections DYNAMICALLY (no hardcoded categories)
5. Classify chemical structure images with Vision
6. Store idempotently in PostgreSQL

Philosophy: Structure over Similarity - data is structured during ingestion
so retrieval can be done deterministically via SQL.

DESIGN: NO HARDCODED SECTIONS - sections are stored exactly as found,
with dynamic mapping table for normalization/grouping.
"""
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from openai import AzureOpenAI
from sqlalchemy import select

from app.db.models import (
    MonographSection,
    SectionMapping,
    ImageClassification,
    IngestionLog,
    DrugMetadata,
    clean_header,
    header_to_snake_case,
    get_or_create_section_mapping
)
from app.db.session import get_session
from app.ingestion.docling_utils import DoclingParser, ParsedDocument, ExtractedImage
from app.ingestion.vision import VisionClassifier
from app.ingestion.section_detector import SectionDetector, SectionCategory
from app.ingestion.layout_extractor import fallback_blocks_from_markdown
from app.utils.hashing import compute_file_hash

logger = logging.getLogger(__name__)


@dataclass
class ChunkedSection:
    """A section extracted from a drug monograph."""
    header: str  # Original header from PDF
    header_cleaned: str  # Cleaned/normalized header
    content: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    images: List[str] = field(default_factory=list)


@dataclass
class IngestionResult:
    """Result of ingesting a single PDF."""
    file_path: str
    file_name: str
    document_hash: str
    drug_name: str
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    sections_created: int = 0
    images_extracted: int = 0
    structures_detected: int = 0
    new_section_types: int = 0  # NEW: track new section discoveries
    success: bool = True
    error_message: Optional[str] = None
    processing_time_ms: int = 0


class DrugMetadataExtractor:
    """
    Extract drug name from the first page of a monograph using LLM.
    
    Uses a single lightweight LLM call on the first ~2000 characters.
    """
    
    EXTRACTION_PROMPT = """You are a pharmaceutical data extractor.

Analyze this text from the first page of a drug monograph and extract:
1. Brand Name (the trademarked commercial name)
2. Generic Name (the chemical/scientific name)

Text:
{text}

Respond in this exact format:
BRAND: [brand name or UNKNOWN]
GENERIC: [generic name or UNKNOWN]

Rules:
- If multiple brand names exist, use the primary one
- Normalize to lowercase
- Remove trademark symbols (®, ™)
- If a name is not found, use UNKNOWN"""

    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        )
        self.model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-agent")
    
    def extract(self, markdown_content: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Extract drug names from markdown content.
        
        Args:
            markdown_content: Full markdown from docling
            
        Returns:
            Tuple of (drug_name, brand_name, generic_name)
            drug_name is the primary identifier (brand or generic)
        """
        # Use first ~2000 characters (first page equivalent)
        first_page_text = markdown_content[:2000]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.EXTRACTION_PROMPT.format(text=first_page_text)
                    }
                ],
                max_tokens=100,
                temperature=0
            )
            
            result = response.choices[0].message.content
            
            # Parse response
            brand_name = None
            generic_name = None
            
            for line in result.strip().split('\n'):
                if line.startswith('BRAND:'):
                    value = line[6:].strip().lower()
                    if value and value != 'unknown':
                        brand_name = value
                elif line.startswith('GENERIC:'):
                    value = line[8:].strip().lower()
                    if value and value != 'unknown':
                        generic_name = value
            
            # Primary drug name (prefer generic, fallback to brand)
            drug_name = generic_name or brand_name or self._fallback_extraction(first_page_text)
            
            logger.info(f"Extracted drug: {drug_name} (brand: {brand_name}, generic: {generic_name})")
            
            return drug_name, brand_name, generic_name
            
        except Exception as e:
            logger.error(f"Drug metadata extraction failed: {e}")
            return self._fallback_extraction(first_page_text), None, None
    
    def _fallback_extraction(self, text: str) -> str:
        """
        Fallback extraction using regex patterns.
        
        Looks for common drug name patterns in monograph titles.
        """
        # Try to find capitalized drug names in common patterns
        patterns = [
            r'(?:product\s+monograph|prescribing\s+information)[\s-]*([A-Z][a-zA-Z]+)',
            r'^([A-Z][a-zA-Z]+)[\s®™]*\n',
            r'([A-Z][a-zA-Z]{3,})\s+tablets?|capsules?|injection',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).lower()
        
        # Last resort: use filename if available
        return "unknown_drug"


class HeaderBasedChunker:
    """
    Split markdown content by headers (not characters).
    
    Rules:
    - Split on # and ## headers
    - Keep all content under same header together
    - Tables MUST NOT be split
    - Sections are stored EXACTLY as found (dynamic, no hardcoding)
    """
    
    # Regex to match markdown headers
    HEADER_PATTERN = re.compile(r'^(#{1,2})\s+(.+)$', re.MULTILINE)
    
    def chunk(self, markdown_content: str) -> List[ChunkedSection]:
        """
        Split markdown into sections by headers.
        
        Args:
            markdown_content: Full markdown from docling
            
        Returns:
            List of ChunkedSection objects
        """
        sections = []
        
        # Find all headers
        headers = list(self.HEADER_PATTERN.finditer(markdown_content))
        
        if not headers:
            # No headers found - treat entire content as single section
            sections.append(ChunkedSection(
                header="DOCUMENT",
                header_cleaned=clean_header("DOCUMENT"),
                content=markdown_content.strip()
            ))
            return sections
        
        # Extract content between headers
        for i, match in enumerate(headers):
            header_level = len(match.group(1))
            header_text = match.group(2).strip()
            
            # Content starts after this header
            content_start = match.end()
            
            # Content ends at next header (or end of document)
            if i + 1 < len(headers):
                content_end = headers[i + 1].start()
            else:
                content_end = len(markdown_content)
            
            content = markdown_content[content_start:content_end].strip()
            
            # Skip empty sections
            if not content:
                continue
            
            # Clean header for storage (but keep original too)
            cleaned = clean_header(header_text)
            
            sections.append(ChunkedSection(
                header=header_text,
                header_cleaned=cleaned,
                content=content
            ))
        
        logger.info(f"Chunked into {len(sections)} sections")
        
        return sections


class IngestionPipeline:
    """
    Main orchestrator for PDF ingestion.
    
    Coordinates:
    - PDF parsing (docling)
    - Metadata extraction (LLM)
    - Header-based chunking
    - DYNAMIC section storage (no hardcoded categories)
    - Chemical structure detection (Vision)
    - Idempotent PostgreSQL storage
    """
    
    def __init__(
        self,
        image_output_dir: str = "./data/images",
        skip_vision: bool = False,
        skip_existing: bool = True
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            image_output_dir: Where to save extracted images
            skip_vision: Skip chemical structure classification
            skip_existing: Skip documents that already exist (by hash)
        """
        self.image_output_dir = Path(image_output_dir)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.skip_vision = skip_vision
        self.skip_existing = skip_existing
        
        # Initialize components
        self.parser = DoclingParser(
            image_output_dir=str(self.image_output_dir),
            extract_images=not skip_vision  # Don't extract images if skipping vision
        )
        self.metadata_extractor = DrugMetadataExtractor()
        self.chunker = HeaderBasedChunker()
        self.section_detector = SectionDetector(use_llm_fallback=True)  # NEW: Layout-aware detection
        self.vision_classifier = VisionClassifier() if not skip_vision else None
        
        logger.info("IngestionPipeline initialized with SectionDetector")
    
    async def document_exists(self, document_hash: str) -> bool:
        """Check if document already exists in database."""
        async with get_session() as session:
            result = await session.execute(
                select(MonographSection)
                .where(MonographSection.document_hash == document_hash)
                .limit(1)
            )
            return result.first() is not None
    
    async def ingest_pdf(self, pdf_path: str) -> IngestionResult:
        """
        Ingest a single PDF into the database.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            IngestionResult with statistics and status
        """
        start_time = datetime.utcnow()
        
        path = Path(pdf_path)
        logger.info(f"Starting ingestion: {path.name}")
        
        try:
            # Step 1: Parse PDF with docling
            parsed = self.parser.parse(pdf_path)
            
            if not parsed.parse_success:
                return IngestionResult(
                    file_path=pdf_path,
                    file_name=path.name,
                    document_hash=parsed.document_hash,
                    drug_name="",
                    success=False,
                    error_message=parsed.error_message
                )
            
            # Step 2: Check idempotency
            if self.skip_existing and await self.document_exists(parsed.document_hash):
                logger.info(f"Skipping existing document: {path.name}")
                return IngestionResult(
                    file_path=pdf_path,
                    file_name=path.name,
                    document_hash=parsed.document_hash,
                    drug_name="",
                    success=True,
                    error_message="Document already exists"
                )
            
            # Step 3: Extract drug metadata
            drug_name, brand_name, generic_name = self.metadata_extractor.extract(
                parsed.markdown_content
            )
            
            # Step 4: Chunk using SectionDetector (Deterministic 4-Layer Engine)
            # Create pseudo-blocks from markdown (hybrid approach)
            blocks = fallback_blocks_from_markdown(parsed.markdown_content)
            
            if blocks:
                # Use the new engine
                detected_boundaries = self.section_detector.detect_sections(blocks)
                sections = self._convert_to_chunked_sections(detected_boundaries, blocks)
                logger.info(f"SectionDetector findings: {len(sections)} sections")
            else:
                # Fallback to legacy chunker if something goes wrong
                logger.warning("Block extraction failed, falling back to legacy chunker")
                sections = self.chunker.chunk(parsed.markdown_content)
            
            # Step 4.5: Validation (Log-only, can be removed later)
            # self._validate_sections_with_detector(sections, parsed.markdown_content) 

            
            # Step 5: Classify chemical structure images
            structure_images = []
            if self.vision_classifier and parsed.images:
                for image in parsed.images:
                    is_structure, image_hash = self.vision_classifier.classify_image(
                        image.image_path
                    )
                    if is_structure:
                        structure_images.append(image)
                        
                        # Cache in DB
                        await self._cache_image_classification(
                            image.image_path, image_hash, True, drug_name
                        )
            
            # Step 6: Store sections in PostgreSQL (DYNAMIC - learns new sections)
            sections_created, new_section_types = await self._store_sections(
                sections=sections,
                parsed=parsed,
                drug_name=drug_name,
                brand_name=brand_name,
                generic_name=generic_name,
                structure_images=structure_images
            )
            
            # Step 7: Update drug metadata
            await self._update_drug_metadata(
                drug_name=drug_name,
                brand_name=brand_name,
                generic_name=generic_name,
                file_path=pdf_path,
                document_hash=parsed.document_hash,
                sections=sections,
                has_structure=len(structure_images) > 0
            )
            
            # Step 8: Log ingestion
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self._log_ingestion(
                file_path=pdf_path,
                document_hash=parsed.document_hash,
                status="success",
                sections_created=sections_created,
                images_extracted=len(parsed.images),
                new_section_types=new_section_types,
                processing_time_ms=processing_time
            )
            
            logger.info(
                f"Completed ingestion: {path.name} → "
                f"{sections_created} sections, {len(structure_images)} structures, "
                f"{new_section_types} new section types"
            )
            
            return IngestionResult(
                file_path=pdf_path,
                file_name=path.name,
                document_hash=parsed.document_hash,
                drug_name=drug_name,
                brand_name=brand_name,
                generic_name=generic_name,
                sections_created=sections_created,
                images_extracted=len(parsed.images),
                structures_detected=len(structure_images),
                new_section_types=new_section_types,
                success=True,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Ingestion failed for {path.name}: {e}")
            
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self._log_ingestion(
                file_path=pdf_path,
                document_hash=compute_file_hash(pdf_path) if path.exists() else "",
                status="failed",
                sections_created=0,
                images_extracted=0,
                new_section_types=0,
                processing_time_ms=processing_time,
                error_message=str(e)
            )
            
            return IngestionResult(
                file_path=pdf_path,
                file_name=path.name,
                document_hash="",
                drug_name="",
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time
            )
            
    def _validate_sections_with_detector(
        self,
        sections: List[ChunkedSection],
        markdown_content: str
    ):
        """
        Validate chunked sections using SectionDetector.
        
        This provides quality metrics and logs potential issues without
        requiring database schema changes.
        """
        try:
            # Create pseudo-blocks from markdown for section detection
            blocks = fallback_blocks_from_markdown(markdown_content)
            
            if not blocks:
                logger.warning("No blocks extracted for section validation")
                return
            
            # Run section detector
            detected_sections = self.section_detector.detect_sections(blocks)
            
            # Compare results
            logger.info(
                f"Section validation: "
                f"Markdown chunker found {len(sections)} sections, "
                f"SectionDetector found {len(detected_sections)} sections"
            )
            
            # Log detection method distribution
            method_counts = {}
            for section in detected_sections:
                method = section.detection_method
                method_counts[method] = method_counts.get(method, 0) + 1
            
            logger.info(f"Detection methods: {method_counts}")
            
        except Exception as e:
            logger.error(f"Section validation failed: {e}")
    
    async def _store_sections(
        self,
        sections: List[ChunkedSection],
        parsed: ParsedDocument,
        drug_name: str,
        brand_name: Optional[str],
        generic_name: Optional[str],
        structure_images: List[ExtractedImage]
    ) -> Tuple[int, int]:
        """
        Store sections in PostgreSQL.
        
        Returns:
            Tuple of (sections_created, new_section_types)
        """
        created_count = 0
        new_sections = 0
        
        # Keywords that likely indicate chemical structure section
        structure_keywords = ['structure', 'chemical', 'molecular', 'formula', 'description']
        
        async with get_session() as session:
            for i, section in enumerate(sections, 1):
                # Log progress
                logger.info(f"Processing section {i}/{len(sections)}: {section.header_cleaned}")
                
                # Register section in dynamic mapping table (UPSERT handles duplicates)
                mapping = await get_or_create_section_mapping(
                    session, section.header_cleaned, "auto"
                )
                
                # Track if this is a new section type
                if mapping.usage_count == 1:
                    new_sections += 1
                    logger.info(f"Discovered new section type: {section.header_cleaned}")
                
                # Determine if this section should have structure images
                section_images = []
                has_structure = False
                
                # Check if this section is likely about chemical structure
                header_lower = section.header_cleaned.lower()
                if any(kw in header_lower for kw in structure_keywords):
                    section_images = [img.image_path for img in structure_images]
                    has_structure = len(section_images) > 0
                
                # Create section record with DYNAMIC section name
                monograph_section = MonographSection(
                    drug_name=drug_name,
                    brand_name=brand_name,
                    generic_name=generic_name,
                    original_filename=parsed.file_name,
                    document_hash=parsed.document_hash,
                    section_name=section.header_cleaned,  # DYNAMIC - stored as-is
                    original_header=section.header,
                    content_text=section.content,
                    content_markdown=section.content,
                    image_paths=section_images,
                    has_chemical_structure=has_structure,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    char_count=len(section.content)
                )
                
                session.add(monograph_section)
                created_count += 1
            
            await session.commit()
        
        return created_count, new_sections
    
    async def _cache_image_classification(
        self,
        image_path: str,
        image_hash: str,
        is_structure: bool,
        drug_name: str
    ):
        """Cache image classification result in database."""
        async with get_session() as session:
            classification = ImageClassification(
                image_hash=image_hash,
                image_path=image_path,
                is_chemical_structure=is_structure,
                drug_name=drug_name if is_structure else None,
                model_used=os.getenv("VISION_MODEL", "gpt-4o")
            )
            session.add(classification)
            await session.commit()
    
    async def _update_drug_metadata(
        self,
        drug_name: str,
        brand_name: Optional[str],
        generic_name: Optional[str],
        file_path: str,
        document_hash: str,
        sections: List[ChunkedSection],
        has_structure: bool
    ):
        """Update or create drug metadata record."""
        async with get_session() as session:
            # Check if drug exists
            result = await session.execute(
                select(DrugMetadata).where(DrugMetadata.drug_name == drug_name)
            )
            existing = result.first()
            
            if existing:
                drug_meta = existing[0]
                # Update existing
                if file_path not in drug_meta.source_files:
                    drug_meta.source_files = drug_meta.source_files + [file_path]
                if brand_name and brand_name not in drug_meta.brand_names:
                    drug_meta.brand_names = drug_meta.brand_names + [brand_name]
                
                # Update available sections (DYNAMIC)
                section_names = [s.header_cleaned for s in sections]
                for name in section_names:
                    if name not in drug_meta.available_sections:
                        drug_meta.available_sections = drug_meta.available_sections + [name]
                
                drug_meta.total_sections += len(sections)
                drug_meta.has_structure_image = drug_meta.has_structure_image or has_structure
                drug_meta.last_updated = datetime.utcnow()
            else:
                # Create new
                drug_meta = DrugMetadata(
                    drug_name=drug_name,
                    brand_names=[brand_name] if brand_name else [],
                    generic_name=generic_name,
                    source_files=[file_path],
                    primary_document_hash=document_hash,
                    available_sections=[s.header_cleaned for s in sections],  # DYNAMIC
                    has_structure_image=has_structure,
                    total_sections=len(sections)
                )
                session.add(drug_meta)
            
            await session.commit()
    
    async def _log_ingestion(
        self,
        file_path: str,
        document_hash: str,
        status: str,
        sections_created: int,
        images_extracted: int,
        new_section_types: int,
        processing_time_ms: int,
        error_message: Optional[str] = None
    ):
        """Log ingestion to database."""
        async with get_session() as session:
            log = IngestionLog(
                file_path=file_path,
                document_hash=document_hash,
                status=status,
                error_message=error_message,
                sections_created=sections_created,
                images_extracted=images_extracted,
                new_section_types=new_section_types,
                processing_time_ms=processing_time_ms,
                completed_at=datetime.utcnow()
            )
            session.add(log)
            await session.commit()
    
    async def ingest_directory(
        self,
        pdf_dir: str,
        batch_size: int = 10,
        continue_on_error: bool = True
    ) -> List[IngestionResult]:
        """
        Ingest all PDFs in a directory.
        
        Args:
            pdf_dir: Directory containing PDFs
            batch_size: Number of PDFs to process in parallel
            continue_on_error: Continue if one PDF fails
            
        Returns:
            List of IngestionResult for all files
        """
        pdf_path = Path(pdf_dir)
        pdf_files = list(pdf_path.glob("*.pdf")) + list(pdf_path.glob("*.PDF"))
        
        logger.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")
        
        results = []
        for pdf_file in pdf_files:
            try:
                result = await self.ingest_pdf(str(pdf_file))
                results.append(result)
            except Exception as e:
                if continue_on_error:
                    logger.error(f"Skipping {pdf_file.name}: {e}")
                    results.append(IngestionResult(
                        file_path=str(pdf_file),
                        file_name=pdf_file.name,
                        document_hash="",
                        drug_name="",
                        success=False,
                        error_message=str(e)
                    ))
                else:
                    raise
        
        # Summary
        success_count = sum(1 for r in results if r.success)
        total_sections = sum(r.sections_created for r in results)
        total_structures = sum(r.structures_detected for r in results)
        total_new_sections = sum(r.new_section_types for r in results)
        
        logger.info(
            f"Ingestion complete: {success_count}/{len(results)} successful, "
            f"{total_sections} sections, {total_structures} structures, "
            f"{total_new_sections} new section types discovered"
        )
        
        return results


# Convenience function for CLI usage
async def ingest_pdfs(
    pdf_dir: str,
    image_dir: str = "./data/images",
    skip_vision: bool = False
) -> List[IngestionResult]:
    """
    Convenience function to ingest PDFs.
    
    Args:
        pdf_dir: Directory containing PDFs
        image_dir: Where to save extracted images
        skip_vision: Skip chemical structure classification
        
    Returns:
        List of IngestionResult
    """
    pipeline = IngestionPipeline(
        image_output_dir=image_dir,
        skip_vision=skip_vision
    )
    return await pipeline.ingest_directory(pdf_dir)
