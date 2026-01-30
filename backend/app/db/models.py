"""
PostgreSQL Database Models with pgvector support.

This module defines the SQLModel schema for the medical drug monograph system.
Uses SQLModel for async PostgreSQL operations with pgvector extension.

DESIGN PRINCIPLE: NO HARDCODED SECTIONS
- Sections are stored exactly as they appear in PDFs
- Dynamic mapping table learns section relationships over time
- LLM suggests mappings for new sections
- All mappings are editable and auditable
"""
import re
from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON, Text, Index, UniqueConstraint
from pgvector.sqlalchemy import Vector


class SectionMapping(SQLModel, table=True):
    """
    Dynamic section mapping table.
    
    Maps raw PDF headers to normalized section names.
    This replaces hardcoded enums - fully dynamic and learnable.
    
    Example:
        original_header: "ADVERSE REACTIONS"
        normalized_name: "adverse_reactions"
        display_name: "Adverse Reactions"
        suggested_by: "llm" or "rule" or "admin"
    """
    __tablename__ = "section_mappings"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Original header (lowercase, cleaned)
    original_header: str = Field(unique=True, index=True, max_length=512)
    
    # Normalized section name (snake_case, for SQL queries)
    normalized_name: str = Field(index=True, max_length=255)
    
    # Display name (Title Case, for UI)
    display_name: str = Field(max_length=255)
    
    # How this mapping was created
    suggested_by: str = Field(max_length=50)  # "llm", "rule", "admin", "auto"
    
    # Confidence score (1.0 = confirmed by admin, <1.0 = auto-suggested)
    confidence: float = Field(default=0.8)
    
    # Is this a common/important section? (for prioritization)
    is_common: bool = Field(default=False)
    
    # Usage count (how many times this header was seen)
    usage_count: int = Field(default=1)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Optional: group similar sections (e.g., all dosage-related)
    section_group: Optional[str] = Field(default=None, max_length=100)


class MonographSection(SQLModel, table=True):
    """
    Main table for storing drug monograph sections.
    
    Each row represents one section (chunk) from a drug monograph PDF.
    Designed for SQL-first retrieval with optional vector fallback.
    
    IMPORTANT: section_name stores the EXACT header from PDF (cleaned).
    Use section_mappings table for normalization/grouping.
    """
    __tablename__ = "monograph_sections"
    
    # Primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Drug identification (normalized to lowercase)
    drug_name: str = Field(index=True, max_length=255)
    brand_name: Optional[str] = Field(default=None, max_length=255)
    generic_name: Optional[str] = Field(default=None, max_length=255)
    
    # Document tracking
    original_filename: str = Field(max_length=512)
    document_hash: str = Field(index=True, max_length=64)  # SHA-256 hex
    
    # Section identification
    # DYNAMIC: stores exact header from PDF (cleaned/lowercased)
    section_name: str = Field(index=True, max_length=512)
    # Original header as it appeared in PDF (preserves case)
    original_header: Optional[str] = Field(default=None, max_length=512)
    
    # Content
    content_text: str = Field(sa_column=Column(Text))
    content_markdown: Optional[str] = Field(default=None, sa_column=Column(Text))
    
    # Images (JSON list of paths)
    image_paths: List[str] = Field(default=[], sa_column=Column(JSON))
    has_chemical_structure: bool = Field(default=False)
    
    # Vector embedding (optional - for semantic fallback)
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(Vector(1536))
    )
    
    # Metadata
    page_start: Optional[int] = Field(default=None)
    page_end: Optional[int] = Field(default=None)
    char_count: Optional[int] = Field(default=None)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Table-level constraints
    __table_args__ = (
        # Prevent duplicate sections for same document
        UniqueConstraint('document_hash', 'section_name', 'page_start', 
                        name='uq_document_section'),
        # Indexes for fast lookups
        Index('ix_drug_section', 'drug_name', 'section_name'),
        Index('ix_document_lookup', 'document_hash'),
        # Full-text search index (created via migration)
        # GIN index on section_name for pg_trgm
    )


class ImageClassification(SQLModel, table=True):
    """
    Cache for vision model classifications.
    
    Stores SHA-256 hash → classification to avoid repeated API calls.
    """
    __tablename__ = "image_classifications"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Image identification
    image_hash: str = Field(unique=True, index=True, max_length=64)  # SHA-256
    image_path: str = Field(max_length=1024)
    
    # Classification result
    is_chemical_structure: bool = Field(default=False)
    confidence: Optional[float] = Field(default=None)
    
    # Linked drug (if classified as structure)
    drug_name: Optional[str] = Field(default=None, max_length=255)
    
    # Metadata
    classified_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str = Field(default="gpt-4o", max_length=50)


class IngestionLog(SQLModel, table=True):
    """
    Audit log for ingestion operations.
    
    Tracks which files were processed, when, and with what result.
    """
    __tablename__ = "ingestion_logs"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # File tracking
    file_path: str = Field(max_length=1024)
    document_hash: str = Field(index=True, max_length=64)
    
    # Status
    status: str = Field(max_length=50)  # "success", "failed", "skipped"
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text))
    
    # Statistics
    sections_created: int = Field(default=0)
    images_extracted: int = Field(default=0)
    new_section_types: int = Field(default=0)  # NEW: track new section discoveries
    processing_time_ms: Optional[int] = Field(default=None)
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)


class DrugMetadata(SQLModel, table=True):
    """
    Master drug table for quick lookups.
    
    One row per unique drug, with aggregated metadata.
    """
    __tablename__ = "drug_metadata"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Drug identification
    drug_name: str = Field(unique=True, index=True, max_length=255)
    brand_names: List[str] = Field(default=[], sa_column=Column(JSON))
    generic_name: Optional[str] = Field(default=None, max_length=255)
    
    # Source tracking
    source_files: List[str] = Field(default=[], sa_column=Column(JSON))
    primary_document_hash: Optional[str] = Field(default=None, max_length=64)
    
    # Available sections (DYNAMIC: list of section_names found)
    available_sections: List[str] = Field(default=[], sa_column=Column(JSON))
    has_structure_image: bool = Field(default=False)
    
    # Statistics
    total_sections: int = Field(default=0)
    total_images: int = Field(default=0)
    
    # Timestamps
    first_ingested: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# DYNAMIC SECTION NORMALIZATION
# =============================================================================

def clean_header(header: str) -> str:
    """
    Clean and normalize a header string.
    
    Args:
        header: Raw header from PDF
        
    Returns:
        Cleaned, lowercase header
    """
    # Lowercase
    cleaned = header.lower().strip()
    
    # Remove common prefixes
    for prefix in ["section ", "part ", "chapter ", "§ "]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    
    # Remove numbering (e.g., "1. Indications" → "indications")
    cleaned = re.sub(r'^[\d.]+\s*', '', cleaned)
    
    # Remove special characters but keep spaces
    cleaned = re.sub(r'[^\w\s-]', '', cleaned)
    
    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned.strip()


def header_to_snake_case(header: str) -> str:
    """
    Convert a header to snake_case for database queries.
    
    Args:
        header: Cleaned header string
        
    Returns:
        snake_case version
        
    Example:
        "adverse reactions" → "adverse_reactions"
        "Dosage and Administration" → "dosage_and_administration"
    """
    cleaned = clean_header(header)
    # Replace spaces and hyphens with underscores
    return re.sub(r'[\s-]+', '_', cleaned)


def header_to_display(header: str) -> str:
    """
    Convert a header to Title Case for display.
    
    Args:
        header: Raw or cleaned header
        
    Returns:
        Title Case version
    """
    cleaned = clean_header(header)
    return cleaned.title()


async def get_or_create_section_mapping(
    session,
    original_header: str,
    suggested_by: str = "auto"
) -> SectionMapping:
    """
    Get existing mapping or create a new one using PostgreSQL UPSERT.
    
    This uses PostgreSQL's INSERT ... ON CONFLICT DO UPDATE for atomic operations.
    This is production-grade, concurrent-safe, and scales infinitely.
    
    Args:
        session: Database session
        original_header: Raw header from PDF
        suggested_by: Who suggested this mapping
        
    Returns:
        SectionMapping object
    """
    from sqlalchemy.dialects.postgresql import insert
    from sqlalchemy import select
    
    cleaned = clean_header(original_header)
    
    # Prepare values for insert
    values = {
        "original_header": cleaned,
        "normalized_name": header_to_snake_case(original_header),
        "display_name": header_to_display(original_header),
        "suggested_by": suggested_by,
        "confidence": 0.8 if suggested_by == "auto" else 1.0,
        "is_common": False,
        "usage_count": 1,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    
    # PostgreSQL UPSERT: INSERT ... ON CONFLICT DO UPDATE
    # This is atomic - no race conditions even with 1000 concurrent workers
    stmt = insert(SectionMapping).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["original_header"],  # Unique constraint column
        set_={
            "usage_count": SectionMapping.usage_count + 1,
            "updated_at": datetime.utcnow()
        }
    ).returning(SectionMapping)
    
    # Execute and return the mapping
    result = await session.execute(stmt)
    mapping = result.scalar_one()
    
    return mapping


async def find_similar_sections(
    session,
    query: str,
    limit: int = 10
) -> List[SectionMapping]:
    """
    Find sections similar to a query using pg_trgm.
    
    Args:
        session: Database session
        query: Search term
        limit: Max results
        
    Returns:
        List of similar SectionMapping objects
    """
    from sqlalchemy import text
    
    result = await session.execute(
        text("""
            SELECT *, similarity(original_header, :query) as sim
            FROM section_mappings
            WHERE similarity(original_header, :query) > 0.3
            ORDER BY sim DESC
            LIMIT :limit
        """),
        {"query": query.lower(), "limit": limit}
    )
    
    return result.fetchall()


async def get_all_section_names(session) -> List[str]:
    """
    Get all unique section names in the database.
    
    Returns:
        List of section names
    """
    from sqlalchemy import select
    
    result = await session.execute(
        select(SectionMapping.normalized_name).distinct()
    )
    return [row[0] for row in result.fetchall()]


async def get_section_stats(session) -> dict:
    """
    Get statistics about sections in the database.
    
    Returns:
        Dict with section statistics
    """
    from sqlalchemy import text
    
    result = await session.execute(
        text("""
            SELECT 
                normalized_name,
                COUNT(*) as count,
                SUM(usage_count) as total_usage
            FROM section_mappings
            GROUP BY normalized_name
            ORDER BY total_usage DESC
        """)
    )
    
    return {
        row.normalized_name: {
            "count": row.count,
            "total_usage": row.total_usage
        }
        for row in result.fetchall()
    }
