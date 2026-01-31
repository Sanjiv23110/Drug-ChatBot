"""
FactSpan Model - Granular Fact Indexing for Maximal Recall

This module extends the database schema to support sub-section retrieval.
Enables precise fact extraction from sentences, bullets, tables, and captions.

CRITICAL CONSTRAINT: All text is stored VERBATIM - no normalization, no paraphrasing.
"""
from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import Text, Index, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import TSVECTOR


class FactSpan(SQLModel, table=True):
    """
    Atomic fact units extracted from drug monograph sections.
    
    Each row represents one retrievable fact unit (sentence, bullet, table row, etc.)
    stored exactly as it appears in the source PDF.
    
    DESIGN PRINCIPLES:
    - Text is VERBATIM - exact copy from PDF, character-for-character
    - No summarization, no paraphrasing, no normalization
    - Enables sub-section retrieval for precise answers
    - Supports BM25-style full-text search via PostgreSQL ts_rank
    
    Example Usage:
        Query: "What is the half-life of AXID?"
        Retrieved Fact Span: "The elimination half-life is 1-2 hours." (sentence type)
        
        Query: "What dosage forms are available?"
        Retrieved Fact Spans: Multiple table_row entries from dosage forms table
    """
    __tablename__ = "fact_spans"
    
    # Primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Drug linkage (denormalized for fast lookups)
    drug_name: str = Field(index=True, max_length=255)
    brand_name: Optional[str] = Field(default=None, max_length=255)
    generic_name: Optional[str] = Field(default=None, max_length=255)
    
    # Source tracking
    section_id: int = Field(foreign_key="monograph_sections.id", ondelete="CASCADE")
    section_name: str = Field(index=True, max_length=512)  # Denormalized for fast filtering
    original_header: Optional[str] = Field(default=None, max_length=512)
    
    # Content (VERBATIM - NO MODIFICATION)
    text: str = Field(sa_column=Column(Text))  # Exact text from PDF
    text_type: str = Field(max_length=50)  # 'sentence' | 'bullet' | 'table_row' | 'caption'
    
    # Position metadata
    page_number: Optional[int] = Field(default=None)
    char_offset: Optional[int] = Field(default=None)  # Position within section content_text
    sequence_num: int = Field(default=0)  # Order within section
    
    # Document tracking
    document_hash: str = Field(index=True, max_length=64)  # SHA-256 hex
    original_filename: Optional[str] = Field(default=None, max_length=512)
    
    # Full-text search vector (for BM25-style ranking)
    # NOTE: This is populated via trigger or explicit update, not via SQLModel default
    search_vector: Optional[str] = Field(
        default=None,
        sa_column=Column(
            "search_vector",
            TSVECTOR,
            server_default=text("to_tsvector('english', text)")
        )
    )
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Table-level constraints
    __table_args__ = (
        # Prevent duplicate fact spans for same section
        UniqueConstraint('section_id', 'sequence_num', 'text_type', 
                        name='uq_section_sequence'),
        
        # Composite index for fast drug + section queries
        Index('ix_factspan_drug_section', 'drug_name', 'section_name'),
        
        # Index for text type filtering
        Index('ix_factspan_type', 'text_type'),
        
        # GIN index for full-text search
        Index('ix_factspan_search', 'search_vector', postgresql_using='gin'),
        
        # Compound index for BM25 queries
        Index('ix_factspan_compound', 'drug_name', 'section_name', 'text_type'),
    )


class FactSpanStats(SQLModel, table=True):
    """
    Statistics table for fact span coverage and quality monitoring.
    
    Tracks extraction metrics per drug/section to ensure comprehensive indexing.
    """
    __tablename__ = "fact_span_stats"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    drug_name: str = Field(index=True, max_length=255)
    section_name: str = Field(max_length=512)
    
    # Extraction counts by type
    total_sentences: int = Field(default=0)
    total_bullets: int = Field(default=0)
    total_table_rows: int = Field(default=0)
    total_captions: int = Field(default=0)
    
    # Quality metrics
    avg_sentence_length: Optional[float] = Field(default=None)
    coverage_percent: Optional[float] = Field(default=None)  # % of section text indexed
    
    # Timestamps
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('drug_name', 'section_name', name='uq_drug_section_stats'),
    )


# ============================================================================
# SQL MIGRATION SCRIPT (for reference)
# ============================================================================

FACT_SPAN_MIGRATION_SQL = """
-- Migration: Create fact_spans table and indexes
-- Version: 1.0
-- Date: 2026-01-31

BEGIN;

-- 1. Create fact_spans table
CREATE TABLE IF NOT EXISTS fact_spans (
    id SERIAL PRIMARY KEY,
    
    -- Drug linkage
    drug_name VARCHAR(255) NOT NULL,
    brand_name VARCHAR(255),
    generic_name VARCHAR(255),
    
    -- Source tracking
    section_id INTEGER NOT NULL REFERENCES monograph_sections(id) ON DELETE CASCADE,
    section_name VARCHAR(512) NOT NULL,
    original_header VARCHAR(512),
    
    -- Content (VERBATIM)
    text TEXT NOT NULL,
    text_type VARCHAR(50) NOT NULL,
    
    -- Position metadata
    page_number INTEGER,
    char_offset INTEGER,
    sequence_num INTEGER NOT NULL DEFAULT 0,
    
    -- Document tracking
    document_hash VARCHAR(64) NOT NULL,
    original_filename VARCHAR(512),
    
    -- Full-text search
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uq_section_sequence UNIQUE (section_id, sequence_num, text_type)
);

-- 2. Create indexes
CREATE INDEX IF NOT EXISTS ix_factspan_drug ON fact_spans(drug_name);
CREATE INDEX IF NOT EXISTS ix_factspan_section ON fact_spans(section_id);
CREATE INDEX IF NOT EXISTS ix_factspan_type ON fact_spans(text_type);
CREATE INDEX IF NOT EXISTS ix_factspan_search ON fact_spans USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS ix_factspan_drug_section ON fact_spans(drug_name, section_name);
CREATE INDEX IF NOT EXISTS ix_factspan_compound ON fact_spans(drug_name, section_name, text_type);

-- 3. Create stats table
CREATE TABLE IF NOT EXISTS fact_span_stats (
    id SERIAL PRIMARY KEY,
    drug_name VARCHAR(255) NOT NULL,
    section_name VARCHAR(512) NOT NULL,
    total_sentences INTEGER DEFAULT 0,
    total_bullets INTEGER DEFAULT 0,
    total_table_rows INTEGER DEFAULT 0,
    total_captions INTEGER DEFAULT 0,
    avg_sentence_length FLOAT,
    coverage_percent FLOAT,
    last_updated TIMESTAMP DEFAULT NOW(),
    CONSTRAINT uq_drug_section_stats UNIQUE (drug_name, section_name)
);

-- 4. Create helper function for BM25-style ranking
CREATE OR REPLACE FUNCTION fact_span_search(
    p_drug_name VARCHAR,
    p_search_terms TEXT,
    p_limit INTEGER DEFAULT 20
)
RETURNS TABLE (
    span_id INTEGER,
    span_text TEXT,
    span_type VARCHAR,
    section_name VARCHAR,
    page_number INTEGER,
    rank_score REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fs.id,
        fs.text,
        fs.text_type,
        fs.section_name,
        fs.page_number,
        ts_rank(fs.search_vector, to_tsquery('english', p_search_terms)) AS rank_score
    FROM fact_spans fs
    WHERE 
        fs.drug_name = p_drug_name
        AND fs.search_vector @@ to_tsquery('english', p_search_terms)
    ORDER BY rank_score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- 5. Create trigger to update stats on insert
CREATE OR REPLACE FUNCTION update_fact_span_stats()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO fact_span_stats (drug_name, section_name, last_updated)
    VALUES (NEW.drug_name, NEW.section_name, NOW())
    ON CONFLICT (drug_name, section_name) 
    DO UPDATE SET 
        total_sentences = CASE WHEN NEW.text_type = 'sentence' THEN fact_span_stats.total_sentences + 1 ELSE fact_span_stats.total_sentences END,
        total_bullets = CASE WHEN NEW.text_type = 'bullet' THEN fact_span_stats.total_bullets + 1 ELSE fact_span_stats.total_bullets END,
        total_table_rows = CASE WHEN NEW.text_type = 'table_row' THEN fact_span_stats.total_table_rows + 1 ELSE fact_span_stats.total_table_rows END,
        total_captions = CASE WHEN NEW.text_type = 'caption' THEN fact_span_stats.total_captions + 1 ELSE fact_span_stats.total_captions END,
        last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_fact_span_stats
AFTER INSERT ON fact_spans
FOR EACH ROW
EXECUTE FUNCTION update_fact_span_stats();

COMMIT;
"""

# Export migration SQL for alembic or manual execution
__all__ = ['FactSpan', 'FactSpanStats', 'FACT_SPAN_MIGRATION_SQL']
