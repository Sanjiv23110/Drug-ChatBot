"""
FactSpan Model - Granular Fact Indexing for Maximal Recall

This module defines the schema for sentence-level ingestion as per strict specificaton.
CRITICAL CONSTRAINT: All text is stored VERBATIM - no normalization, no paraphrasing.
"""
from typing import Optional, Any, List
from datetime import datetime
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, Text, Index, UniqueConstraint, text, Computed, JSON
from sqlalchemy.dialects.postgresql import TSVECTOR


class FactSpan(SQLModel, table=True):
    """
    Atomic fact units (sentences/bullets) extracted from monograph sections.
    Stores metadata for precise retrieval.
    """
    __tablename__ = "fact_spans"
    
    # Required Fields (Exact Semantics)
    fact_span_id: Optional[int] = Field(default=None, primary_key=True)
    
    drug_name: str = Field(index=True, max_length=255)
    
    sentence_text: str = Field(sa_column=Column(Text, nullable=False)) # EXACT verbatim sentence
    
    page_number: Optional[int] = Field(default=None)
    
    section_enum: str = Field(sa_column=Column(Text, index=True)) # normalized section name
    original_header: Optional[str] = Field(default=None, sa_column=Column(Text))
    
    sentence_index: int = Field(default=0) # order within section
    source_type: str = Field(max_length=50) # sentence | bullet | table_row | caption
    
    # Optional Metadata (Safe to Add)
    attribute_tags: List[str] = Field(default=[], sa_column=Column(JSON))
    assertion_type: Optional[str] = Field(default=None, max_length=50) # FACT | CONDITIONAL | ...
    population_context: Optional[str] = Field(default=None, max_length=100)
    
    table_id: Optional[str] = Field(default=None, max_length=100)
    table_header_text: Optional[str] = Field(default=None)
    
    # Linkage to Source Section (Critical for grouping)
    section_id: int = Field(foreign_key="monograph_sections.id")
    
    # Document Tracking
    document_hash: str = Field(index=True, max_length=64)
    original_filename: Optional[str] = Field(default=None, max_length=512)
    
    # Full-text search vector (Generated from sentence_text)
    search_vector: Any = Field(
        default=None,
        sa_column=Column(
            TSVECTOR,
            Computed("to_tsvector('english', sentence_text)", persisted=True)
        )
    )
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    __table_args__ = (
        # Unique constraint: section + sequence + type
        UniqueConstraint('section_id', 'sentence_index', 'source_type', name='uq_fact_span_sequence'),
        
        # Indexes for common queries
        Index('ix_factspan_drug_section', 'drug_name', 'section_enum'),
        Index('ix_factspan_source_type', 'source_type'),
        Index('ix_factspan_search', 'search_vector', postgresql_using='gin'),
    )


class FactSpanStats(SQLModel, table=True):
    """
    Statistics table for monitoring ingestion quality.
    """
    __tablename__ = "fact_span_stats"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    drug_name: str = Field(index=True)
    section_name: str = Field()
    
    total_spans: int = Field(default=0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
