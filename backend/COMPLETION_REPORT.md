# Ingestion Layer Implementation Report

**Status:** Complete
**Date:** 2026-01-31

## Actions Performed

1.  **Database Reset**
    - Executed `scripts/reset_db.py`.
    - All existing data (sections, chunks) has been cleared.

2.  **Schema Implementation (`app.db.fact_span_model`)**
    - Created `FactSpan` model matching strict requirements.
    - Fields: `fact_span_id`, `sentence_text`, `source_type`, `section_enum`, `sentence_index`, `page_number`, `original_header`.
    - Metadata: `attribute_tags` (JSON), `assertion_type`, `population_context`.
    - Engine: Uses `Computed` column used for `search_vector` (TSVector) for BM25.

3.  **Ingestion Logic (`app.ingestion.ingest.py`)**
    - Integrated `FactSpanExtractor` into `_store_sections`.
    - Logic: After `MonographSection` is flushed (ID generated), valid FactSpans are extracted and added to the session.
    - Consistency: Runs in same transaction as section storage.

4.  **Compatibility (`app.retrieval.retrieve.py`)**
    - Updated retrieval paths (Attribute Lookup, BM25, Global Scan) to use new field names (`sentence_text`, `source_type`).

## Usage Instructions

1.  **Re-Ingest Data**: You must run your PDF ingestion pipeline (e.g. `python scripts/ingest_folder.py` or similar) to populate the database.
2.  **Verification**: After ingestion, query the `fact_spans` table to verify granular data exists.

## Validation Code (Manual)
```python
# Check count
select(func.count(FactSpan.fact_span_id))
```
