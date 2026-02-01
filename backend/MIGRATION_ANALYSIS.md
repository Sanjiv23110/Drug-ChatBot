# Migration & Data Requirement Analysis

## Executive Summary

**Does this require re-ingestion?**
**Technically No**, but **Functionally Yes** (Data Population is Required).

You do **NOT** need to re-process the original PDF files if your current database (`monograph_sections`) is intact. However, you **MUST** populate the new `fact_spans` table for the new features to work.

We recommend a **Backfill Strategy** rather than a full Re-ingestion from PDFs.

## Technical Analysis

### 1. Data Dependency
The new features (`AttributeAggregator`, `BM25_FACTSPAN`, `GLOBAL_FACTSPAN_SCAN`) rely exclusively on the **`fact_spans`** table.
- **Current State**: `fact_spans` table is empty (or non-existent).
- **Required State**: `fact_spans` contains verbatim sentences/bullets extracted from sections.

### 2. Source Data Availability
The existing `monograph_sections` table contains the full `content_text` of every section.
- This text is the "Verified Source of Truth".
- It is sufficient to generate `FactSpans`.

### 3. Migration Strategy (Recommended: Backfill)

Instead of re-running the expensive PDF parsing pipeline (Docling/Layout parsing), we can run a script to transform existing sections into FactSpans.

#### Workflow:
1. **Apply Schema**: Run SQL migration to create `fact_spans` table (if not done).
2. **Backfill Script**:
   ```python
   # Pseudocode for Backfill
   sections = db.query(MonographSection).all()
   extractor = FactSpanExtractor()
   
   for section in sections:
       spans = extractor.extract(section.content_text)
       save_to_fact_spans(spans, section_id=section.id)
   ```
3. **Index Generation**: PostgreSQL automatically updates the `tsvector` index for BM25 upon insertion.

### 4. Comparison

| Strategy | Speed | CPU Cost | Risk | Recommendation |
|----------|-------|----------|------|----------------|
| **Full Re-ingestion** | Methods: ~1-2 mins/PDF | High (OCR/Parsing) | Parsing errors may change | No |
| **Database Backfill** | ~10ms/section | Low (Text splitting only) | Zero (Source unchanged) | **YES** |

## Implementation Impact

If you run the system **without** populating data:
- **Path A/A+ (Existing)**: Will work normally.
- **Path A++ (Attribute)**: Will fall back to section lookup (Safe).
- **BM25 Path**: Will return 0 results (Silent failure).
- **Global Scan**: Will return 0 results.

**Verdict**: The system remains stable but strictly functionally limited to the old capabilities until data is populated.

## Action Plan (Research Only)

To fully activate the system, you would need to:
1. Execute the SQL definitions in `app/db/fact_span_model.py`.
2. Run a script to iterate `monograph_sections` and populate `fact_spans`.
3. Analyze `FactSpanStats` to ensure extraction quality.
