# Maximal Recall Extension - Implementation Summary

## What Was Implemented

I've created the foundational architecture for extending your RAG system from 73% to 95%+ recall while strictly preserving verbatim-only constraints.

### Files Created

#### 1. **MAXIMAL_RECALL_IMPLEMENTATION_PLAN.md**
Comprehensive 4-week implementation plan covering:
- FactSpan indexing architecture
- LLM query planning (no PDF access)
- BM25 recall amplification
- Multi-section evidence assembly
- Global scan safety net
- Migration strategies and testing approach

#### 2. **app/db/fact_span_model.py**
SQLModel schema for granular fact indexing:
- `FactSpan` table for atomic retrievable units
- Support for sentences, bullets, table rows, captions
- PostgreSQL full-text search (ts_vector)
- BM25-style ranking via ts_rank
- Helper functions and migration SQL included

Key features:
- All text stored VERBATIM (no normalization)
- Enables sub-section retrieval
- Indexed for fast BM25 queries
- Linked to source sections (foreign key)

#### 3. **app/ingestion/factspan_extractor.py**
Production-grade fact extraction logic:
- Sentence splitting (spacy + regex fallback)
- Bullet point extraction
- Table row parsing (markdown format)
- Caption/footnote detection
- Extraction statistics

All extraction preserves exact text character-for-character.

#### 4. **app/retrieval/query_planner.py**
LLM-based retrieval strategy generator:
- Analyzes queries WITHOUT seeing PDFs
- Outputs structured JSON plans
- Expands search terms with medical synonyms
- Classifies query modes (SECTION, ATTRIBUTE, MULTI_SECTION, etc.)
- Suggests candidate sections
- Determines extraction granularity

Critical: LLM never sees source documents, only generates search strategies.

---

## Architecture Overview

### Current System (Preserved)
```
Query → Intent Classifier → SQL Retrieval → Verbatim Answer
```

### Extended System (New Capabilities)
```
Query → Intent Classifier → Query Planner (LLM)
    ↓
    ├─ SQL Exact/Fuzzy (existing)
    ├─ Attribute Lookup (existing)
    ├─ BM25 FactSpan Search (NEW)
    ├─ Multi-Section Evidence (NEW)
    └─ Global Scan Safety Net (NEW)
    ↓
Verbatim Answer (preserved)
```

---

## Next Steps (To Complete Implementation)

### Phase 1: Database Setup (1-2 days)
1. Run migration to create `fact_spans` table:
   ```bash
   # Option A: Via alembic
   alembic revision --autogenerate -m "Add fact_spans table"
   alembic upgrade head
   
   # Option B: Direct SQL (from fact_span_model.py)
   psql -d your_db -f migration.sql
   ```

2. Test table creation:
   ```python
   from app.db.fact_span_model import FactSpan
   # Verify schema
   ```

### Phase 2: Ingestion Integration (2-3 days)
1. Modify `app/ingestion/ingest.py`:
   ```python
   from app.ingestion.factspan_extractor import FactSpanExtractor
   
   # After creating monograph_section:
   extractor = FactSpanExtractor()
   spans = extractor.extract(section.content_text)
   
   for span in spans:
       fact_span = FactSpan(
           drug_name=section.drug_name,
           section_id=section_db.id,
           text=span.text,  # VERBATIM
           text_type=span.text_type,
           # ... other fields
       )
       session.add(fact_span)
   ```

2. Backfill existing data (optional):
   ```python
   # Script to extract fact_spans from existing monograph_sections
   ```

### Phase 3: BM25 Retrieval Path (3-4 days)
1. Extend `app/retrieval/retrieve.py`:
   - Add `RetrievalPath.BM25_FACTSPAN`
   - Add `RetrievalPath.GLOBAL_FACTSPAN_SCAN`
   - Implement `_path_d_bm25_factspan()` method
   - Implement `_path_e_global_scan()` method

2. Integrate `QueryPlanner`:
   ```python
   from app.retrieval.query_planner import QueryPlanner
   
   async def retrieve(self, query):
       intent = self.classifier.classify(query)
       plan = await QueryPlanner().plan(query)  # NEW
       
       # Use plan for BM25 search
       result = await self._path_d_bm25_factspan(plan, result)
   ```

### Phase 4: Multi-Section Evidence (2-3 days)
1. Create `app/generation/multi_section_formatter.py`:
   ```python
   def format_evidence_bundle(sections):
       # Format multiple sections with headers
       return "[SECTION: X]\n<verbatim>\n\n[SECTION: Y]\n<verbatim>"
   ```

2. Update `answer_generator.py`:
   - Detect multi-section queries
   - Use bundled evidence format
   - Preserve verbatim constraint

### Phase 5: Testing & Validation (5-7 days)
1. Re-run AXID comprehensive test:
   ```bash
   python test_axid_comprehensive.py
   ```

2. Measure improvement:
   - Target: 95%+ success rate (57/60+)
   - Track path distribution
   - Verify verbatim preservation

3. Edge case testing:
   - Table-heavy queries
   - Multi-section reasoning
   - Rare attributes

---

## Key Design Decisions

### ✅ What We Did Right

1. **Verbatim Preservation**: All fact extraction maintains exact text
2. **LLM Isolation**: Planner never sees PDFs, only generates strategies
3. **Incremental Extension**: Adds capabilities without modifying existing code
4. **Production-Ready**: Migrations, indexes, error handling included
5. **Scalable**: BM25 via PostgreSQL (no external dependencies)

### ⚠️ Important Constraints

1. **No LLM in QA Loop**: Answer generation remains verbatim-only
2. **No Inference**: System returns NO_RESULT if text doesn't exist
3. **No Paraphrasing**: Multi-section answers are verbatim blocks
4. **No Ranking by LLM**: BM25 ranking is lexical via ts_rank

---

## Performance Expectations

### Storage Impact
- **FactSpan Table Size**: ~5-10x monograph_sections (due to sentence granularity)
- **Index Size**: GIN index on search_vector (~30% of text size)
- **Total Overhead**: Estimate 2-3x current database size

### Query Performance
- **BM25 Search**: <100ms for typical queries (PostgreSQL FTS is fast)
- **Global Scan**: <500ms (retrieves up to 50 fact spans)
- **Extraction Time**: ~200ms per section during ingestion

### Accuracy Goals
- **Current**: 73.3% (44/60)
- **After BM25**: ~85% (51/60) - better recall
- **After Multi-Section**: ~92% (55/60) - cross-section queries
- **After Global Scan**: ~95% (57/60) - maximal coverage

---

## Rollback Plan

If issues arise:
1. New paths are ADDITIVE - can disable via feature flags
2. Existing SQL paths remain unchanged
3. FactSpan table can be dropped without affecting core system
4. Query planner failures fall back to existing intent classifier

---

## Success Criteria

### ✅ Implementation Complete When:
1. All fact_spans indexed (100% of existing sections)
2. BM25 path returns results for attribute queries
3. Multi-section queries return bundled evidence
4. Test suite shows 95%+ success rate
5. All answers remain verbatim (verified manually)

### ❌ Failure Conditions:
- LLM generates paraphrased answers
- Inference-based responses appear
- Performance degrades below 200ms/query
- False positives increase (returning wrong sections)

---

## Files Ready for Review

1. **Implementation Plan**: `MAXIMAL_RECALL_IMPLEMENTATION_PLAN.md`
2. **Database Model**: `app/db/fact_span_model.py`
3. **Extractor**: `app/ingestion/factspan_extractor.py`
4. **Query Planner**: `app/retrieval/query_planner.py`

All files include:
- Comprehensive documentation
- Type hints
- Error handling
- Example usage
- Non-negotiable constraint enforcement

---

## Questions for User

Before proceeding with implementation:

1. **Database Migration Strategy**: 
   - Use alembic or direct SQL?
   - Backfill existing data or only new ingestions?

2. **Feature Flag**:
   - Enable BM25 path gradually or all at once?
   - A/B test against current system?

3. **Performance Budget**:
   - Acceptable query latency (current: ~100ms)?
   - Storage budget for fact_spans table?

4. **Testing Scope**:
   - Run on AXID only or all drugs?
   - Manual review process for verbatim validation?

Ready to proceed with Phase 1 (database migration) when you approve.
