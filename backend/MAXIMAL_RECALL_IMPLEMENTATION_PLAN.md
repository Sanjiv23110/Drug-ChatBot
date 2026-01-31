# Maximal Recall RAG Extension - Implementation Plan

**Date**: 2026-01-31  
**Architect**: Principal Software Architect  
**Objective**: Extend existing verbatim-only medical QA system to 95-98% recall

---

## Executive Summary

This plan extends the current PostgreSQL-based RAG system with four new architectural components while **strictly preserving** the verbatim-only constraint. No LLM will see PDF content; all answers remain word-for-word extractions.

**Current State**: 73.3% success rate (44/60 AXID questions)  
**Target State**: 95-98% recall for all answerable questions  
**Method**: Add granular indexing + BM25 recall + multi-section evidence + LLM planning

---

## Architecture Overview

### Current System (Preserved)
```
User Query
    ↓
IntentClassifier (rule + optional LLM)
    ↓
RetrievalEngine (SQL-first)
    → SQL_EXACT
    → SQL_FUZZY (pg_trgm)
    → ATTRIBUTE_LOOKUP
    → VECTOR_SCOPED
    ↓
AnswerGenerator (verbatim-only prompts)
    ↓
Exact Text from PDF
```

### New Extensions (Added)
```
1. FactSpan Index → Granular retrieval units
2. LLM Planner → Query expansion (no PDF access)
3. BM25 Layer → Lexical recall amplifier
4. Multi-Section Assembly → Verbatim evidence bundling
5. Global Scan Safety Net → Maximal recall guarantee
```

---

## Part 1: FactSpan Index (NEW TABLE)

### Objective
Enable sub-section retrieval for precise fact extraction.

### Database Schema

```sql
CREATE TABLE fact_spans (
    id SERIAL PRIMARY KEY,
    
    -- Drug linkage
    drug_name VARCHAR(255) NOT NULL,
    brand_name VARCHAR(255),
    generic_name VARCHAR(255),
    
    -- Source tracking
    section_id INTEGER REFERENCES monograph_sections(id) ON DELETE CASCADE,
    section_name VARCHAR(512) NOT NULL,
    original_header VARCHAR(512),
    
    -- Content (VERBATIM - no normalization)
    text TEXT NOT NULL,  -- Exact text from PDF
    text_type VARCHAR(50) NOT NULL,  -- 'sentence' | 'bullet' | 'table_row' | 'caption'
    
    -- Position metadata
    page_number INTEGER,
    char_offset INTEGER,  -- Position within section
    sequence_num INTEGER, -- Order within section
    
    -- Document tracking
    document_hash VARCHAR(64) NOT NULL,
    
    -- Search optimization
    search_vector tsvector,  -- For BM25 via PostgreSQL FTS
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Indexes
    CONSTRAINT unique_span UNIQUE (document_hash, section_id, sequence_num)
);

-- Indexes for fast retrieval
CREATE INDEX idx_fact_spans_drug ON fact_spans(drug_name);
CREATE INDEX idx_fact_spans_section ON fact_spans(section_id);
CREATE INDEX idx_fact_spans_type ON fact_spans(text_type);
CREATE INDEX idx_fact_spans_search ON fact_spans USING GIN(search_vector);
CREATE INDEX idx_fact_spans_compound ON fact_spans(drug_name, section_name);
```

### Extraction Logic (Ingestion)

**Location**: `app/ingestion/factspan_extractor.py` (NEW FILE)

```python
class FactSpanExtractor:
    """
    Extract atomic fact units from sections.
    
    CRITICAL: All text is preserved verbatim - no modification.
    """
    
    @staticmethod
    def extract_spans(section_text: str, section_id: int) -> List[FactSpan]:
        """
        Parse section into fact spans.
        
        Returns:
            List of FactSpan objects ready for DB insertion
        """
        spans = []
        
        # 1. Extract sentences
        spans.extend(FactSpanExtractor._extract_sentences(section_text))
        
        # 2. Extract bullets
        spans.extend(FactSpanExtractor._extract_bullets(section_text))
        
        # 3. Extract table rows
        spans.extend(FactSpanExtractor._extract_table_rows(section_text))
        
        # 4. Extract captions
        spans.extend(FactSpanExtractor._extract_captions(section_text))
        
        return spans
    
    @staticmethod
    def _extract_sentences(text: str) -> List[Dict]:
        """
        Split into sentences using spaCy or NLTK.
        
        CRITICAL: Preserve exact whitespace and punctuation.
        """
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        return [
            {
                'text': sent.text,  # VERBATIM
                'text_type': 'sentence',
                'char_offset': sent.start_char,
                'sequence_num': i
            }
            for i, sent in enumerate(doc.sents)
        ]
    
    @staticmethod
    def _extract_bullets(text: str) -> List[Dict]:
        """
        Extract bullet points using regex patterns.
        """
        # Patterns: • - * ◦ numbered lists
        import re
        pattern = r'^[\s]*[•\-\*◦][\s]+(.+)$'
        
        bullets = []
        for i, line in enumerate(text.split('\n')):
            match = re.match(pattern, line)
            if match:
                bullets.append({
                    'text': line.strip(),  # VERBATIM including bullet
                    'text_type': 'bullet',
                    'sequence_num': i
                })
        
        return bullets
    
    @staticmethod
    def _extract_table_rows(text: str) -> List[Dict]:
        """
        Extract table rows from markdown or detected tables.
        
        Uses existing docling table extraction, preserves verbatim.
        """
        # Leverage existing markdown from docling
        # Tables are in markdown format: | col1 | col2 |
        import re
        
        table_rows = []
        in_table = False
        row_num = 0
        
        for line in text.split('\n'):
            if '|' in line:
                in_table = True
                table_rows.append({
                    'text': line.strip(),  # VERBATIM row
                    'text_type': 'table_row',
                    'sequence_num': row_num
                })
                row_num += 1
            elif in_table and not line.strip():
                in_table = False
        
        return table_rows
    
    @staticmethod
    def _extract_captions(text: str) -> List[Dict]:
        """
        Extract figure/table captions.
        
        Patterns: "Figure X:", "Table X:", "Image X:"
        """
        import re
        pattern = r'^(Figure|Table|Image|Diagram)\s+\d+[:\.](.+?)(?=\n|$)'
        
        captions = []
        for i, match in enumerate(re.finditer(pattern, text, re.MULTILINE)):
            captions.append({
                'text': match.group(0),  # VERBATIM caption
                'text_type': 'caption',
                'char_offset': match.start(),
                'sequence_num': i
            })
        
        return captions
```

### Integration with Ingestion

**Modify**: `app/ingestion/ingest.py`

```python
async def ingest_pdf(pdf_path: str):
    # ... existing ingestion logic ...
    
    # NEW: After creating monograph_sections
    for section in sections:
        section_db = await create_section(session, section)
        
        # Extract and store fact spans
        fact_extractor = FactSpanExtractor()
        spans = fact_extractor.extract_spans(
            section_text=section.content_text,
            section_id=section_db.id
        )
        
        for span in spans:
            fact_span = FactSpan(
                drug_name=section.drug_name,
                section_id=section_db.id,
                section_name=section.section_name,
                text=span['text'],  # VERBATIM
                text_type=span['text_type'],
                sequence_num=span['sequence_num'],
                document_hash=section.document_hash,
                search_vector=func.to_tsvector('english', span['text'])
            )
            session.add(fact_span)
        
        await session.commit()
```

---

## Part 2: LLM Retrieval Planner (NEW MODULE)

### Objective
Use LLM to generate retrieval instructions WITHOUT seeing PDF content.

### Implementation

**Location**: `app/retrieval/query_planner.py` (NEW FILE)

```python
from dataclasses import dataclass
from typing import List, Optional, Literal
from openai import AzureOpenAI
import json

@dataclass
class RetrievalPlan:
    """
    Structured retrieval instructions generated by LLM.
    
    CRITICAL: LLM never sees PDF content - only generates search strategy.
    """
    drug: str
    query_mode: Literal['SECTION', 'ATTRIBUTE', 'MULTI_SECTION', 'GENERIC', 'GLOBAL']
    attribute: Optional[str] = None
    candidate_sections: List[str] = None
    search_phrases: List[str] = None
    extraction_level: Literal['sentence', 'block'] = 'sentence'

class QueryPlanner:
    """
    LLM-based query planning for retrieval optimization.
    
    The LLM acts as a PLANNER, not a retriever or generator.
    """
    
    PLANNER_PROMPT = """You are a medical query analyzer for a drug information retrieval system.

Your ONLY job is to output a JSON plan for HOW to retrieve information - you do NOT retrieve or answer anything.

Given a user query, analyze it and output a structured retrieval plan in this EXACT JSON format:

{
  "drug": "string (lowercase drug name)",
  "query_mode": "SECTION | ATTRIBUTE | MULTI_SECTION | GENERIC | GLOBAL",
  "attribute": "string or null (e.g., 'half_life', 'bioavailability')",
  "candidate_sections": ["list of likely section names"],
  "search_phrases": ["list of search terms including synonyms"],
  "extraction_level": "sentence | block"
}

Query Modes:
- SECTION: Asking about a specific section (e.g., "contraindications")
- ATTRIBUTE: Asking about a specific medical attribute (e.g., "half-life")
- MULTI_SECTION: Requires info from multiple sections (e.g., "why adjust dose in renal patients")
- GENERIC: Broad question (e.g., "what is AXID")
- GLOBAL: No clear target (scan all content)

Medical Attributes (if applicable):
half_life, bioavailability, tmax, cmax, metabolism, elimination, onset_of_action, duration_of_action, pregnancy_risk, lactation, active_ingredient, mechanism_of_action

Section Examples:
indications, contraindications, dosage, warnings, adverse_reactions, pharmacology, pharmacokinetics, interactions, storage, overdosage, composition, pregnancy, patient_information

Search Phrase Strategy:
- Include medical synonyms (e.g., "elimination half-life", "terminal half-life", "t1/2")
- Include lay terms (e.g., "how long in body" → "half-life")
- Include abbreviations (e.g., "PK" → "pharmacokinetics")

Extraction Level:
- sentence: Need precise single fact
- block: Need complete explanation/list

User Query: {query}

Output ONLY valid JSON, no explanations."""

    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-12-01-preview"
        )
        self.model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
    
    async def plan(self, query: str) -> RetrievalPlan:
        """
        Generate retrieval plan from user query.
        
        Args:
            query: User's natural language question
            
        Returns:
            RetrievalPlan with structured instructions
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": self.PLANNER_PROMPT.format(query=query)}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            plan_json = json.loads(response.choices[0].message.content)
            
            return RetrievalPlan(
                drug=plan_json['drug'],
                query_mode=plan_json['query_mode'],
                attribute=plan_json.get('attribute'),
                candidate_sections=plan_json.get('candidate_sections', []),
                search_phrases=plan_json.get('search_phrases', []),
                extraction_level=plan_json.get('extraction_level', 'sentence')
            )
            
        except Exception as e:
            logger.error(f"Query planning failed: {e}")
            # Fallback: basic plan
            return RetrievalPlan(
                drug="unknown",
                query_mode="GLOBAL",
                search_phrases=[query]
            )
```

---

## Part 3: BM25 Recall Amplifier (NEW RETRIEVAL PATH)

### Objective
Use PostgreSQL full-text search (BM25-like) on fact_spans as recall safety net.

### Implementation

**Location**: Extend `app/retrieval/retrieve.py`

```python
class RetrievalPath(str, Enum):
    # Existing paths
    SQL_EXACT = "SQL_EXACT"
    SQL_FUZZY = "SQL_FUZZY"
    ATTRIBUTE_LOOKUP = "ATTRIBUTE_LOOKUP"
    # NEW PATHS
    BM25_FACTSPAN = "BM25_FACTSPAN"
    MULTI_SECTION_EVIDENCE = "MULTI_SECTION_EVIDENCE"
    GLOBAL_FACTSPAN_SCAN = "GLOBAL_FACTSPAN_SCAN"
    # Existing
    VECTOR_SCOPED = "VECTOR_SCOPED"
    NO_RESULT = "NO_RESULT"

# NEW METHOD in RetrievalEngine
async def _path_d_bm25_factspan(
    self,
    plan: RetrievalPlan,
    result: RetrievalResult
) -> RetrievalResult:
    """
    Path D: BM25-style full-text search on fact_spans.
    
    Uses PostgreSQL's ts_rank for BM25-like scoring.
    Searches using planner-generated phrases.
    """
    async with get_session() as session:
        # Build search query from plan
        search_terms = " | ".join(plan.search_phrases)  # OR logic
        
        query = text("""
            SELECT 
                fs.text,
                fs.text_type,
                fs.section_name,
                fs.page_number,
                ms.content_text as section_context,
                ts_rank(fs.search_vector, to_tsquery('english', :search_terms)) as rank
            FROM fact_spans fs
            JOIN monograph_sections ms ON fs.section_id = ms.id
            WHERE 
                fs.drug_name = :drug_name
                AND fs.search_vector @@ to_tsquery('english', :search_terms)
            ORDER BY rank DESC
            LIMIT 20
        """)
        
        db_result = await session.execute(
            query,
            {
                "drug_name": plan.drug,
                "search_terms": search_terms
            }
        )
        
        fact_spans = db_result.fetchall()
        
        if fact_spans:
            # Convert fact_spans to section-like format for QA
            result.sections = [
                {
                    'content_text': fs.text,  # VERBATIM fact
                    'section_name': fs.section_name,
                    'page_num': fs.page_number,
                    'drug_name': plan.drug,
                    'text_type': fs.text_type,
                    '_is_fact_span': True
                }
                for fs in fact_spans
            ]
            result.path_used = RetrievalPath.BM25_FACTSPAN
            result.total_results = len(fact_spans)
        
        return result
```

### Placement in Retrieval Pipeline

**Modify**: `RetrievalEngine.retrieve()` method

```python
async def retrieve(self, query: str) -> RetrievalResult:
    # Step 1: Intent classification (existing)
    intent = self.classifier.classify(query)
    
    # Step 2: Query planning (NEW)
    planner = QueryPlanner()
    plan = await planner.plan(query)
    
    result = RetrievalResult(original_query=query, intent=intent)
    
    # Existing paths (1-4)
    if intent.target_drug and intent.target_section:
        result = await self._path_a_sql_match(intent, result)
        if result.sections:
            return result
    
    if intent.target_drug and intent.target_attribute:
        result = await self._path_a_attribute_lookup(intent, result)
        if result.sections:
            return result
    
    if intent.target_drug and not intent.target_section:
        result = await self._path_a_sql_drug_only(intent, result)
        if result.sections:
            return result
    
    # NEW: Path D - BM25 FactSpan Search
    if plan.drug != "unknown":
        result = await self._path_d_bm25_factspan(plan, result)
        if result.sections:
            return result
    
    # NEW: Path E - Global FactSpan Scan (FINAL SAFETY NET)
    if plan.drug != "unknown":
        result = await self._path_e_global_scan(plan, result)
        if result.sections:
            return result
    
    # Existing vector fallback
    if self.enable_vector_fallback and intent.target_drug:
        result = await self._path_c_vector_scoped(intent, result)
    
    return result
```

---

## Part 4: Multi-Section Verbatim Evidence Assembly

### Objective
Return verbatim text from multiple sections when query requires cross-section reasoning.

### Implementation

**Location**: `app/generation/multi_section_formatter.py` (NEW FILE)

```python
class MultiSectionFormatter:
    """
    Format multi-section evidence without synthesis.
    
    CRITICAL: No glue text, no explanations - just verbatim blocks.
    """
    
    @staticmethod
    def format_evidence_bundle(sections: List[Dict]) -> str:
        """
        Format multiple sections as verbatim evidence bundle.
        
        Args:
            sections: List of section dicts with content_text
            
        Returns:
            Formatted string with section headers
        """
        evidence_parts = []
        
        for section in sections:
            section_name = section.get('section_name', 'Unknown Section').upper()
            content = section.get('content_text', '')
            
            # Format: [SECTION: NAME]\n<verbatim text>\n\n
            evidence_parts.append(
                f"[SECTION: {section_name}]\n{content}\n"
            )
        
        return "\n".join(evidence_parts)
```

**Modify**: `app/generation/answer_generator.py`

```python
def generate(self, query: str, context_chunks: List[Dict]) -> Dict:
    # Detect multi-section scenario
    unique_sections = {chunk.get('section_name') for chunk in context_chunks}
    
    if len(unique_sections) > 1:
        # Multi-section evidence mode
        from app.generation.multi_section_formatter import MultiSectionFormatter
        
        bundled_evidence = MultiSectionFormatter.format_evidence_bundle(context_chunks)
        
        # Use special prompt for multi-section
        user_prompt = f"""Multiple relevant sections found. Extract information from these sections:

{bundled_evidence}

User Question: {query}

INSTRUCTION: If the answer exists in ANY of the above sections, extract it verbatim. If it requires combining information from multiple sections, present each relevant piece separately with its section label."""
        
        response = self._call_llm(user_prompt)
        
        return {
            'answer': response,
            'sources': [{'section': s} for s in unique_sections],
            'has_answer': True,
            'multi_section': True
        }
    
    # Existing single-section logic
    # ...
```

---

## Part 5: Global FactSpan Safety Net

### Implementation

```python
async def _path_e_global_scan(
    self,
    plan: RetrievalPlan,
    result: RetrievalResult
) -> RetrievalResult:
    """
    Path E: Global fact span scan - FINAL SAFETY NET.
    
    Retrieves ALL fact_spans for the drug and filters using search phrases.
    Guarantees maximal recall at cost of more processing.
    """
    async with get_session() as session:
        # Get ALL fact_spans for drug
        query = text("""
            SELECT 
                fs.text,
                fs.text_type,
                fs.section_name,
                fs.page_number,
                ts_rank(fs.search_vector, to_tsquery('english', :search_terms)) as rank
            FROM fact_spans fs
            WHERE 
                fs.drug_name = :drug_name
            ORDER BY rank DESC
            LIMIT 50  -- Broader than BM25 path
        """)
        
        # Use plan search phrases
        search_terms = " | ".join(plan.search_phrases) if plan.search_phrases else plan.drug
        
        db_result = await session.execute(
            query,
            {
                "drug_name": plan.drug,
                "search_terms": search_terms
            }
        )
        
        spans = db_result.fetchall()
        
        if spans:
            result.sections = [
                {
                    'content_text': span.text,
                    'section_name': span.section_name,
                    'page_num': span.page_number,
                    'drug_name': plan.drug,
                    'text_type': span.text_type
                }
                for span in spans if span.rank > 0.01  # Minimal relevance threshold
            ]
            result.path_used = RetrievalPath.GLOBAL_FACTSPAN_SCAN
            result.total_results = len(result.sections)
        
        return result
```

---

## Part 6: Integration Checklist

### Files to Create
1. `app/db/models.py` - Add `FactSpan` model
2. `app/ingestion/factspan_extractor.py` - Fact extraction logic
3. `app/retrieval/query_planner.py` - LLM planner
4. `app/generation/multi_section_formatter.py` - Evidence bundling

### Files to Modify
1. `app/retrieval/retrieve.py` - Add BM25 and global scan paths
2. `app/ingestion/ingest.py` - Add factspan extraction to pipeline
3. `app/generation/answer_generator.py` - Add multi-section handling

### Database Migrations
```sql
-- Migration: Add fact_spans table
-- Run via alembic or direct SQL

-- 1. Create fact_spans table (see schema above)
-- 2. Create indexes
-- 3. Backfill from existing monograph_sections (optional)
```

### Testing Strategy
1. **Unit Tests**: FactSpanExtractor (sentence/bullet/table extraction)
2. **Integration Tests**: BM25 retrieval path
3. **E2E Tests**: Re-run AXID test suite (target: 95%+ success)

---

## Success Metrics

### Before Implementation (Current)
- Success Rate: 73.3% (44/60)
- NO_RESULT: 16 queries (26.7%)
- Path Distribution: 50% SQL

### After Implementation (Target)
- Success Rate: 95%+ (57/60)
- NO_RESULT: <3% (provably unanswerable)
- Path Distribution:
  - SQL: 40%
  - BM25: 30%
  - Multi-Section: 20%
  - Global Scan: 5%
  - NO_RESULT: 5%

### Forbidden Outcomes
- ❌ LLM paraphrasing
- ❌ Inference-based answers
- ❌ Summarized responses
- ❌ LLM seeing PDF content

---

## Implementation Timeline

### Phase 1: Foundation (Week 1)
- Day 1-2: Create FactSpan model and migration
- Day 3-4: Implement FactSpanExtractor
- Day 5: Test extraction on sample PDFs

### Phase 2: Retrieval (Week 2)
- Day 1-2: Implement QueryPlanner
- Day 3-4: Add BM25 retrieval path
- Day 5: Add global scan safety net

### Phase 3: Multi-Section (Week 3)
- Day 1-2: Implement MultiSectionFormatter
- Day 3-4: Integrate with AnswerGenerator
- Day 5: Testing and refinement

### Phase 4: Validation (Week 4)
- Day 1-3: Re-run AXID comprehensive test
- Day 4-5: Fix edge cases, optimize performance

**Total**: 4 weeks to 95%+ recall

---

## Risk Mitigation

### Risk: Factspan extraction too slow
**Mitigation**: Batch processing, async extraction, caching

### Risk: BM25 returns too many irrelevant spans
**Mitigation**: Rank threshold, limit to top-K, use plan filtering

### Risk: Multi-section answers confuse users
**Mitigation**: Clear section labels, UI improvements

### Risk: Storage explosion (fact_spans table)
**Mitigation**: Compression, selective indexing, archival strategy

---

## Conclusion

This architecture preserves all verbatim constraints while adding:
1. **Granular retrieval** (fact_spans)
2. **Smart query expansion** (LLM planner)
3. **Lexical recall** (BM25)
4. **Multi-section evidence** (verbatim bundling)
5. **Maximal coverage** (global scan)

Expected outcome: **95-98% recall** with **zero inference**, **zero paraphrasing**, **zero hallucination**.
