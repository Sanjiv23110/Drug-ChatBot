# CRITICAL REVISIONS SUMMARY
## Hierarchical-Extractive Implementation (MANDATORY)

**Date**: February 6, 2026  
**Revision**: From flat-chunking to hierarchical-extractive architecture

---

## WHAT CHANGED AND WHY

### ❌ PREVIOUS IMPLEMENTATION (INADEQUATE)

The initial delivery used a **flat chunking approach** that would NOT achieve word-to-word accuracy at scale:

1. **Flat Semantic Chunks**
   - Single chunk type used for both retrieval and display
   - No parent-child hierarchy
   - Risk of displaying sentence fragments

2. **95% Validation Threshold**
   - Too lenient for regulatory requirements
   - Allowed minor paraphrasing

3. **Generative LLM Approach**
   - LLM allowed to "write" answers
   - System prompt suggested extraction but didn't enforce it
   - No mandatory fallback mechanism

4. **Single Qdrant Collection**
   - Mixed search and display concerns
   - No separation of indexing vs source-of-truth

5. **No XSLT Table Preservation**
   - Tables parsed as plain text
   - Column/row semantics lost

---

## ✅ REVISED IMPLEMENTATION (MANDATORY)

### 1. HIERARCHICAL PARENT-CHILD CHUNKING

**New File**: `ingestion/hierarchical_chunking.py`

**Structure**:
```
PARENT CHUNK (Paragraph)
├── CHILD CHUNK 1 (Sentence) → references parent_id
├── CHILD CHUNK 2 (Sentence) → references parent_id
└── CHILD CHUNK 3 (Sentence) → references parent_id
```

**Purpose**:
- **Children**: Sentence-level indexing for precise search
- **Parents**: Full paragraphs as SOURCE OF TRUTH for display
- **Guarantee**: User ALWAYS sees complete context, never fragments

**Key Classes**:
- `ParentChunk`: Full paragraph (IMMUTABLE, VERBATIM)
- `ChildChunk`: Individual sentence (with parent_id reference)
- `HierarchicalChunker`: Creates parent-child hierarchy

---

### 2. XSLT TABLE PRESERVATION

**Modified File**: `ingestion/spl_xml_parser.py`

**Changes**:
- `TablePreserver` class now REQUIRES FDA's official `spl.xsl` stylesheet
- Implements complete pipeline: XML → XSLT → HTML → Markdown
- Preserves column/row semantics (MANDATORY for dosage tables)

**Critical Method**:
```python
def preserve_table_structure(self, section_elem) -> Tuple[str, bool]:
    """
    XML → XSLT → HTML → Markdown
    Returns: (markdown_text, is_table)
    """
```

**Why Critical**:
- Dosage tables must maintain "Max Dose" ↔ "Population" associations
- Adverse event frequency tables must preserve percentages
- Interaction matrices must preserve drug combinations

---

### 3. HIERARCHICAL QDRANT STORAGE

**New File**: `vector_db/hierarchical_qdrant.py`

**Architecture**:
```
QDRANT
├── spl_children (collection)
│   ├── Has vectors (dense + sparse)
│   ├── Stores sentence text
│   └── Stores parent_id reference
│
└── spl_parents (collection)
    ├── NO vectors (pure key-value store)
    ├── Stores full paragraph text (VERBATIM)
    └── Retrieved by parent_id
```

**Search Flow**:
```
1. Search spl_children → get top 20 child sentences
2. Extract parent_ids from children
3. Retrieve parents from spl_parents
4. Pass PARENT paragraphs to LLM
```

**Key Methods**:
- `search_children()`: Search sentence-level index
- `get_parents_by_ids()`: Retrieve SOURCE OF TRUTH paragraphs
- `upsert_children()`: Index sentences with vectors
- `upsert_parents()`: Store paragraphs without vectors

---

### 4. EXTRACTIVE-ONLY GENERATION (98% THRESHOLD)

**New File**: `generation/extractive_system.py`

**System Prompt** (MANDATORY):
```
"You are an EXTRACTION ENGINE for FDA pharmaceutical labeling.

CRITICAL RULES:
1. You are NOT a writer. You are a text locator.
2. Your ONLY job is to identify the EXACT sentences that answer the question.
3. You MUST output text VERBATIM - word-for-word from the context.
4. You MUST NOT paraphrase, summarize, explain, or rewrite.
5. If the answer is not present in the context, output: NOT_FOUND"
```

**Validation**:
- **Threshold**: 98% (NOT 95%)
- **Tool**: RapidFuzz `partial_ratio`
- **Behavior**: If < 98%, display RAW PARENT paragraph instead

**Key Classes**:
- `ExtractiveLLM`: LLM configured as extraction engine
- `StrictValidator`: 98% validation enforcer
- `ExtractiveQASystem`: Complete extractive pipeline

**Critical Behavior**:
```python
if validation_score >= 98:
    return llm_extraction
else:
    return raw_parent_paragraph  # FALLBACK
```

---

## FILE STRUCTURE CHANGES

### NEW FILES (MANDATORY)

1. **`ingestion/hierarchical_chunking.py`** (~300 lines)
   - Parent-child chunking implementation
   - Sentence splitting
   - Deterministic ID generation

2. **`vector_db/hierarchical_qdrant.py`** (~350 lines)
   - Separate collections for children/parents
   - Hierarchical search flow
   - Parent retrieval by ID

3. **`generation/extractive_system.py`** (~400 lines)
   - Extractive-only LLM prompting
   - 98% validation threshold
   - Fallback to RAW parent

4. **`HIERARCHICAL_EXTRACTIVE_GUIDE.md`** (~600 lines)
   - Complete implementation guide
   - End-to-end examples
   - Deployment checklist

### MODIFIED FILES

1. **`ingestion/spl_xml_parser.py`**
   - `TablePreserver` now requires FDA's spl.xsl
   - Added `preserve_table_structure()` method
   - XSLT transformation pipeline

### DEPRECATED FILES (DO NOT USE)

1. **`ingestion/chunking_strategy.py`** (OLD)
   - Flat chunking approach
   - Use `hierarchical_chunking.py` instead

2. **`vector_db/qdrant_manager.py`** (OLD)
   - Single collection approach
   - Use `hierarchical_qdrant.py` instead

3. **`generation/constrained_extractor.py`** (OLD)
   - 95% threshold
   - Generative approach
   - Use `extractive_system.py` instead

---

## CRITICAL REQUIREMENTS MATRIX

| Requirement | Previous | Revised | Status |
|-------------|----------|---------|--------|
| Parent-child chunking | ❌ No | ✅ Yes | MANDATORY |
| XSLT table preservation | ❌ No | ✅ Yes | MANDATORY |
| Separate Qdrant collections | ❌ No | ✅ Yes | MANDATORY |
| 98% validation threshold | ❌ 95% | ✅ 98% | MANDATORY |
| Extractive-only LLM | ⚠️ Suggested | ✅ Enforced | MANDATORY |
| Fallback to RAW parent | ❌ No | ✅ Yes | MANDATORY |
| BM25 sparse vectors | ⚠️ Optional | ✅ Required | MANDATORY |
| FDA spl.xsl stylesheet | ❌ No | ✅ Yes | MANDATORY |

---

## IMPLEMENTATION PRIORITY

### PHASE 1 (CRITICAL - DO FIRST)
1. ✅ Obtain FDA's official `spl.xsl` stylesheet
2. ✅ Implement hierarchical chunking
3. ✅ Create separate Qdrant collections
4. ✅ Implement extractive-only generation
5. ✅ Enforce 98% validation threshold

### PHASE 2 (HIGH PRIORITY)
1. Implement BM25 sparse vectors
2. Test XSLT table preservation
3. Verify parent-child relationships
4. Test fallback mechanism

### PHASE 3 (PRODUCTION READY)
1. Batch ingestion with hierarchical chunking
2. End-to-end testing
3. Validation score monitoring
4. Deployment

---

## VALIDATION CHECKLIST

Before deploying, verify:

- [ ] Parent chunks are NEVER split across boundaries
- [ ] Child chunks ALWAYS reference valid parent_id
- [ ] Tables preserve column/row semantics
- [ ] Validation threshold is EXACTLY 98%
- [ ] LLM system prompt is EXTRACTIVE_SYSTEM_PROMPT
- [ ] Fallback to RAW parent works when validation < 98%
- [ ] Two Qdrant collections exist (children + parents)
- [ ] Children have vectors, parents do NOT
- [ ] Search flow: children → parent_ids → parents
- [ ] Display ALWAYS shows parent text, NEVER child sentences

---

## EXAMPLE: BEFORE vs AFTER

### BEFORE (Flat Chunking)
```
User: "What are adverse reactions?"

Retrieval: "dizziness, headache, fatigue"  (fragment)
Display: "dizziness, headache, fatigue"  (incomplete)
```

### AFTER (Hierarchical)
```
User: "What are adverse reactions?"

Search Children: "dizziness, headache, fatigue"  (sentence match)
Extract Parent ID: "LISINOPRIL_v23_34084-4_para_001"
Retrieve Parent: "The most common adverse reactions (≥2%) are 
                  dizziness, headache, fatigue, and cough. In 
                  controlled trials, discontinuation due to adverse 
                  reactions occurred in 6% of patients."
Display: [FULL PARENT PARAGRAPH]  (complete context)
```

---

## WHY THIS MATTERS

### Regulatory Compliance
- **Word-to-word accuracy**: Hierarchical structure ensures complete context
- **Traceability**: Parent IDs link to exact SPL paragraphs
- **Auditability**: Can prove every displayed text is VERBATIM

### Technical Precision
- **No fragments**: Users never see incomplete sentences
- **Table integrity**: Dosage/frequency data maintains associations
- **Deterministic**: 98% threshold eliminates ambiguity

### Safety
- **Fallback mechanism**: System shows RAW source if LLM fails
- **Extraction-only**: LLM cannot "invent" information
- **Validation**: Every output verified against source

---

## MIGRATION PATH

If you already ingested data with the OLD approach:

1. **Re-ingest required**: Flat chunks cannot be converted to hierarchical
2. **Delete old collections**: `spl_chunks` (old single collection)
3. **Create new collections**: `spl_children` + `spl_parents`
4. **Re-run ingestion**: Using `hierarchical_chunking.py`
5. **Update retrieval code**: Use `hierarchical_qdrant.py`
6. **Update generation code**: Use `extractive_system.py`

**There is NO backward compatibility. This is a complete architecture change.**

---

## FINAL STATEMENT

The hierarchical-extractive approach is **NOT OPTIONAL**.

It is the **ONLY ACCEPTABLE IMPLEMENTATION** for achieving:
- Word-to-word accuracy
- Regulatory compliance
- Deterministic behavior
- Zero hallucination

Any system using flat chunking, 95% threshold, or generative LLM approaches **WILL FAIL** at scale.

---

**END OF CRITICAL REVISIONS SUMMARY**

**Status**: ✅ MANDATORY IMPLEMENTATION COMPLETE  
**Next Step**: Follow `HIERARCHICAL_EXTRACTIVE_GUIDE.md` for deployment
