# HIERARCHICAL-EXTRACTIVE IMPLEMENTATION GUIDE
## MANDATORY REQUIREMENTS FOR WORD-TO-WORD ACCURACY

**THIS IS NOT OPTIONAL. THIS IS THE ONLY ACCEPTABLE IMPLEMENTATION.**

---

## CRITICAL ARCHITECTURE CHANGES

### ❌ PREVIOUS APPROACH (FAILED)
- Flat chunking (semantic chunks for both retrieval and display)
- 95% validation threshold
- LLM allowed to "generate" answers
- Single collection in Qdrant

### ✅ MANDATORY APPROACH (HIERARCHICAL-EXTRACTIVE)
- **Parent-child chunking** (children for search, parents for display)
- **98% validation threshold** (NOT 95%)
- **LLM as extraction engine ONLY** (NOT a writer)
- **Separate Qdrant collections** (children + parents)
- **XSLT table preservation** (FDA's official spl.xsl)
- **Fallback to RAW parent** (if validation fails)

---

## PHASE 1: INGESTION (HIERARCHICAL STRUCTURE)

### Step 1.1: Parse SPL XML with XSLT Table Conversion

**File**: `ingestion/spl_xml_parser.py`

```python
from ingestion.spl_xml_parser import SPLXMLParser, TablePreserver

# MANDATORY: Provide path to FDA's official spl.xsl stylesheet
parser = SPLXMLParser(xsl_path='path/to/fda/spl.xsl')

# Parse document
metadata, sections = parser.parse_document('lisinopril_spl.xml')

# Tables are now preserved with column/row semantics
for section in sections:
    if section.is_table:
        print(f"Table preserved in {section.section_name}")
        print(section.text_content)  # Markdown table format
```

**CRITICAL**: Without FDA's spl.xsl, table semantics will be lost. This is a CRITICAL DEFECT.

### Step 1.2: Create Hierarchical Parent-Child Chunks

**File**: `ingestion/hierarchical_chunking.py`

```python
from ingestion.hierarchical_chunking import HierarchicalChunker

# Initialize chunker
chunker = HierarchicalChunker()

# Create parent-child hierarchy
parents, children = chunker.chunk_document(metadata, sections)

print(f"Parents (SOURCE OF TRUTH): {len(parents)}")
print(f"Children (SEARCH INDEX): {len(children)}")

# Verify parent-child relationship
first_child = children[0]
print(f"Child sentence: {first_child.sentence_text}")
print(f"References parent: {first_child.parent_id}")

# Find parent
parent = next(p for p in parents if p.parent_id == first_child.parent_id)
print(f"Parent paragraph: {parent.raw_text}")
```

**STRUCTURE**:
```
PARENT (Paragraph):
"The most common adverse reactions (≥2%) are dizziness, headache, 
fatigue, and cough. In controlled trials, discontinuation due to 
adverse reactions occurred in 6% of patients."

    ├── CHILD 1 (Sentence): "The most common adverse reactions..."
    └── CHILD 2 (Sentence): "In controlled trials, discontinuation..."
```

### Step 1.3: Enrich with RxNorm

```python
from normalization.rxnorm_integration import DrugNormalizer

normalizer = DrugNormalizer()

# Get RxCUI
drug_info = normalizer.normalize_drug_name(metadata.drug_name)
metadata.rxcui = drug_info['rxcui']

# Update all chunks with RxCUI
for parent in parents:
    parent.rxcui = metadata.rxcui

for child in children:
    child.rxcui = metadata.rxcui
```

### Step 1.4: Generate Embeddings

```python
from retrieval.hybrid_retriever import DenseEmbedder
from qdrant_client.models import SparseVector

# Initialize embedder
embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")

# Generate embeddings for CHILDREN ONLY (search index)
child_texts = [child.sentence_text for child in children]
child_embeddings = embedder.embed(child_texts)

# Sparse vectors (BM25) - MANDATORY for keyword matching
# For now, use empty sparse vectors (implement BM25 separately)
sparse_vectors = [SparseVector(indices=[], values=[]) for _ in children]
```

### Step 1.5: Upsert to Qdrant (HIERARCHICAL)

```python
from vector_db.hierarchical_qdrant import HierarchicalQdrantManager

# Initialize Qdrant with HIERARCHICAL collections
qm = HierarchicalQdrantManager(
    host="localhost",
    port=6333,
    child_collection="spl_children",
    parent_collection="spl_parents"
)

# Create BOTH collections
qm.create_collections(dense_vector_size=768, recreate=False)

# Upsert CHILDREN (with vectors for search)
qm.upsert_children(
    children=[child.to_dict() for child in children],
    dense_embeddings=child_embeddings,
    sparse_embeddings=sparse_vectors
)

# Upsert PARENTS (NO vectors - pure storage)
qm.upsert_parents(
    parents=[parent.to_dict() for parent in parents]
)

print("✓ Hierarchical ingestion complete")
```

---

## PHASE 2: RETRIEVAL (HIERARCHICAL SEARCH)

### Step 2.1: Search Children, Retrieve Parents

```python
import numpy as np

# Generate query embedding
query = "What are the adverse reactions of Lisinopril?"
query_embedding = embedder.embed_query(query)
query_sparse = SparseVector(indices=[], values=[])

# Step 1: Search CHILDREN (sentence-level)
child_results = qm.search_children(
    query_dense=query_embedding,
    query_sparse=query_sparse,
    filter_conditions={
        "rxcui": "203644",  # Lisinopril
        "loinc_code": "34084-4"  # ADVERSE REACTIONS
    },
    limit=20  # Get top 20 child sentences
)

print(f"Found {len(child_results)} child sentences")

# Step 2: Extract unique parent IDs
parent_ids = list(set([child['parent_id'] for child in child_results]))
print(f"Unique parents: {len(parent_ids)}")

# Step 3: Retrieve PARENT chunks (SOURCE OF TRUTH)
parents = qm.get_parents_by_ids(parent_ids)

print(f"Retrieved {len(parents)} parent paragraphs")
print(f"First parent: {parents[0]['raw_text'][:200]}...")
```

**CRITICAL FLOW**:
```
User Query
    ↓
Search CHILDREN (sentence-level indexing)
    ↓
Extract parent_ids from matched children
    ↓
Retrieve PARENTS from parent collection
    ↓
Pass PARENTS to LLM (SOURCE OF TRUTH)
```

---

## PHASE 3: ANSWER GENERATION (EXTRACTIVE ONLY)

### Step 3.1: Extract Answer (98% Validation)

```python
from generation.extractive_system import (
    ExtractiveLLM,
    StrictValidator,
    ExtractiveQASystem
)

# Initialize extractive system
extractor = ExtractiveLLM(model_name="gpt-4o")
validator = StrictValidator(similarity_threshold=98)  # MUST be 98
qa_system = ExtractiveQASystem(extractor=extractor, validator=validator)

# Generate answer from PARENT chunks
result = qa_system.generate_answer(
    query=query,
    parent_chunks=parents  # VERBATIM SOURCE OF TRUTH
)

print(f"Status: {result['status']}")
print(f"Answer: {result['answer']}")
print(f"Validation: {result['validation_score']:.1f}%")
```

### Step 3.2: Handle Validation Failure (MANDATORY FALLBACK)

```python
if result['status'] == 'rejected_fallback':
    # Validation failed - system returned RAW PARENT instead
    print("⚠️ LLM extraction failed validation")
    print(f"Validation score: {result['validation_score']:.1f}% < 98%")
    print(f"Rejected extraction: {result['rejected_extraction']}")
    print(f"\nFalling back to RAW PARENT paragraph:")
    print(result['answer'])  # This is the VERBATIM parent text
```

**CRITICAL BEHAVIOR**:
- If validation ≥ 98% → Return LLM extraction
- If validation < 98% → Return RAW PARENT paragraph (VERBATIM)
- NEVER reject without showing source text

---

## COMPLETE END-TO-END EXAMPLE

```python
# ═══════════════════════════════════════════════════════
# COMPLETE HIERARCHICAL-EXTRACTIVE PIPELINE
# ═══════════════════════════════════════════════════════

from ingestion.spl_xml_parser import SPLXMLParser
from ingestion.hierarchical_chunking import HierarchicalChunker
from normalization.rxnorm_integration import DrugNormalizer
from retrieval.hybrid_retriever import DenseEmbedder
from vector_db.hierarchical_qdrant import HierarchicalQdrantManager
from generation.extractive_system import (
    ExtractiveLLM, StrictValidator, ExtractiveQASystem
)
from qdrant_client.models import SparseVector

# ─────────────────────────────────────────────────────────
# PHASE 1: INGESTION
# ─────────────────────────────────────────────────────────

print("=== PHASE 1: INGESTION ===")

# Parse SPL XML (with XSLT table preservation)
parser = SPLXMLParser(xsl_path='path/to/spl.xsl')
metadata, sections = parser.parse_document('lisinopril_spl.xml')

# Create hierarchical chunks
chunker = HierarchicalChunker()
parents, children = chunker.chunk_document(metadata, sections)

# Enrich with RxNorm
normalizer = DrugNormalizer()
drug_info = normalizer.normalize_drug_name(metadata.drug_name)
metadata.rxcui = drug_info['rxcui']

# Update chunks
for parent in parents:
    parent.rxcui = metadata.rxcui
for child in children:
    child.rxcui = metadata.rxcui

# Generate embeddings (children only)
embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
child_texts = [child.sentence_text for child in children]
child_embeddings = embedder.embed(child_texts)
sparse_vectors = [SparseVector(indices=[], values=[]) for _ in children]

# Upsert to Qdrant (hierarchical)
qm = HierarchicalQdrantManager(host="localhost", port=6333)
qm.create_collections(dense_vector_size=768, recreate=False)
qm.upsert_children(
    children=[child.to_dict() for child in children],
    dense_embeddings=child_embeddings,
    sparse_embeddings=sparse_vectors
)
qm.upsert_parents(parents=[parent.to_dict() for parent in parents])

print(f"✓ Ingested {len(parents)} parents, {len(children)} children")

# ─────────────────────────────────────────────────────────
# PHASE 2: RETRIEVAL
# ─────────────────────────────────────────────────────────

print("\n=== PHASE 2: RETRIEVAL ===")

query = "What are the adverse reactions of Lisinopril?"
query_embedding = embedder.embed_query(query)

# Search children
child_results = qm.search_children(
    query_dense=query_embedding,
    query_sparse=SparseVector(indices=[], values=[]),
    filter_conditions={"rxcui": metadata.rxcui, "loinc_code": "34084-4"},
    limit=20
)

# Extract parent IDs
parent_ids = list(set([child['parent_id'] for child in child_results]))

# Retrieve parents (SOURCE OF TRUTH)
retrieved_parents = qm.get_parents_by_ids(parent_ids)

print(f"✓ Retrieved {len(retrieved_parents)} parent paragraphs")

# ─────────────────────────────────────────────────────────
# PHASE 3: EXTRACTIVE GENERATION
# ─────────────────────────────────────────────────────────

print("\n=== PHASE 3: EXTRACTIVE GENERATION ===")

# Initialize extractive system
extractor = ExtractiveLLM(model_name="gpt-4o")
validator = StrictValidator(similarity_threshold=98)
qa_system = ExtractiveQASystem(extractor=extractor, validator=validator)

# Generate answer
result = qa_system.generate_answer(query, retrieved_parents)

# Display result
print(f"\nStatus: {result['status']}")
print(f"Validation: {result['validation_score']:.1f}%")
print(f"\nAnswer:\n{result['answer']}")

if result['status'] == 'rejected_fallback':
    print(f"\n⚠️ Fallback mode: Displaying RAW PARENT paragraph")
    print(f"Rejected extraction: {result['rejected_extraction']}")
```

---

## TECHNOLOGY STACK (FIXED - NO SUBSTITUTIONS)

| Component | Technology | Justification |
|-----------|------------|---------------|
| XML Parsing | lxml + XSLT | Namespace-aware, FDA spl.xsl support |
| Table Preservation | FDA spl.xsl | Official FDA stylesheet (MANDATORY) |
| Chunking | Hierarchical (parent-child) | WORD-TO-WORD accuracy requirement |
| Vector DB | Qdrant (Docker) | Rust-based, low RAM, hybrid search |
| Embeddings | S-PubMedBert | Biomedical domain, local inference |
| Sparse Vectors | BM25 | Keyword matching (drug names, doses) |
| LLM | Azure OpenAI GPT-4o | Extractive prompting only |
| Validation | RapidFuzz | 98% threshold (MANDATORY) |
| Drug Normalization | RxNorm/RxClass | Free NLM APIs |

**NO SUBSTITUTIONS ALLOWED**

---

## VALIDATION REQUIREMENTS

### ✅ CORRECT BEHAVIOR

**Scenario 1: Validation passes (≥98%)**
```
LLM Output: "The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough."
Parent Text: "The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough."
Similarity: 100%
Result: ✓ Display LLM output
```

**Scenario 2: Validation fails (<98%)**
```
LLM Output: "Patients may experience dizziness, headaches, tiredness, or coughing."
Parent Text: "The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough."
Similarity: 72%
Result: ✗ Display RAW PARENT paragraph instead
```

**Scenario 3: Not found**
```
LLM Output: "NOT_FOUND"
Result: ✓ Display "NOT_FOUND" to user
```

### ❌ FORBIDDEN BEHAVIOR

- Displaying paraphrased text (validation < 98%)
- Rejecting without showing source
- Using 95% threshold instead of 98%
- Allowing LLM to "generate" answers
- Flat chunking without parent-child hierarchy
- Tables without XSLT preservation

---

## DEPLOYMENT CHECKLIST

- [ ] FDA's spl.xsl stylesheet obtained and path configured
- [ ] Qdrant Docker container running
- [ ] TWO collections created (children + parents)
- [ ] S-PubMedBert model downloaded
- [ ] Azure OpenAI credentials configured
- [ ] RxNorm/RxClass API connectivity verified
- [ ] Hierarchical chunking implemented
- [ ] 98% validation threshold enforced
- [ ] Fallback to RAW parent implemented
- [ ] EXTRACTIVE system prompt loaded
- [ ] Parent-child relationships verified
- [ ] Table preservation tested

---

## CRITICAL REMINDERS

1. **Parent-child chunking is MANDATORY** - Not optional
2. **98% validation threshold** - Not 95%, not 90%
3. **LLM is extraction engine ONLY** - Not a writer
4. **Fallback to RAW parent** - Never reject without showing source
5. **XSLT table preservation** - Use FDA's official spl.xsl
6. **Separate Qdrant collections** - Children for search, parents for display
7. **Search children, display parents** - Never display child sentences

---

## PHILOSOPHY

> **"This system LOCATES and HIGHLIGHTS legal truth."**
> **"It does NOT answer questions. It EXTRACTS answers."**

Any deviation from this specification is a FAILURE.

---

**END OF MANDATORY IMPLEMENTATION GUIDE**
