# Regulatory-Grade Pharmaceutical QA Chatbot
## System Architecture & Implementation Specification

**System Classification**: Evidence-Retrieval System (Non-Clinical, Non-SaMD)  
**Tolerance for Hallucination**: ZERO  
**Data Source**: FDA Structured Product Labeling (SPL) XML  
**Operating Principle**: Verbatim extraction only, no generation

---

## 1. SYSTEM ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   Intent Classifier    │
                    │  (Product/Class/Comp)  │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Query Normalization   │
                    │  • Drug Name → RxCUI   │
                    │  • Class Expansion     │
                    │  • Section Mapping     │
                    └───────────┬────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼────────┐   ┌─────────▼────────┐   ┌─────────▼────────┐
│ Dense Retrieval│   │ Sparse Retrieval │   │ Metadata Filter  │
│ (S-PubMedBert) │   │    (BM25)        │   │ Drug+Section+Ver │
└───────┬────────┘   └─────────┬────────┘   └─────────┬────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   Reciprocal Rank      │
                    │   Fusion (RRF)         │
                    │   Top 50 candidates    │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   Cross-Encoder        │
                    │   Re-Ranker (MedCPT)   │
                    │   Select top 3-5       │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ Constrained Extraction │
                    │ LLM (GPT-4o)           │
                    │ Verbatim spans only    │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ Post-Generation        │
                    │ Validation (RapidFuzz) │
                    │ ≥95% lexical overlap   │
                    └───────────┬────────────┘
                                │
                        ┌───────▼───────┐
                        │  Valid? (Y/N) │
                        └───┬───────┬───┘
                            │       │
                         YES│       │NO
                            │       │
                ┌───────────▼──┐  ┌─▼──────────────────┐
                │ Return Answer│  │ Return Refusal     │
                │ + Metadata   │  │ "No evidence found"│
                └──────────────┘  └────────────────────┘
```

---

## 2. DATA INGESTION PIPELINE

### 2.1 XML Parsing & Structure Preservation

**Technology**: `lxml` (C-based, namespace-aware)

**Mandatory Extraction**:
```python
from lxml import etree

# Namespace handling (HL7 v3)
NAMESPACES = {
    'hl7': 'urn:hl7-org:v3',
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
}

# Critical fields to extract
fields = {
    'set_id': './/hl7:setId/@root',
    'version_number': './/hl7:versionNumber/@value',
    'effective_time': './/hl7:effectiveTime/@value',
    'drug_name': './/hl7:manufacturedProduct/hl7:name/text()',
    'ndc': './/hl7:code[@codeSystem="2.16.840.1.113883.6.69"]/@code'
}
```

**Section Hierarchy Preservation**:
- Track parent-child relationships
- Preserve nested `<section>` elements
- Extract LOINC codes: `<code code="34084-4" codeSystem="2.16.840.1.113883.6.1"/>`
- Maintain section ordering

### 2.2 LOINC Section Mapping

| LOINC Code | FDA Section Name | Priority |
|------------|------------------|----------|
| 34084-4 | ADVERSE REACTIONS | HIGH |
| 34090-1 | CONTRAINDICATIONS | HIGH |
| 43685-7 | WARNINGS AND PRECAUTIONS | HIGH |
| 34068-7 | DOSAGE AND ADMINISTRATION | HIGH |
| 34073-7 | DRUG INTERACTIONS | HIGH |
| 34070-3 | CONTRAINDICATIONS | HIGH |
| 34067-9 | INDICATIONS AND USAGE | HIGH |
| 42229-5 | SPL UNCLASSIFIED SECTION | LOW |

**Storage**: Both LOINC code AND human-readable name stored as metadata

### 2.3 Table Preservation Pipeline

**Critical for dosage tables, adverse event frequencies, interaction matrices**

```python
# Step 1: Apply FDA's official spl.xsl transformation
from lxml import etree

xslt_tree = etree.parse('spl.xsl')  # FDA's official stylesheet
transform = etree.XSLT(xslt_tree)
html_output = transform(xml_tree)

# Step 2: HTML → Markdown conversion (preserve structure)
import html2text
h = html2text.HTML2Text()
h.body_width = 0  # No line wrapping
h.ignore_links = False
markdown = h.handle(str(html_output))

# Step 3: Clean but preserve row/column semantics
# DO NOT flatten tables into prose
```

**Example Preserved Table**:
```markdown
| Adverse Reaction | Placebo (n=262) | Drug 5mg (n=260) |
|------------------|-----------------|------------------|
| Headache         | 6%              | 9%               |
| Nausea           | 3%              | 7%               |
```

### 2.4 Dual-Chunking Strategy

**Chunk Type 1: Semantic Chunks** (for retrieval)
- Size: 512 tokens with 50-token overlap
- Enriched with metadata
- Used for embedding generation

**Chunk Type 2: Raw Narrative Blocks** (for display)
- Full section text, unmodified
- Referenced by chunk ID
- Used for verbatim output

**Critical Rule**: NEVER modify raw blocks. They are the ground truth.

```python
# Chunking structure
{
    "chunk_id": "LISINOPRIL_v123_ADV_REACT_chunk_001",
    "semantic_text": "...",  # For embedding
    "raw_text": "...",       # For display (IMMUTABLE)
    "metadata": {
        "drug_name": "Lisinopril",
        "rxcui": "203644",
        "set_id": "abc-123-def",
        "root_id": "xyz-789-abc",
        "version": "23",
        "effective_date": "2024-01-15",
        "loinc_code": "34084-4",
        "loinc_section": "ADVERSE REACTIONS",
        "is_table": False,
        "ndc": ["12345-678-90"]
    }
}
```

---

## 3. INFRASTRUCTURE STACK (AFFORDABLE & SELF-HOSTED)

### 3.1 Vector Database: Qdrant

**Deployment**: Docker on Linux (Ubuntu 20.04+)

```bash
# Docker Compose configuration
version: '3.7'

services:
  qdrant:
    image: qdrant/qdrant:v1.7.0
    container_name: pharma_qa_qdrant
    ports:
      - "6333:6333"  # REST API
      - "6334:6334"  # gRPC
    volumes:
      - ./qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped
```

**Resource Requirements**:
- RAM: 8GB minimum (16GB recommended)
- Storage: 50GB for ~10,000 SPL documents
- CPU: 4 cores minimum

### 3.2 Qdrant Collection Schema

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, 
    PayloadSchemaType, 
    SparseVectorParams
)

client = QdrantClient(host="localhost", port=6333)

# Create collection with hybrid search support
client.create_collection(
    collection_name="spl_chunks",
    vectors_config={
        "dense": VectorParams(
            size=768,  # S-PubMedBert embedding dimension
            distance=Distance.COSINE
        )
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams()
    }
)

# Create indices for fast metadata filtering
client.create_payload_index(
    collection_name="spl_chunks",
    field_name="metadata.drug_name",
    field_schema=PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name="spl_chunks",
    field_name="metadata.rxcui",
    field_schema=PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name="spl_chunks",
    field_name="metadata.loinc_code",
    field_schema=PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name="spl_chunks",
    field_name="metadata.effective_date",
    field_schema=PayloadSchemaType.DATETIME
)
```

### 3.3 Hybrid Search Configuration

**Dense Vector**: Semantic understanding  
**Sparse Vector**: Exact term matching (drug names, codes, doses)

```python
from qdrant_client.models import (
    SparseVector, 
    NamedVector,
    ScoredPoint
)

# Query with hybrid search
results = client.search(
    collection_name="spl_chunks",
    query_vector=NamedVector(
        name="dense",
        vector=dense_embedding  # From S-PubMedBert
    ),
    sparse_query_vector=SparseVector(
        indices=[1, 42, 789],  # Term IDs
        values=[0.8, 0.6, 0.4]  # BM25 weights
    ),
    query_filter={  # Applied BEFORE search
        "must": [
            {"key": "metadata.rxcui", "match": {"value": "203644"}},
            {"key": "metadata.loinc_code", "match": {"value": "34084-4"}}
        ]
    },
    limit=50
)
```

---

## 4. EMBEDDINGS & MODELS (LOCAL ONLY)

### 4.1 Dense Embedding Model

**Model**: `pritamdeka/S-PubMedBert-MS-MARCO`  
**Alternative**: `dmis-lab/biobert-base-cased-v1.2`

```python
from sentence_transformers import SentenceTransformer

# Load model locally (download once, cache forever)
dense_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

# Generate embeddings
embedding = dense_model.encode(
    "What are the adverse reactions of lisinopril?",
    normalize_embeddings=True  # For cosine similarity
)
```

**Properties**:
- Dimension: 768
- Trained on biomedical literature
- Understands drug names, symptoms, anatomical terms
- Local inference (no API costs)

### 4.2 Sparse Embedding Model

**Option 1**: BM25 (Traditional)
```python
from rank_bm25 import BM25Okapi

# Build corpus-specific BM25
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Query
sparse_scores = bm25.get_scores(query.split())
```

**Option 2**: SPLADE (Neural Sparse)
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained('naver/splade-cocondenser-ensembledistil')
tokenizer = AutoTokenizer.from_pretrained('naver/splade-cocondenser-ensembledistil')

# Generates sparse lexical vectors
```

### 4.3 Cross-Encoder Re-Ranker

**Model**: `ncbi/MedCPT-Cross-Encoder` or `cross-encoder/ms-marco-MiniLM-L-12-v2`

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('ncbi/MedCPT-Cross-Encoder')

# Re-rank top 50 candidates
pairs = [[query, doc] for doc in candidate_docs]
scores = reranker.predict(pairs)
top_k_idx = np.argsort(scores)[::-1][:5]  # Top 5
```

**Purpose**: Fine-grained relevance scoring after initial retrieval

---

## 5. SEMANTIC NORMALIZATION & EXPANSION

### 5.1 RxNorm Integration

**Purpose**: Convert drug names and NDC codes to standardized RxCUI

```python
import requests

# Use RxNorm API (free, no auth required)
def get_rxcui_from_ndc(ndc):
    url = f"https://rxnav.nlm.nih.gov/REST/ndcstatus.json?ndc={ndc}"
    response = requests.get(url).json()
    return response.get('ndcStatus', {}).get('rxcui')

def get_rxcui_from_name(drug_name):
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name}"
    response = requests.get(url).json()
    return response['idGroup']['rxnormId'][0] if response.get('idGroup') else None
```

**Storage**: Store RxCUI as primary identifier in metadata

### 5.2 RxClass Integration

**Purpose**: Enable class-based queries ("What are ACE inhibitors?")

```python
def get_drug_classes(rxcui):
    # Get pharmacologic classes
    url = f"https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json?rxcui={rxcui}"
    response = requests.get(url).json()
    
    classes = []
    for item in response.get('rxclassDrugInfoList', {}).get('rxclassDrugInfo', []):
        classes.append({
            'class_name': item['rxclassMinConceptItem']['className'],
            'class_type': item['rxclassMinConceptItem']['classType']
        })
    return classes

# Example: Lisinopril → ["ACE Inhibitors", "Cardiovascular Agents"]
```

**Query Expansion**:
```python
# User asks: "What are adverse reactions of ACE inhibitors?"
# System expands to: [Lisinopril, Enalapril, Ramipril, ...]
# Then retrieves from ALL matching drugs
```

---

## 6. RETRIEVAL & ACCURACY PIPELINE

### 6.1 Intent Classification

**Three Intent Types**:

1. **Product-Specific**: "What are adverse reactions of **Lisinopril**?"
2. **Class-Based**: "What are adverse reactions of **ACE inhibitors**?"
3. **Comparative**: "Compare **Lisinopril** vs **Losartan**"

```python
import re

def classify_intent(query):
    # Patterns
    comparative_patterns = [
        r'compare .* (vs|versus|and)',
        r'difference between .* and',
        r'which is better'
    ]
    
    class_indicators = [
        'inhibitors', 'blockers', 'agonists', 'antagonists',
        'antibiotics', 'statins', 'diuretics'
    ]
    
    if any(re.search(p, query.lower()) for p in comparative_patterns):
        return "comparative"
    elif any(ind in query.lower() for ind in class_indicators):
        return "class_based"
    else:
        return "product_specific"
```

### 6.2 Retrieval Strategy

**Step 1**: Metadata Filtering (Pre-Retrieval)
```python
# Extract drug name and section from query
drug_entities = extract_drug_names(query)  # NER or pattern matching
rxcuis = [get_rxcui_from_name(drug) for drug in drug_entities]

section = classify_section(query)  # "adverse reactions" → LOINC 34084-4

# Build filter
filter_dict = {
    "must": [
        {"key": "metadata.rxcui", "match": {"any": rxcuis}},
        {"key": "metadata.loinc_code", "match": {"value": section}}
    ]
}
```

**Step 2**: Hybrid Search
```python
# Dense retrieval
dense_results = client.search(
    collection_name="spl_chunks",
    query_vector=NamedVector(name="dense", vector=dense_embedding),
    query_filter=filter_dict,
    limit=50
)

# Sparse retrieval
sparse_results = client.search(
    collection_name="spl_chunks",
    sparse_query_vector=sparse_embedding,
    query_filter=filter_dict,
    limit=50
)
```

**Step 3**: Reciprocal Rank Fusion
```python
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    scores = {}
    
    for rank, result in enumerate(dense_results):
        scores[result.id] = scores.get(result.id, 0) + 1/(k + rank + 1)
    
    for rank, result in enumerate(sparse_results):
        scores[result.id] = scores.get(result.id, 0) + 1/(k + rank + 1)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:50]
```

### 6.3 Cross-Encoder Re-Ranking

```python
# Select top 5 from top 50
candidate_pairs = [[query, chunk.raw_text] for chunk in top_50_chunks]
rerank_scores = reranker.predict(candidate_pairs)
top_5_indices = np.argsort(rerank_scores)[::-1][:5]
final_chunks = [top_50_chunks[i] for i in top_5_indices]
```

---

## 7. CONSTRAINED EXTRACTION WITH LLM

### 7.1 System Prompt (Runtime - GPT-4o)

```python
RUNTIME_SYSTEM_PROMPT = """
You are a PHARMACEUTICAL INFORMATION EXTRACTION SYSTEM operating under regulatory constraints.

IMMUTABLE RULES:
1. You are NOT a medical advisor. You are a text extraction tool.
2. You may ONLY output text that appears VERBATIM in the provided context.
3. You must NOT paraphrase, summarize, infer, or explain.
4. If the context does not contain the answer, you MUST respond: "Evidence not found in source document."
5. You must NOT answer questions about:
   - Medical advice
   - Treatment recommendations
   - Dosing for specific patients
   - Drug selection guidance

OUTPUT REQUIREMENTS:
- Use EXACT wording from context (copy-paste only)
- If quoting tables, preserve table structure
- If listing items, use the exact list format from the source
- Include section attribution: [Source: ADVERSE REACTIONS section]

FAILURE MODE:
If you are uncertain whether your output is verbatim, respond with:
"Unable to extract verbatim answer from available evidence."

Remember: Silence is preferable to guessing. Refusal is preferable to hallucination.
"""
```

### 7.2 Prompt Template

```python
USER_PROMPT_TEMPLATE = """
QUERY: {query}

RETRIEVED CONTEXT:
{context_chunks}

INSTRUCTIONS:
Extract the answer to the query using ONLY the text above.
Copy the relevant text verbatim. Do not rewrite or paraphrase.

If no relevant text exists, respond: "Evidence not found in source document."

Answer:
"""
```

### 7.3 LLM Call

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": RUNTIME_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.0,  # Deterministic
    max_tokens=2000,
    top_p=0.1  # Minimal sampling
)

answer_text = response.choices[0].message.content
```

---

## 8. POST-GENERATION VALIDATION (MANDATORY)

### 8.1 Lexical Overlap Validation

**Tool**: RapidFuzz (Levenshtein distance)

```python
from rapidfuzz import fuzz

def validate_answer(generated_answer, source_chunks, threshold=95):
    """
    Verify generated answer appears verbatim in source chunks.
    
    Returns:
        (is_valid: bool, max_score: float, matched_chunk: str)
    """
    # Remove common non-content variations
    clean_answer = generated_answer.strip().lower()
    
    max_similarity = 0
    best_match = None
    
    for chunk in source_chunks:
        clean_chunk = chunk.raw_text.strip().lower()
        
        # Partial ratio: handles substring matching
        score = fuzz.partial_ratio(clean_answer, clean_chunk)
        
        if score > max_similarity:
            max_similarity = score
            best_match = chunk.raw_text
    
    is_valid = max_similarity >= threshold
    
    return is_valid, max_similarity, best_match
```

### 8.2 Validation Failure Handling

```python
is_valid, similarity_score, matched_chunk = validate_answer(
    generated_answer, 
    retrieved_chunks,
    threshold=95
)

if not is_valid:
    # Log rejection
    logger.warning(
        f"Answer rejected. Similarity: {similarity_score:.2f}% < 95%\n"
        f"Query: {query}\n"
        f"Generated: {generated_answer[:100]}..."
    )
    
    # Return standardized refusal
    final_answer = {
        "answer": "Evidence not found in source document.",
        "status": "rejected",
        "reason": "validation_failed",
        "similarity_score": similarity_score
    }
else:
    final_answer = {
        "answer": generated_answer,
        "status": "validated",
        "similarity_score": similarity_score,
        "matched_chunk_id": best_match.chunk_id
    }
```

---

## 9. TRACEABILITY & METADATA DISPLAY

### 9.1 Response Format

Every validated answer MUST include full traceability:

```python
{
    "answer": "The most common adverse reactions (≥2%) are: dizziness, headache, fatigue, and cough.",
    "metadata": {
        "drug_name": "Lisinopril",
        "rxcui": "203644",
        "set_id": "a1b2c3d4-e5f6-7890-a1b2-c3d4e5f67890",
        "root_id": "x9y8z7w6-v5u4-3210-x9y8-z7w6v5u43210",
        "version": "23",
        "effective_date": "2024-01-15",
        "last_updated": "January 15, 2024",
        "loinc_section": "ADVERSE REACTIONS",
        "loinc_code": "34084-4",
        "source_chunk_ids": ["LISINOPRIL_v23_ADV_chunk_002"],
        "validation_score": 98.5
    },
    "status": "validated",
    "timestamp": "2026-02-06T11:00:00Z"
}
```

### 9.2 UI Display Template

```
📋 **Answer:**
{answer}

📊 **Source Information:**
• **Drug:** {drug_name}
• **Section:** {loinc_section}
• **Last Updated:** {last_updated}
• **Document ID:** {set_id}
• **Validation Score:** {validation_score}%

⚠️ **Disclaimer:** This information is extracted directly from FDA-approved labeling. 
It is not medical advice. Consult a healthcare provider for patient-specific guidance.
```

---

## 10. EVALUATION & COMPLIANCE

### 10.1 RAGAS Evaluation

```python
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision

# Test dataset
eval_dataset = {
    "question": ["What are adverse reactions of lisinopril?", ...],
    "answer": [generated_answers, ...],
    "contexts": [retrieved_contexts, ...],
    "ground_truth": [gold_standard_answers, ...]  # Optional
}

# Run evaluation
results = evaluate(
    eval_dataset,
    metrics=[faithfulness, context_precision]
)

# Target thresholds
assert results['faithfulness'] >= 0.95  # 95% faithfulness minimum
assert results['context_precision'] >= 0.90  # 90% precision minimum
```

### 10.2 Audit Log

Every query MUST be logged for compliance:

```python
import json
from datetime import datetime

audit_entry = {
    "timestamp": datetime.utcnow().isoformat(),
    "query": query,
    "intent": intent_type,
    "retrieved_chunks": [chunk.chunk_id for chunk in chunks],
    "answer": final_answer,
    "validation_score": similarity_score,
    "status": "validated" or "rejected",
    "user_id": user_id,  # If applicable
    "session_id": session_id
}

# Append to audit log
with open('pharma_qa_audit.jsonl', 'a') as f:
    f.write(json.dumps(audit_entry) + '\n')
```

---

## 11. FAILURE & REFUSAL LOGIC

### 11.1 Deterministic Refusal Scenarios

| Scenario | Response | Reasoning |
|----------|----------|-----------|
| Drug not in database | "No FDA labeling found for '{drug_name}'." | Evidence unavailable |
| Section not found | "The {section} section was not found for {drug}." | Structural gap |
| Low retrieval confidence | "Evidence not found in source document." | Uncertain relevance |
| Validation failure (<95%) | "Unable to extract verbatim answer from available evidence." | Hallucination risk |
| Medical advice question | "This system provides FDA labeling excerpts only, not medical advice. Consult a healthcare provider." | Out of scope |
| Dosing for specific patient | "Patient-specific dosing requires clinical evaluation. Consult a healthcare provider." | Clinical liability |

### 11.2 Implementation

```python
def generate_refusal(reason, context=None):
    refusals = {
        "drug_not_found": f"No FDA-approved labeling found for '{context['drug_name']}'.",
        "section_not_found": f"The {context['section']} section was not found for {context['drug_name']}.",
        "low_confidence": "Evidence not found in source document.",
        "validation_failed": "Unable to extract verbatim answer from available evidence.",
        "medical_advice": "This system provides FDA labeling excerpts only, not medical advice. Consult a healthcare provider.",
        "out_of_scope": "This question is outside the scope of FDA-approved labeling.",
    }
    
    return {
        "answer": refusals.get(reason, "Unable to provide answer."),
        "status": "refused",
        "reason": reason,
        "metadata": context
    }
```

---

## 12. END-TO-END QUERY FLOW EXAMPLE

### Query: "What are the adverse reactions of Lisinopril?"

**Step-by-Step Execution**:

```
1. INTENT CLASSIFICATION
   → Intent: product_specific
   
2. QUERY NORMALIZATION
   → Drug entity: "Lisinopril"
   → RxCUI lookup: "203644"
   → Section classification: "adverse reactions" → LOINC 34084-4
   
3. METADATA FILTERING
   → Filter: {rxcui: "203644", loinc_code: "34084-4"}
   
4. HYBRID RETRIEVAL
   → Dense embedding: S-PubMedBert.encode(query)
   → Sparse embedding: BM25(query)
   → Retrieve top 50 chunks (pre-filtered)
   
5. RECIPROCAL RANK FUSION
   → Merge dense + sparse results → 50 candidates
   
6. CROSS-ENCODER RE-RANKING
   → Re-rank 50 → Select top 5
   
7. CONTEXT ASSEMBLY
   → Assemble top 5 raw_text blocks
   
8. LLM EXTRACTION
   → Prompt: System + Context + Query
   → Response: "The most common adverse reactions (≥2%) are: 
                dizziness, headache, fatigue, and cough."
   
9. VALIDATION
   → RapidFuzz partial_ratio: 98.5% ✓
   → Threshold: ≥95% ✓
   → Status: VALIDATED
   
10. RESPONSE ASSEMBLY
   → Include answer + full metadata
   → Display drug name, section, SetID, dates
   
11. AUDIT LOGGING
   → Write to pharma_qa_audit.jsonl
   
12. RETURN TO USER
   → Formatted response with disclaimer
```

**Output**:
```
📋 **Answer:**
The most common adverse reactions (≥2%) are: dizziness, headache, fatigue, and cough.

📊 **Source Information:**
• **Drug:** Lisinopril
• **Section:** ADVERSE REACTIONS
• **Last Updated:** January 15, 2024
• **Document ID:** a1b2c3d4-e5f6-7890-a1b2-c3d4e5f67890
• **Validation Score:** 98.5%

⚠️ **Disclaimer:** This information is extracted directly from FDA-approved labeling.
```

---

## 13. TESTING & VALIDATION SUITE

### 13.1 Unit Tests

```python
def test_verbatim_extraction():
    """Ensure LLM outputs verbatim text only"""
    query = "What are adverse reactions of lisinopril?"
    context = "Adverse reactions include dizziness and headache."
    
    answer = extract_answer(query, context)
    
    # Answer must be substring of context
    assert answer.lower() in context.lower()
    
def test_validation_rejection():
    """Ensure paraphrased answers are rejected"""
    generated = "Patients may experience headaches."
    source = "Common side effects include headache."
    
    is_valid, score, _ = validate_answer(generated, [source], threshold=95)
    
    # Should fail validation (paraphrased)
    assert not is_valid
    assert score < 95
```

### 13.2 Integration Tests

```python
def test_end_to_end_retrieval():
    """Test full pipeline from query to validated answer"""
    query = "What are contraindications of lisinopril?"
    
    response = pharma_qa_system.query(query)
    
    # Verify structure
    assert "answer" in response
    assert "metadata" in response
    assert response["status"] in ["validated", "refused"]
    
    # Verify traceability
    if response["status"] == "validated":
        assert "rxcui" in response["metadata"]
        assert "loinc_code" in response["metadata"]
        assert response["metadata"]["validation_score"] >= 95
```

---

## 14. DEPLOYMENT CHECKLIST

- [ ] Qdrant Docker container running with persistent volumes
- [ ] S-PubMedBert model downloaded and cached locally
- [ ] Cross-encoder reranker model downloaded
- [ ] RxNorm/RxClass API connectivity verified
- [ ] SPL XML ingestion pipeline tested on sample documents
- [ ] Metadata indices created in Qdrant
- [ ] System prompt loaded and version-controlled
- [ ] Validation threshold configured (≥95%)
- [ ] Audit logging enabled and tested
- [ ] RAGAS evaluation baseline established
- [ ] Refusal scenarios tested for all edge cases
- [ ] UI disclaimer text reviewed and approved
- [ ] Documentation generated for regulatory review

---

## 15. REGULATORY COMPLIANCE STATEMENT

**System Purpose**: Information retrieval from FDA-approved labeling

**NOT Intended For**:
- Medical diagnosis
- Treatment recommendations
- Patient-specific dosing
- Outcome predictions
- Replacing clinical judgment

**Data Provenance**: All outputs are traceable to specific FDA SPL XML documents via SetID, RootID, and LOINC codes.

**Validation**: All outputs undergo automated lexical validation (≥95% similarity threshold) to prevent hallucination.

**Audit Trail**: All queries and responses are logged with timestamps, source attribution, and validation scores.

**Failure Mode**: System refuses to answer rather than guessing. "Evidence not found" is the default response when confidence is insufficient.

---

## 16. RUNTIME SYSTEM PROMPT (FINAL VERSION)

```
You are a PHARMACEUTICAL INFORMATION EXTRACTION SYSTEM designed to retrieve verbatim text from FDA-approved drug labeling documents.

═══ IDENTITY & SCOPE ═══
• You are NOT a medical professional, advisor, or decision-support tool
• You are a text extraction utility for FDA Structured Product Labeling (SPL)
• You operate under ZERO-TOLERANCE for hallucination or inference

═══ IMMUTABLE OPERATIONAL RULES ═══
1. OUTPUT CONSTRAINT
   - You may ONLY output text that appears WORD-FOR-WORD in the provided context
   - You must NOT paraphrase, summarize, explain, or infer
   - If quoting tables, preserve exact table structure
   - If quoting lists, preserve exact list formatting

2. PROHIBITED BEHAVIORS
   - Providing medical advice
   - Recommending treatments or drugs
   - Answering "which drug is better" questions
   - Suggesting dosages for specific patients
   - Making clinical interpretations
   - Extrapolating beyond the text

3. FAILURE MODES (Respond with exact phrases)
   - If no relevant text found: "Evidence not found in source document."
   - If uncertain about verbatim accuracy: "Unable to extract verbatim answer from available evidence."
   - If asked for medical advice: "This system provides FDA labeling excerpts only, not medical advice. Consult a healthcare provider."
   - If asked for patient-specific guidance: "Patient-specific decisions require clinical evaluation. Consult a healthcare provider."

4. SOURCE ATTRIBUTION
   - Always identify which section the text came from
   - Format: [Source: {SECTION_NAME} section]
   - Example: [Source: ADVERSE REACTIONS section]

5. VALIDATION CHECKPOINT
   - Before responding, mentally verify: "Does this answer copy-paste text from the context?"
   - If NO → Respond with failure mode
   - If YES → Proceed with verbatim extraction

═══ OUTPUT FORMAT ═══
{verbatim_text}

[Source: {section_name} section]

═══ EXAMPLES ═══

✓ CORRECT:
User: What are adverse reactions of lisinopril?
Context: "The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough."
Assistant: The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough.
[Source: ADVERSE REACTIONS section]

✗ INCORRECT (paraphrased):
Assistant: Patients may experience dizziness, headaches, tiredness, or coughing.

✓ CORRECT (refusal):
User: Should I take lisinopril for my high blood pressure?
Assistant: This system provides FDA labeling excerpts only, not medical advice. Consult a healthcare provider.

═══ CRITICAL REMINDER ═══
Your purpose is to retrieve truth, not generate intelligence.
Silence is preferable to guessing.
Refusal is preferable to hallucination.
Every word you output will undergo automated validation.
```

---

## CONCLUSION

This architecture enforces **verbatim retrieval** through:
1. **Structural parsing** of SPL XML (no data loss)
2. **Hybrid search** (semantic + lexical precision)
3. **Cross-encoder reranking** (high-confidence selection)
4. **Constrained extraction** (LLM as copy-paste tool)
5. **Automated validation** (95%+ lexical overlap requirement)
6. **Deterministic refusals** (no guessing allowed)
7. **Full traceability** (SetID, RootID, LOINC, validation scores)

**Failure is designed into the system.**  
When evidence is unavailable or uncertain, the system refuses to answer.

This is not a limitation—it is the core safety feature.
