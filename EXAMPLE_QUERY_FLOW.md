# End-to-End Query Flow Example
## Regulatory-Grade Pharmaceutical QA System

This document traces a complete query from user input to validated answer, showing every intermediate step.

---

## Example Query

**User Input**: "What are the adverse reactions of Lisinopril?"

---

## STEP 1: Intent Classification

**Input**: "What are the adverse reactions of Lisinopril?"

**Process**:
```python
intent_classifier = IntentClassifier(drug_normalizer)
intent, metadata = intent_classifier.classify(query)
```

**Checks Performed**:
- ❌ Medical advice request? → NO
- ❌ Patient-specific question? → NO
- ❌ Comparative query? → NO
- ❌ Class-based query? → NO
- ✅ Product-specific query? → YES

**Output**:
```python
{
    "intent": "product_specific",
    "metadata": {}
}
```

---

## STEP 2: Drug Name Extraction & Normalization

**Input**: "Lisinopril" (extracted from query)

**Process**:
```python
drug_normalizer = DrugNormalizer()
drug_info = drug_normalizer.normalize_drug_name("Lisinopril")
```

**RxNorm API Call**:
```
GET https://rxnav.nlm.nih.gov/REST/rxcui.json?name=Lisinopril
```

**RxNorm Response**:
```json
{
    "idGroup": {
        "rxnormId": ["203644"]
    }
}
```

**RxClass API Call** (get drug classes):
```
GET https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json?rxcui=203644
```

**RxClass Response**:
```json
{
    "rxclassDrugInfoList": {
        "rxclassDrugInfo": [
            {
                "rxclassMinConceptItem": {
                    "className": "ACE Inhibitors",
                    "classType": "PE"
                }
            },
            {
                "rxclassMinConceptItem": {
                    "className": "Cardiovascular Agents",
                    "classType": "EPC"
                }
            }
        ]
    }
}
```

**Output**:
```python
{
    "drug_name": "Lisinopril",
    "rxcui": "203644",
    "classes": [
        {"class_name": "ACE Inhibitors", "class_type": "PE"},
        {"class_name": "Cardiovascular Agents", "class_type": "EPC"}
    ]
}
```

---

## STEP 3: Section Classification

**Input**: "What are the **adverse reactions** of Lisinopril?"

**Process**:
```python
section_classifier = SectionClassifier()
loinc_code = section_classifier.classify(query)
```

**Keyword Matching**:
- Query contains "adverse reactions"
- Maps to LOINC code: **34084-4** (ADVERSE REACTIONS section)

**Output**:
```python
{
    "loinc_code": "34084-4",
    "section_name": "ADVERSE REACTIONS"
}
```

---

## STEP 4: Metadata Filtering (Pre-Retrieval)

**Build Filter Conditions**:
```python
filter_conditions = {
    "rxcui": "203644",           # Lisinopril only
    "loinc_code": "34084-4"      # ADVERSE REACTIONS section only
}
```

**Qdrant Filter Translation**:
```python
Filter(
    must=[
        FieldCondition(key="metadata.rxcui", match=MatchValue(value="203644")),
        FieldCondition(key="metadata.loinc_code", match=MatchValue(value="34084-4"))
    ]
)
```

This ensures we **ONLY** search within Lisinopril's adverse reactions section - no other drugs, no other sections.

---

## STEP 5: Query Embedding Generation

### Dense Embedding (S-PubMedBert)

**Input**: "What are the adverse reactions of Lisinopril?"

**Process**:
```python
dense_embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
query_embedding = dense_embedder.embed_query(query)
```

**Output**:
```python
# 768-dimensional vector (normalized)
array([0.0234, -0.0156, 0.0891, ..., 0.0423])  # Shape: (768,)
```

### Sparse Embedding (BM25) - Optional

**Input**: "What are the adverse reactions of Lisinopril?"

**Process**:
```python
sparse_embedder = SparseEmbedder(corpus)
query_sparse = sparse_embedder.embed_query(query)
```

**Output**:
```python
SparseVector(
    indices=[42, 789, 1523, 2048],
    values=[0.85, 0.62, 0.45, 0.31]
)
```

---

## STEP 6: Hybrid Search in Qdrant

**Dense Search**:
```python
dense_results = qdrant_client.search(
    collection_name="spl_chunks",
    query_vector=NamedVector(name="dense", vector=query_embedding),
    query_filter=filter_conditions,
    limit=50
)
```

**Sparse Search** (if enabled):
```python
sparse_results = qdrant_client.search(
    collection_name="spl_chunks",
    sparse_query_vector=query_sparse,
    query_filter=filter_conditions,
    limit=50
)
```

**Results Retrieved**:
- Dense search: 50 candidates
- Sparse search: 50 candidates
- Total unique candidates: ~70 (overlap between methods)

**Sample Dense Result**:
```python
{
    "chunk_id": "LISINOPRIL_v23_34084-4_chunk_002",
    "raw_text": "The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough.",
    "metadata": {
        "drug_name": "Lisinopril",
        "rxcui": "203644",
        "loinc_section": "ADVERSE REACTIONS",
        "set_id": "a1b2c3d4-e5f6-7890-a1b2-c3d4e5f67890"
    },
    "score": 0.876  # Cosine similarity
}
```

---

## STEP 7: Reciprocal Rank Fusion (RRF)

**Purpose**: Combine dense and sparse results

**Algorithm**:
```python
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    scores = {}
    
    # Score dense results
    for rank, result in enumerate(dense_results):
        chunk_id = result['chunk_id']
        scores[chunk_id] = 1 / (k + rank + 1)
    
    # Add sparse results
    for rank, result in enumerate(sparse_results):
        chunk_id = result['chunk_id']
        scores[chunk_id] += 1 / (k + rank + 1)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Example Scores**:
```
LISINOPRIL_v23_34084-4_chunk_002: RRF=0.0328 (rank 1 dense, rank 3 sparse)
LISINOPRIL_v23_34084-4_chunk_005: RRF=0.0274 (rank 5 dense, rank 1 sparse)
LISINOPRIL_v23_34084-4_chunk_001: RRF=0.0262 (rank 2 dense, rank 8 sparse)
...
```

**Output**: Top 50 candidates ranked by RRF score

---

## STEP 8: Cross-Encoder Re-Ranking

**Purpose**: Fine-grained relevance scoring

**Model**: `cross-encoder/ms-marco-MiniLM-L-12-v2`

**Process**:
```python
reranker = CrossEncoderReranker()
pairs = [
    [query, chunk['raw_text']]
    for chunk in top_50_candidates
]
scores = reranker.predict(pairs)
```

**Re-Ranking Scores**:
```
LISINOPRIL_v23_34084-4_chunk_002: 9.23  (highest)
LISINOPRIL_v23_34084-4_chunk_005: 7.84
LISINOPRIL_v23_34084-4_chunk_008: 6.91
LISINOPRIL_v23_34084-4_chunk_001: 6.45
LISINOPRIL_v23_34084-4_chunk_013: 5.92
...
```

**Output**: Top 5 highest-scoring chunks

---

## STEP 9: Context Assembly

**Selected Chunks** (top 5):

**Chunk 1**:
```
[ADVERSE REACTIONS section]
The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough.
```

**Chunk 2**:
```
[ADVERSE REACTIONS section]
In controlled clinical trials, the following adverse reactions occurred in ≥1% of patients: 
hypotension (1.2%), rash (1.5%), and chest pain (1.3%).
```

**Chunk 3**:
```
[ADVERSE REACTIONS section]
Other adverse reactions reported include: angioedema, muscle cramps, nausea, and diarrhea.
```

**Chunk 4**:
```
[ADVERSE REACTIONS section]
Discontinuation of therapy due to adverse reactions occurred in approximately 6% of patients.
```

**Chunk 5**:
```
[ADVERSE REACTIONS section]
The most serious adverse reactions include angioedema and hypotension.
```

**Context for LLM**:
```
[Chunk 1 - ADVERSE REACTIONS]
The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough.

[Chunk 2 - ADVERSE REACTIONS]
In controlled clinical trials, the following adverse reactions occurred in ≥1% of patients: 
hypotension (1.2%), rash (1.5%), and chest pain (1.3%).

[Chunk 3 - ADVERSE REACTIONS]
Other adverse reactions reported include: angioedema, muscle cramps, nausea, and diarrhea.

[Chunk 4 - ADVERSE REACTIONS]
Discontinuation of therapy due to adverse reactions occurred in approximately 6% of patients.

[Chunk 5 - ADVERSE REACTIONS]
The most serious adverse reactions include angioedema and hypotension.
```

---

## STEP 10: Constrained Extraction (LLM)

**System Prompt**: (See `generation/constrained_extractor.py` for full prompt)
```
You are a PHARMACEUTICAL INFORMATION EXTRACTION SYSTEM...
You may ONLY output text that appears WORD-FOR-WORD in the provided context...
```

**User Prompt**:
```
QUERY: What are the adverse reactions of Lisinopril?

RETRIEVED CONTEXT:
[5 chunks as shown above]

INSTRUCTIONS:
Extract the answer using ONLY the text above.
Copy the relevant text verbatim.
```

**LLM Call** (GPT-4o):
```python
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": RUNTIME_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.0,
    max_tokens=2000,
    top_p=0.1
)
```

**LLM Response**:
```
The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough. 
In controlled clinical trials, the following adverse reactions occurred in ≥1% of patients: 
hypotension (1.2%), rash (1.5%), and chest pain (1.3%). Other adverse reactions reported 
include: angioedema, muscle cramps, nausea, and diarrhea. The most serious adverse 
reactions include angioedema and hypotension.

[Source: ADVERSE REACTIONS section]
```

---

## STEP 11: Post-Generation Validation

**Purpose**: Ensure output is verbatim (no hallucination)

**Validation Method**: RapidFuzz partial string matching

**Process**:
```python
validator = PostGenerationValidator(similarity_threshold=95)

# Clean answer (remove source attribution)
clean_answer = answer.split("[Source:")[0].strip()

# Check against each chunk
max_similarity = 0
for chunk in retrieved_chunks:
    score = fuzz.partial_ratio(clean_answer, chunk['raw_text'])
    max_similarity = max(max_similarity, score)
```

**Validation Scores**:
```
vs Chunk 1: 98.5%  ✓
vs Chunk 2: 97.2%  ✓
vs Chunk 3: 96.8%  ✓
vs Chunk 4: 91.3%
vs Chunk 5: 95.6%  ✓

Maximum: 98.5%
```

**Threshold**: 95%

**Result**: ✅ **VALIDATION PASSED** (98.5% ≥ 95%)

---

## STEP 12: Response Assembly

**Final Response Structure**:
```json
{
    "answer": "The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough. In controlled clinical trials, the following adverse reactions occurred in ≥1% of patients: hypotension (1.2%), rash (1.5%), and chest pain (1.3%). Other adverse reactions reported include: angioedema, muscle cramps, nausea, and diarrhea. The most serious adverse reactions include angioedema and hypotension.\n\n[Source: ADVERSE REACTIONS section]",
    
    "status": "validated",
    
    "metadata": {
        "drug_name": "Lisinopril",
        "rxcui": "203644",
        "set_id": "a1b2c3d4-e5f6-7890-a1b2-c3d4e5f67890",
        "root_id": "x9y8z7w6-v5u4-3210-x9y8-z7w6v5u43210",
        "version": "23",
        "effective_date": "20240115",
        "last_updated": "January 15, 2024",
        "loinc_section": "ADVERSE REACTIONS",
        "loinc_code": "34084-4",
        "source_chunk_ids": [
            "LISINOPRIL_v23_34084-4_chunk_002",
            "LISINOPRIL_v23_34084-4_chunk_005",
            "LISINOPRIL_v23_34084-4_chunk_008"
        ],
        "num_chunks_used": 5,
        "prompt_tokens": 1250,
        "completion_tokens": 145
    },
    
    "validation_score": 98.5,
    
    "timestamp": "2026-02-06T11:00:00Z"
}
```

---

## STEP 13: Audit Logging

**Write to `pharma_qa_audit.jsonl`**:
```json
{
    "timestamp": "2026-02-06T11:00:00Z",
    "query": "What are the adverse reactions of Lisinopril?",
    "intent": "product_specific",
    "answer": "The most common adverse reactions (≥2%) are dizziness...",
    "status": "validated",
    "validation_score": 98.5,
    "metadata": {
        "drug_name": "Lisinopril",
        "rxcui": "203644",
        "set_id": "a1b2c3d4-e5f6-7890-a1b2-c3d4e5f67890",
        "loinc_section": "ADVERSE REACTIONS"
    },
    "user_id": null,
    "session_id": null
}
```

---

## STEP 14: User Display

**Formatted Output**:

```
📋 Answer:
The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough. 
In controlled clinical trials, the following adverse reactions occurred in ≥1% of patients: 
hypotension (1.2%), rash (1.5%), and chest pain (1.3%). Other adverse reactions reported 
include: angioedema, muscle cramps, nausea, and diarrhea. The most serious adverse 
reactions include angioedema and hypotension.

📊 Source Information:
• Drug: Lisinopril
• Section: ADVERSE REACTIONS
• Last Updated: January 15, 2024
• Document ID: a1b2c3d4-e5f6-7890-a1b2-c3d4e5f67890
• Validation Score: 98.5%

⚠️ Disclaimer: This information is extracted directly from FDA-approved labeling. 
It is not medical advice. Consult a healthcare provider for patient-specific guidance.
```

---

## Summary of Safety Controls

✅ **Intent Classification** → Blocked out-of-scope queries  
✅ **Drug Normalization** → Standardized to RxCUI  
✅ **Metadata Filtering** → Pre-filtered to correct drug/section  
✅ **Hybrid Search** → Semantic + lexical precision  
✅ **Cross-Encoder** → High-confidence candidate selection  
✅ **Constrained Prompt** → LLM instructed to extract verbatim only  
✅ **Post-Validation** → 95%+ similarity required  
✅ **Audit Logging** → Full traceability  

**Result**: Verbatim answer with zero hallucination

---

## Failure Example: Validation Rejection

**What if LLM paraphrased?**

**LLM Output** (hypothetical):
```
Patients may experience dizziness, headaches, tiredness, or a persistent cough.
```

**Validation Check**:
```python
clean_answer = "patients may experience dizziness, headaches, tiredness, or a persistent cough"
chunk_1 = "the most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough"

similarity = fuzz.partial_ratio(clean_answer, chunk_1)
# Result: 72.3%
```

**Threshold**: 95%

**Result**: ❌ **VALIDATION FAILED** (72.3% < 95%)

**System Response**:
```json
{
    "answer": "Unable to extract verbatim answer from available evidence.",
    "status": "rejected",
    "reason": "validation_failed",
    "validation_score": 72.3,
    "rejected_answer": "Patients may experience dizziness, headaches..."
}
```

**Output to User**:
```
Unable to extract verbatim answer from available evidence.
```

**Philosophy**: Better to refuse than to hallucinate.

---

## Total Processing Time (Estimated)

| Step | Time |
|------|------|
| Intent classification | ~50ms |
| Drug normalization (RxNorm API) | ~200ms |
| Section classification | ~10ms |
| Query embedding | ~100ms |
| Hybrid search | ~150ms |
| Cross-encoder reranking | ~300ms |
| LLM extraction | ~2000ms |
| Validation | ~50ms |
| Audit logging | ~20ms |
| **TOTAL** | **~2.9 seconds** |

Performance can be optimized with caching and batching.

---

## End of Flow

This completes the journey from user question to validated, traceable answer.

Every step enforces the principle:  
**This system retrieves truth. It does not generate intelligence.**
