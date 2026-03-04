"""
Updated Request Flow Diagram - With Production Fixes
"""

# ======================================================================
# REVISED PIPELINE FLOW
# ======================================================================

"""
┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Step 1: Intent Classification                          │
│                                                                     │
│  • Check for out-of-scope patterns (medical advice, patient-       │
│    specific, comparative)                                           │
│  • Route to: product_specific | class_based | out_of_scope         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Out of Scope? │
                    └─────────────────┘
                      Yes │       │ No
                          │       │
                          │       ▼
                          │  ┌─────────────────────────────────────────┐
                          │  │ ★ NEW: Step 2A: Entity Validation       │
                          │  │                                          │
                          │  │  EntityValidator.validate(query)         │
                          │  │                                          │
                          │  │  • Check if query contains explicit drug │
                          │  │  • Match against cached drug name set    │
                          │  │  • Case-insensitive substring matching   │
                          │  │                                          │
                          │  │  IF INVALID:                             │
                          │  │    → Return refusal                      │
                          │  │    → "No drug specified. Please provide  │
                          │  │       the drug name."                    │
                          │  │                                          │
                          │  │  IF VALID:                               │
                          │  │    → Proceed to retrieval                │
                          │  └─────────────────────────────────────────┘
                          │                │
                          │                ▼
                          │  ┌─────────────────────────────────────────┐
                          │  │     Step 2B: Drug Normalization          │
                          │  │                                          │
                          │  │  • Extract drug name from query          │
                          │  │  • Resolve to RxCUI via RxNorm           │
                          │  │  • Get drug metadata                     │
┌─────────────────┐       │  └─────────────────────────────────────────┘
│  Refuse & Log   │◄──────┤                │
└─────────────────┘       │                ▼
                          │  ┌─────────────────────────────────────────┐
                          │  │     Step 3: Section Classification       │
                          │  │                                          │
                          │  │  • Map query to LOINC code               │
                          │  │  • Determine section filter              │
                          │  └─────────────────────────────────────────┘
                          │                │
                          │                ▼
                          │  ┌─────────────────────────────────────────┐
                          │  │     Step 4: Hybrid Retrieval             │
                          │  │                                          │
                          │  │  WITH MANDATORY FILTER:                  │
                          │  │    filter_conditions = {                 │
                          │  │      "rxcui": <validated_rxcui>,         │
                          │  │      "loinc_code": <optional>            │
                          │  │    }                                     │
                          │  │                                          │
                          │  │  • Dense embedding (PubMedBERT)          │
                          │  │  • Sparse embedding (optional)           │
                          │  │  • Hybrid search in Qdrant               │
                          │  │  • Retrieve top 50 candidates            │
                          │  └─────────────────────────────────────────┘
                          │                │
                          │                ▼
                          │  ┌─────────────────────────────────────────┐
                          │  │     Step 5: Cross-Encoder Reranking      │
                          │  │                                          │
                          │  │  • Score all 50 candidates               │
                          │  │  • Sort by relevance score               │
                          │  │  • Select top 15                         │
                          │  │  • ADD RANK METADATA (0-14)              │
                          │  └─────────────────────────────────────────┘
                          │                │
                          │                ▼
                          │  ┌─────────────────────────────────────────┐
                          │  │ ★ NEW: Step 6: Hierarchical Conflict     │
                          │  │              Resolution                  │
                          │  │                                          │
                          │  │  HierarchicalConflictResolver.resolve()  │
                          │  │                                          │
                          │  │  FOR each drug group:                    │
                          │  │    FOR each chunk pair (i, j):           │
                          │  │      IF similarity(i.text, j.text) ≥95%: │
                          │  │        IF i.rank < j.rank:               │
                          │  │          REMOVE j                        │
                          │  │                                          │
                          │  │  • Preserves rank ordering               │
                          │  │  • Only compares same-drug chunks        │
                          │  │  • O(K²) for K=15 (fast)                 │
                          │  └─────────────────────────────────────────┘
                          │                │
                          │                ▼
                          │  ┌─────────────────────────────────────────┐
                          │  │     Step 7: Extractive Generation        │
                          │  │                                          │
                          │  │  • Pass deduplicated chunks to LLM       │
                          │  │  • Extract verbatim sentences            │
                          │  │  • NO DUPLICATION in final answer        │
                          │  └─────────────────────────────────────────┘
                          │                │
                          └────────────────┘
                                           │
                                           ▼
                          ┌─────────────────────────────────────┐
                          │     Step 8: Audit Logging            │
                          │                                      │
                          │  • Log query, intent, result         │
                          │  • Store in JSONL audit file         │
                          └─────────────────────────────────────┘
                                           │
                                           ▼
                          ┌─────────────────────────────────────┐
                          │        RETURN RESPONSE               │
                          │                                      │
                          │  {                                   │
                          │    "answer": "...",                  │
                          │    "status": "success",              │
                          │    "metadata": {...},                │
                          │    "validation_score": 95.0          │
                          │  }                                   │
                          └─────────────────────────────────────┘
"""

# ======================================================================
# KEY CHANGES FROM ORIGINAL FLOW
# ======================================================================

"""
ISSUE 2 FIX - Pre-Retrieval Entity Validation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BEFORE:
  Query → Intent → Drug Extraction → Retrieval (with fallback to unfiltered)

AFTER:
  Query → Intent → ★ Entity Validation ★ → Drug Extraction → Retrieval (always filtered)

New Behavior:
  • Query "What is the dosage?" → REFUSED (no drug entity)
  • Query "What brand name is this?" → REFUSED (no drug entity)
  • Unfiltered retrieval is NEVER allowed
  • Drug name must be explicitly present in query


ISSUE 1 FIX - Post-Rerank Conflict Resolution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BEFORE:
  Hybrid Search → Rerank → Extraction (may include duplicates)

AFTER:
  Hybrid Search → Rerank → ★ Conflict Resolution ★ → Extraction (deduplicated)

New Behavior:
  • Parent section text: "Common adverse reactions include nausea..."
  • Child subsection text: "Common adverse reactions include nausea..."
  • Before: Both chunks passed to LLM → duplicate lines in answer
  • After: Conflict resolver removes lower-ranked duplicate → single line in answer


UNCHANGED COMPONENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • Intent classification logic
  • Section classification (LOINC mapping)
  • Embedding models
  • Reranking model
  • Extractive LLM prompt
  • Validation logic
  • Audit logging
"""

# ======================================================================
# MODULE INTERACTION DIAGRAM
# ======================================================================

"""
┌────────────────────────────────────────────────────────────────────┐
│                   RegulatoryQAOrchestrator                         │
│                                                                    │
│  Attributes:                                                       │
│    • drug_normalizer: DrugNormalizer                              │
│    • retriever: HybridRetriever                                   │
│    • generator: RegulatoryQAGenerator                             │
│    • intent_classifier: IntentClassifier                          │
│    • section_classifier: SectionClassifier                        │
│    • entity_validator: EntityValidator ★ NEW ★                    │
│    • conflict_resolver: HierarchicalConflictResolver ★ NEW ★      │
│                                                                    │
│  Methods:                                                          │
│    query(query_str) → Result                                      │
│    └─► _handle_product_specific()                                │
│         ├─► entity_validator.validate()  ★ STEP 2A ★             │
│         ├─► _extract_drug_name()                                  │
│         ├─► normalizer.normalize_drug_name()                      │
│         ├─► section_classifier.classify()                         │
│         ├─► retriever.retrieve() [with mandatory filter]          │
│         ├─► conflict_resolver.resolve()  ★ STEP 6 ★              │
│         └─► generator.generate_answer()                           │
└────────────────────────────────────────────────────────────────────┘
           │                                     │
           ▼                                     ▼
┌─────────────────────────┐    ┌──────────────────────────────────┐
│   EntityValidator       │    │ HierarchicalConflictResolver     │
│                         │    │                                  │
│  Attributes:            │    │  Attributes:                     │
│    • vector_db          │    │    • similarity_threshold=95.0   │
│    • drug_names: Set    │    │                                  │
│                         │    │  Methods:                        │
│  Methods:               │    │    resolve(chunks) → filtered    │
│    validate(query)      │    │      ├─► Group by drug (rxcui)   │
│      → {valid, drug,    │    │      ├─► Compare within group    │
│         reason}         │    │      ├─► Remove if sim≥95%       │
│                         │    │      └─► Preserve rank order     │
│    refresh_drug_names() │    │                                  │
│      (call after        │    │                                  │
│       ingestion)        │    │                                  │
└─────────────────────────┘    └──────────────────────────────────┘
"""

# ======================================================================
# ERROR PATHS
# ======================================================================

"""
Scenario 1: Query without drug → Entity Validation Refusal
─────────────────────────────────────────────────────────────

  Query: "What is the dosage?"
  
  Flow:
    Intent Classifier → "product_specific"
    Entity Validator  → {"valid": False, "reason": "No drug specified..."}
    Orchestrator      → Create refusal response
    Result            → {"status": "refused", "answer": "No drug specified..."}
  
  RETRIEVAL NEVER EXECUTES ✓


Scenario 2: Duplicate chunks from hierarchy → Conflict Resolution
─────────────────────────────────────────────────────────────────

  Query: "What are the adverse reactions of Lisinopril?"
  
  Retrieval Results (after reranking):
    [
      {rank: 0, text: "Common adverse reactions include headache, dizziness, cough.", rxcui: "123"},
      {rank: 1, text: "Common adverse reactions include headache, dizziness, cough.", rxcui: "123"},  # 100% similar
      {rank: 2, text: "Serious reactions include angioedema.", rxcui: "123"}
    ]
  
  Conflict Resolver:
    Compare rank 0 vs rank 1: similarity=100% → REMOVE rank 1
    Compare rank 0 vs rank 2: similarity=30% → KEEP rank 2
  
  Output: [rank 0, rank 2]  ← Duplicate removed ✓
"""

# ======================================================================
# SCALABILITY NOTES
# ======================================================================

"""
EntityValidator Performance (20,000 drugs):
───────────────────────────────────────────

  • Drug names loaded once at startup
  • Stored in Python set() for O(1) lookup
  • Query validation: O(D) where D = number of drug names in set
  • With 20k drugs and query iteration: ~0.1ms per validation
  • Memory: ~2MB for 20k drug names (negligible)
  
  Bottleneck: Initial scroll through vector DB at startup
    → Acceptable (runs once, can be async)


HierarchicalConflictResolver Performance (K=50 max):
────────────────────────────────────────────────────

  • K = number of chunks after reranking (typically 15)
  • Complexity: O(K²) per drug group
  • With K=15: ~225 comparisons per drug
  • RapidFuzz ratio() is optimized in C: ~0.01ms per comparison
  • Total time: 225 * 0.01ms = 2.25ms
  
  Bottleneck: Text similarity computation
    → Acceptable for K ≤ 50
"""
