# ⚕️ Solomind.ai — Regulatory-Grade Pharmaceutical QA Chatbot

> **Zero-hallucination. Verbatim extraction only. Every answer traceable to an FDA source.**

A production-grade Retrieval-Augmented Generation (RAG) system that answers pharmaceutical questions by extracting **word-for-word text** from over 15,000 FDA Structured Product Labeling (SPL) XML documents. Built for pharmacists and medical regulatory professionals who cannot tolerate paraphrased or generated medical information.

---

## 🏗️ System Architecture

```
                          ┌──────────────────────┐
                          │   FDA SPL XML Files  │
                          │   (data/xml/*.xml)   │
                          └──────────┬───────────┘
                                     │ scripts/ingest_v3.py
                                     ▼
                          ┌──────────────────────┐
                          │   Ingestion Pipeline │
                          │  SPLXMLParser →      │
                          │  HierarchicalChunker │
                          │  DenseEmbedder +     │
                          │  SparseEmbedder      │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │     Qdrant Cloud     │
                          │  spl_children (RAG)  │
                          │  spl_parents (Truth) │
                          └──────────┬───────────┘
                                     │
┌───────────┐    POST /chat   ┌──────▼────────────┐   Azure OpenAI
│  Frontend │ ◄─────────────► │  FastAPI Backend  │ ◄────────────►
│ (Next.js) │                 │  backend_server.py│   gpt-4o-agent
└───────────┘                 └──────┬────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  QA Orchestrator     │
                          │  Intent → Entity →   │
                          │  Section → Retrieve  │
                          │  → Extract → Validate│
                          └──────────────────────┘
```

### End-to-End Request Lifecycle

1. User sends `POST /chat` with a natural language question
2. **IntentClassifier** gates the query (Regex Tier 1 → LLM Arbiter Tier 2)
3. **EntityValidator** confirms a known drug name exists in the query against the Qdrant index
4. **SectionClassifier** maps query keywords to a LOINC section code (e.g., "adverse reactions" → `34084-4`)
5. **HybridRetriever** runs dense + sparse search on `spl_children`, applies Section-First reranking (3 tiers)
6. **CrossEncoderReranker** scores top candidates; parent paragraphs are fetched from `spl_parents`
7. **ConstrainedExtractor** routes by query shape (FACT/LIST/MANAGEMENT) — LLM only used for ambiguous queries
8. **PostGenerationValidator** verifies ≥95% fuzzy similarity between answer and source chunk
9. Validated answer returned with full source citation (LOINC section, SetID, RxCUI)

---

## 📁 Project Structure

```
solomindUS/
│
├── backend_server.py              # FastAPI app — single /chat endpoint
├── docker-compose.yml             # Local Qdrant container (v1.16.0)
├── requirements_regulatory.txt    # All Python dependencies
├── .env                           # Environment secrets (Azure, Qdrant)
│
├── ingestion/                     # SPL XML → Vector DB pipeline
│   ├── spl_xml_parser.py          # HL7 v3 XML parser + 101-code LOINC map
│   ├── hierarchical_chunking.py   # Parent (paragraphs) + Child (sentences) chunks
│   ├── chunking_strategy.py       # Alternative chunking strategy
│   └── table_preserver.py         # FDA XSLT → HTML → Markdown table pipeline
│
├── normalization/
│   └── rxnorm_integration.py      # RxNormClient, RxClassClient, DrugNormalizer
│
├── vector_db/
│   ├── qdrant_manager.py          # QdrantManager (flat collection interface)
│   └── hierarchical_qdrant.py     # HierarchicalQdrantManager (parent+child)
│
├── retrieval/
│   └── hybrid_retriever.py        # DenseEmbedder, SparseEmbedder,
│                                  # CrossEncoderReranker, HybridRetriever
│
├── orchestrator/
│   ├── qa_orchestrator.py         # IntentClassifier, SectionClassifier,
│   │                              # RegulatoryQAOrchestrator (main pipeline)
│   ├── entity_validator.py        # Pre-retrieval drug entity validation
│   ├── section_intent_normalizer.py # Section boost + synonym normalization
│   └── hierarchical_conflict_resolver.py # Deduplication & conflict resolution
│
├── generation/
│   ├── constrained_extractor.py   # ConstrainedExtractor, PostGenerationValidator,
│   │                              # RegulatoryQAGenerator
│   └── extractive_system.py       # Alternate extractive system
│
├── config/
│   └── section_intent_map.py      # 22 canonical section intent groups + synonyms
│
├── scripts/                       # Ingestion scripts
│   └── ingest_v3.py               # PRIMARY ingestion script
│
├── frontend/                      # Next.js chat UI
│   ├── app/                       # App router pages
│   └── components/                # React components
│
├── data/
│   └── xml/                       # FDA SPL XML files (place here)
│
├── pharma_qa_audit.jsonl          # Audit log — every query + outcome
└── ingestion_failures.log         # Failed file log during ingestion
```

---

## 🧠 Core Components Deep-Dive

### 1. SPL XML Parser (`ingestion/spl_xml_parser.py`)

Parses HL7 v3 XML (FDA's SPL format) using `lxml` with full namespace support.

**What it extracts:**
- `SPLMetadata`: `set_id`, `root_id`, `version_number`, `effective_time`, `drug_name`, `ndc_codes`
- `SPLSection`: `loinc_code`, `section_name`, `text_content`, `html_content`, `is_table`

**LOINC mapping:** Contains all 101 official FDA SPL LOINC codes, e.g.:
| LOINC Code | Section |
|---|---|
| `34084-4` | ADVERSE REACTIONS |
| `34070-3` | CONTRAINDICATIONS |
| `34068-7` | DOSAGE & ADMINISTRATION |
| `43685-7` | WARNINGS AND PRECAUTIONS |
| `34088-5` | OVERDOSAGE |
| `34067-9` | INDICATIONS & USAGE |

**Table preservation pipeline:** `XML → FDA XSLT (spl.xsl) → HTML → Markdown` via `html2text`. Tables are treated as atomic parent chunks with no child splitting.

---

### 2. Hierarchical Chunking (`ingestion/hierarchical_chunking.py`)

The core data model enforcing word-for-word accuracy.

```
SPL Section
    │
    ├── ParentChunk (full paragraph — SOURCE OF TRUTH, never modified)
    │       raw_text, loinc_code, loinc_section, drug_name, rxcui,
    │       set_id, root_id, version, effective_date, is_table, ndc
    │
    └── ChildChunk (individual sentence — SEARCH INDEX ONLY)
            sentence_text = "Drug: X. Section: Y. <sentence>"
            parent_id → points back to ParentChunk
```

**ID generation** (deterministic, collision-free):
- Parent: `{DRUG}_v{version}_{loinc}_sec_{idx:03d}_para_{idx:03d}`
- Child: `{parent_id}_sent_{idx:03d}`

**Splitting rules:**
- Paragraphs: split on `\n\n`
- Sentences: regex `(?<=[.!?])\s+(?=[A-Z])`, minimum 10 chars
- Tables: **atomic** — entire table is one parent, no children

---

### 3. Drug Normalization (`normalization/rxnorm_integration.py`)

Three-class system interfacing with NLM's free RxNav API (no auth required):

| Class | Purpose |
|---|---|
| `RxNormClient` | Name → RxCUI, NDC → RxCUI, RxCUI → Name. 10,000-entry LRU cache. |
| `RxClassClient` | Drug class lookup (MESHPA/PE/EPC), class expansion to RxCUI lists |
| `DrugNormalizer` | High-level facade combining both clients |

**Fallback strategy:** Exact match → Approximate match (`/approximateTerm.json`)

---

### 4. Vector Database (`vector_db/`)

**Two Qdrant collections:**

| Collection | Content | Vector Config |
|---|---|---|
| `spl_children` | Child chunks (sentences) | Dense 768-dim (Cosine) + Sparse BM25 |
| `spl_parents` | Parent chunks (paragraphs) | Dummy size=1 (payload-only retrieval) |

**Payload indices** (for fast filter-based search):
- Children: `drug_name`, `rxcui`, `loinc_code`, `loinc_section`, `parent_id`
- Parents: `parent_id`, `drug_name`, `set_id`

**Client:** `QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)` — Cloud or local.

---

### 5. Hybrid Retriever (`retrieval/hybrid_retriever.py`)

**Dense Embedder:** `pritamdeka/S-PubMedBert-MS-MARCO` — 768-dim biomedical embeddings, cosine similarity, normalized.

**Sparse Embedder:** BM25-style term frequency with `zlib.crc32` stable hashing for Qdrant sparse vectors.

**Cross-Encoder Reranker:** `cross-encoder/ms-marco-MiniLM-L-12-v2` — scores query-document pairs, `top_k=15`.

**Section-First Reranking (3 tiers):**
1. **Tier 1** — Exact LOINC code match from parsed XML
2. **Tier 2** — Section name synonym match via `SECTION_INTENT_MAP` (handles old XMLs with free-text titles)
3. **Tier 3** — Full candidate set fallback (no section detected)

**Parent fetch strategy:**
- Accumulates **sum of child rerank scores** per parent (stable across query paraphrases)
- Sorts by: child count ↓, score sum ↓, parent_id ↑ (alphabetic tiebreaker)
- Post-fetch **section guard** drops parents from wrong LOINC sections
- Returns single best parent's `raw_text` as the canonical answer source

**Safety contract:** Never falls back to cross-drug retrieval. If drug filter returns 0 results, the system refuses — it does not broaden the search.

---

### 6. QA Orchestrator (`orchestrator/qa_orchestrator.py`)

**IntentClassifier** — Hybrid guardrail system:
- **Tier 1 Regex (fast):** Blocks obvious medical advice ("should I take", "prescribe me") or allows obvious labeling queries ("adverse reactions", "contraindications")
- **Tier 2 LLM Arbiter:** For ambiguous queries, calls `gpt-4o` with a binary LABELING/ADVICE prompt (`max_tokens=10, temperature=0.0`)
- Routes to: `product_specific`, `class_based`, `comparative` (comparative currently refused)

**SectionClassifier** — Maps 80+ keyword phrases to LOINC codes. Longer keywords take priority (e.g., "overdosage" before "dosage"). Returns `None` for fact-level queries (half-life, generic name) which are handled by SECTION_INTENT_MAP instead.

**EntityValidator** — Loads all drug names from Qdrant at startup into a dict. O(1) case-insensitive lookup. Falls back to token-intersection scoring for complex FDA label names (e.g., "DR SCHOLLS TOLNAFTATE ANTIFUNGAL").

**HierarchicalConflictResolver** — Deduplicates chunks with ≥95% similarity before sending to extractor.

---

### 7. Constrained Extractor (`generation/constrained_extractor.py`)

**Query shape routing (pure Python, no LLM):**

| Shape | Trigger keywords | Extraction method |
|---|---|---|
| `FACT` | "what is", "generic name", "strength", "ndc" | Shortest verbatim span containing query tokens |
| `LIST` | "what are", "adverse reaction", "contraindication" | Full paragraph verbatim |
| `MANAGEMENT` | "treat", "overdose", "how to", "monitoring" | Full paragraph verbatim |

**Section-specific bypass:** When a LOINC section is detected, the LLM is **skipped entirely**. The retriever's best parent paragraph is returned directly — zero API calls.

**LLM path (general queries only):** GPT-4o acts as a **SENTENCE LOCATOR** — it outputs JSON `{"indices": [0, 1, 2]}` pointing to which retrieved sentences answer the question. The raw text is then extracted at those indices. The LLM never generates or paraphrases.

**System prompt key constraints:**
```
"You do NOT write or generate text"
"You ONLY identify which existing sentences answer the question"
"Output format: A JSON object with 'indices' (list of integers)"
```

---

### 8. Post-Generation Validator (`generation/constrained_extractor.py`)

Uses `rapidfuzz.fuzz.partial_ratio` to verify the extracted answer appears verbatim in the source chunks.

- **Threshold:** 95% similarity (configurable, set to 75% in backend for fallback robustness)
- **Refusal phrases** are always auto-validated (score = 100%)
- Source attribution `[Source: SECTION_NAME]` is stripped before comparison

---

### 9. Section Intent Map (`config/section_intent_map.py`)

22 canonical intent groups covering 80+ synonym phrases:

```python
"PHARMACOKINETICS": ["pharmacokinetics", "absorption", "distribution",
                     "metabolism", "elimination", "half-life", "bioavailability"]

"CLINICAL_PHARMACOLOGY": ["clinical pharmacology", "mechanism of action",
                          "pharmacodynamics", "mode of action", "action"]
```

Used by the retriever's Tier 2 section matching to handle legacy FDA XMLs that store sections under `42229-5` (SPL UNCLASSIFIED) with free-text English titles.

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.9+
- Docker Desktop (for local Qdrant)
- Azure OpenAI account with `gpt-4o` deployment
- Qdrant Cloud account (or local Docker)

### 1. Clone & Install Dependencies

```bash
git clone <repo-url>
cd solomindUS
pip install -r requirements_regulatory.txt
```

### 2. Configure Environment Variables

Create/edit `.env` in the project root:

```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o

# Qdrant (Cloud)
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

> ⚠️ **Never commit `.env` to version control.** It contains live API keys.

### 3. Start Qdrant (Local Alternative)

```bash
docker-compose up -d

# Verify health
curl http://localhost:6333/health
```

For local Qdrant, set in `.env`:
```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # leave blank for local
```

### 4. Download Embedding Models

Models download automatically on first run from HuggingFace:
- `pritamdeka/S-PubMedBert-MS-MARCO` (~500MB)
- `cross-encoder/ms-marco-MiniLM-L-12-v2` (~120MB)

---

## 📥 Data Ingestion

### Prepare FDA SPL XML Files

Place FDA SPL XML files in `data/xml/`:
```
data/
└── xml/
    ├── lisinopril_spl.xml
    ├── metformin_spl.xml
    └── ... (15,000+ files)
```

FDA SPL files can be downloaded from the [FDA DailyMed database](https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm).

### Run Ingestion

```bash
python scripts/ingest_v3.py
```

**What happens:**
1. Reads all `.xml` from `data/xml/`
2. Checks `spl_parents` for already-ingested `set_id`s (skip duplicates)
3. For each new file: parse → chunk → RxNorm enrich → embed → upsert
4. Failed files logged to `ingestion_failures.log`

**Ingestion stats output:**
```
=== INGESTION COMPLETE ===
Stats: {'total': 500, 'skipped': 420, 'success': 78, 'failed': 2}
```

### Verify Ingestion

```bash
python scripts/check_db_stats.py        # Collection counts
python scripts/count_unique_drugs.py    # Unique drug names indexed
python scripts/check_adverse.py         # Test adverse reaction query
```

---

## 🖥️ Running the Backend

```bash
python backend_server.py
# Server starts on http://0.0.0.0:8000
```

### API Endpoint

**`POST /chat`**

Request:
```json
{
  "query": "What are the adverse reactions of Lisinopril?",
  "user_id": "pharmacist_001",
  "session_id": "session_abc"
}
```

Response:
```json
{
  "answer": "The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough.\n\n[Source: ADVERSE REACTIONS SECTION]",
  "status": "extracted",
  "metadata": {
    "drug_name": "Lisinopril",
    "rxcui": "203644",
    "set_id": "abc-123-def",
    "loinc_section": "ADVERSE REACTIONS SECTION",
    "loinc_code": "34084-4",
    "extraction_mode": "full_paragraph",
    "query_shape": "LIST",
    "llm_used": false
  },
  "timestamp": "2025-04-30T12:00:00Z"
}
```

### Refusal Responses

| Scenario | `status` | Example answer |
|---|---|---|
| Drug not in database | `refused` | `"No FDA labeling found for 'XYZ'."` |
| Medical advice detected | `out_of_scope` | `"This system provides FDA labeling excerpts only, not medical advice."` |
| Section not found | `refused` | `"The OVERDOSAGE section was not found for Aspirin."` |
| Evidence not found | `refused` | `"Evidence not found in source document."` |
| No drug specified | `refused` | `"No drug specified. Please provide the drug name."` |

---

## 🌐 Frontend

The `frontend/` directory contains a **Next.js** application.

```bash
cd frontend
npm install
npm run dev
# Runs on http://localhost:3000
```

The frontend connects to the backend at `http://localhost:8000/chat` (configurable).



---

## 📊 Audit Logging

Every query is appended to `pharma_qa_audit.jsonl`:

```json
{
  "timestamp": "2025-04-30T12:00:00Z",
  "query": "What are adverse reactions of Lisinopril?",
  "answer": "The most common adverse reactions...",
  "status": "extracted",
  "intent": "product_specific",
  "metadata": {
    "drug_name": "Lisinopril",
    "loinc_section": "ADVERSE REACTIONS SECTION"
  }
}
```

---

## 🔐 Regulatory Compliance

**This system is classified as:** Evidence Retrieval Tool (Non-Clinical, Non-SaMD)

**NOT intended for:**
- Medical diagnosis or treatment recommendations
- Patient-specific dosing decisions
- Replacing clinical judgment

**Data provenance:** Every answer is traceable to:
- `set_id` — Document family identifier
- `root_id` — Specific version identifier  
- `loinc_code` — Standardized section code
- `parent_id` — Exact paragraph within the document

**Failure mode:** The system refuses rather than guesses. Silence is preferable to hallucination.

---

## ⚙️ Key Configuration Parameters

| Parameter | Location | Default | Purpose |
|---|---|---|---|
| `retrieval_limit` | `qa_orchestrator.py` | `75` | Max candidates from Qdrant |
| `rerank_top_k` | `qa_orchestrator.py` | `15` | Candidates sent to cross-encoder |
| `similarity_threshold` | `backend_server.py` | `75` | Post-gen validation threshold |
| `section_boost_weight` | `section_intent_map.py` | `0.15` | Section match score boost |
| `max_age` (tracker) | Qdrant scroll | `1000` | Batch size for drug name loading |

---

## 🧩 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | latest | REST API framework |
| `uvicorn` | latest | ASGI server |
| `qdrant-client` | ≥1.7.0 | Vector database client |
| `sentence-transformers` | ≥2.2.0 | Dense embeddings + cross-encoder |
| `rank-bm25` | ≥0.2.2 | Sparse BM25 embeddings |
| `openai` | ≥1.0.0 | Azure OpenAI (GPT-4o) |
| `rapidfuzz` | ≥3.0.0 | Post-generation fuzzy validation |
| `lxml` | ≥4.9.0 | FDA XML parsing |
| `html2text` | ≥2020.1.16 | HTML table → Markdown |
| `requests` | ≥2.28.0 | RxNorm/RxClass API calls |
| `torch` | ≥2.0.0 | PyTorch for model inference |
| `ragas` | ≥0.1.0 | RAG evaluation metrics |

---

## 🗺️ Development Workflow

### Add New Drug Data
```bash
# Place XML in data/xml/, then:
python scripts/ingest_v3.py

# Verify:
python scripts/count_unique_drugs.py
```

### Reset Database
```python
# In Python:
from vector_db.hierarchical_qdrant import HierarchicalQdrantManager
qm = HierarchicalQdrantManager()
qm.create_collections(dense_vector_size=768, recreate=True)
```

### Debug a Specific Query
```bash
python scripts/debug_retrieval_full.py
# Edit the query variable at the bottom of the script
```

### Monitor Logs
- `ingestion_failures.log` — Files that failed during ingestion
- `pharma_qa_audit.jsonl` — All queries and outcomes
- Console stdout — Real-time pipeline trace (intent → entity → section → retrieval → extraction)

---

*"This system retrieves truth, it does not generate intelligence. Silence is preferable to guessing. Refusal is preferable to hallucination."*
