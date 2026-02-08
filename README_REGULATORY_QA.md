# Regulatory-Grade Pharmaceutical QA Chatbot

## Overview

This is a **REGULATORY-GRADE, ZERO-HALLUCINATION** pharmaceutical QA chatbot designed for verbatim extraction from FDA Structured Product Labeling (SPL) XML documents.

**System Classification**: Evidence-Retrieval System (Non-Clinical, Non-SaMD)

## Key Features

✅ **Zero Hallucination Tolerance** - Verbatim extraction only, never paraphrases  
✅ **Post-Generation Validation** - 95%+ lexical similarity required  
✅ **Full Traceability** - Every answer linked to SetID, RootID, LOINC codes  
✅ **Hybrid Search** - Dense semantic + sparse lexical retrieval  
✅ **Cross-Encoder Reranking** - High-precision candidate selection  
✅ **RxNorm/RxClass Integration** - Drug normalization and class expansion  
✅ **Self-Hosted Infrastructure** - Qdrant vector DB (no managed services)  
✅ **Audit Logging** - Full query history for compliance  
✅ **Deterministic Refusals** - Refuses to answer when evidence unavailable  

## Architecture

```
Query → Intent Classification → Drug Normalization → Hybrid Retrieval 
  → Cross-Encoder Reranking → Constrained Extraction (LLM) 
  → Post-Generation Validation → Validated Answer + Metadata
```

See `REGULATORY_QA_CHATBOT_ARCHITECTURE.md` for complete system design.

## Installation

### 1. Prerequisites

- Python 3.9+
- Docker and Docker Compose (for Qdrant)
- 8GB+ RAM
- Azure OpenAI credentials

### 2. Install Python Dependencies

```bash
pip install -r requirements_regulatory.txt
```

### 3. Start Qdrant Vector Database

```bash
docker-compose up -d
```

Verify Qdrant is running:
```bash
curl http://localhost:6333/health
```

### 4. Set Environment Variables

Create `.env` file:
```bash
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

### 5. Download Embedding Models

Models will auto-download on first run:
- `pritamdeka/S-PubMedBert-MS-MARCO` (dense embeddings)
- `cross-encoder/ms-marco-MiniLM-L-12-v2` (reranker)

## Data Ingestion Pipeline

### Step 1: Parse SPL XML Files

```python
from ingestion.spl_xml_parser import SPLXMLParser

parser = SPLXMLParser()
metadata, sections = parser.parse_document('path/to/lisinopril_spl.xml')
```

### Step 2: Chunk Documents

```python
from ingestion.chunking_strategy import DualChunker

chunker = DualChunker(chunk_size=512, overlap=50)
chunks = chunker.chunk_document(metadata, sections)
```

### Step 3: Generate Embeddings

```python
from retrieval.hybrid_retriever import DenseEmbedder

embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
embeddings = embedder.embed([chunk.semantic_text for chunk in chunks])
```

### Step 4: Upsert to Qdrant

```python
from vector_db.qdrant_manager import QdrantManager

qm = QdrantManager(host="localhost", port=6333)
qm.create_collection(dense_vector_size=768)
qm.upsert_chunks(
    chunks=[chunk.to_dict() for chunk in chunks],
    dense_embeddings=embeddings,
    sparse_embeddings=[]  # Add BM25 if needed
)
```

## Query Processing

### Complete End-to-End Query

```python
from orchestrator.qa_orchestrator import RegulatoryQAOrchestrator

# Initialize orchestrator (see orchestrator/qa_orchestrator.py for full setup)
orchestrator = RegulatoryQAOrchestrator(
    drug_normalizer=normalizer,
    retriever=retriever,
    generator=generator
)

# Process query
result = orchestrator.query("What are the adverse reactions of Lisinopril?")

print(f"Answer: {result['answer']}")
print(f"Status: {result['status']}")
print(f"Validation: {result['validation_score']:.1f}%")
print(f"Source: {result['metadata']['loinc_section']}")
```

### Response Format

```json
{
  "answer": "The most common adverse reactions (≥2%) are dizziness, headache, fatigue, and cough.",
  "status": "validated",
  "metadata": {
    "drug_name": "Lisinopril",
    "rxcui": "203644",
    "set_id": "abc-123-def",
    "loinc_section": "ADVERSE REACTIONS",
    "effective_date": "20240115",
    "validation_score": 98.5
  },
  "timestamp": "2026-02-06T11:00:00Z"
}
```

## Validation & Compliance

### RAGAS Evaluation

```python
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision

results = evaluate(
    eval_dataset,
    metrics=[faithfulness, context_precision]
)

# Target thresholds
assert results['faithfulness'] >= 0.95
assert results['context_precision'] >= 0.90
```

### Audit Log

All queries are logged to `pharma_qa_audit.jsonl`:

```json
{
  "timestamp": "2026-02-06T11:00:00Z",
  "query": "What are adverse reactions of Lisinopril?",
  "answer": "The most common adverse reactions...",
  "status": "validated",
  "validation_score": 98.5,
  "metadata": {...}
}
```

## Refusal Logic

The system refuses to answer in these scenarios:

| Scenario | Response |
|----------|----------|
| Drug not in database | "No FDA labeling found for '{drug}'." |
| Section not found | "The {section} section was not found." |
| Low confidence | "Evidence not found in source document." |
| Validation failed (<95%) | "Unable to extract verbatim answer." |
| Medical advice | "This system provides FDA labeling excerpts only, not medical advice." |

## Testing

### Unit Tests

```bash
pytest tests/test_extraction.py -v
pytest tests/test_validation.py -v
```

### Integration Tests

```bash
pytest tests/test_end_to_end.py -v
```

## Deployment Checklist

- [ ] Qdrant container running with persistent volumes
- [ ] S-PubMedBert model downloaded and cached
- [ ] Cross-encoder reranker downloaded
- [ ] RxNorm/RxClass API connectivity verified
- [ ] SPL XML ingestion pipeline tested
- [ ] Metadata indices created in Qdrant
- [ ] System prompt version-controlled
- [ ] Validation threshold configured (≥95%)
- [ ] Audit logging enabled
- [ ] RAGAS baseline established
- [ ] Refusal scenarios tested
- [ ] UI disclaimer approved

## System Prompt

The runtime system prompt enforces zero-hallucination constraints. See `generation/constrained_extractor.py` for the complete prompt.

Key rules:
1. Output ONLY verbatim text from context
2. Never paraphrase, summarize, or infer
3. Refuse to answer if uncertain
4. Never provide medical advice
5. Always attribute source section

## Directory Structure

```
solomind US/
├── REGULATORY_QA_CHATBOT_ARCHITECTURE.md  # Complete system design
├── requirements_regulatory.txt             # Python dependencies
├── docker-compose.yml                      # Qdrant deployment
├── ingestion/
│   ├── spl_xml_parser.py                  # SPL XML parser
│   └── chunking_strategy.py               # Dual-chunking
├── normalization/
│   └── rxnorm_integration.py              # Drug normalization
├── vector_db/
│   └── qdrant_manager.py                  # Vector DB interface
├── retrieval/
│   └── hybrid_retriever.py                # Dense + sparse search
├── generation/
│   └── constrained_extractor.py           # LLM + validation
└── orchestrator/
    └── qa_orchestrator.py                 # End-to-end pipeline
```

## Regulatory Compliance

**Purpose**: Information retrieval from FDA-approved labeling

**NOT Intended For**:
- Medical diagnosis
- Treatment recommendations
- Patient-specific dosing
- Replacing clinical judgment

**Data Provenance**: All outputs traceable to FDA SPL XML via SetID, RootID, LOINC codes

**Validation**: Automated lexical validation (≥95% similarity)

**Failure Mode**: System refuses rather than guesses

## Support

For questions about implementation, see `REGULATORY_QA_CHATBOT_ARCHITECTURE.md`.

---

**Remember**: This system retrieves truth, it does not generate intelligence.  
Silence is preferable to guessing. Refusal is preferable to hallucination.
