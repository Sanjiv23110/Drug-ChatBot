# Quick Start Implementation Guide
## Regulatory-Grade Pharmaceutical QA Chatbot

This guide walks you through implementing the complete system from scratch.

---

## Phase 1: Infrastructure Setup (Day 1)

### 1.1 Install System Dependencies

**Python Environment**:
```bash
# Create virtual environment
python -m venv venv_regulatory

# Activate (Windows)
.\venv_regulatory\Scripts\activate

# Activate (Linux/Mac)
source venv_regulatory/bin/activate

# Install dependencies
pip install -r requirements_regulatory.txt
```

**Docker Installation** (for Qdrant):
- Windows: Download Docker Desktop from docker.com
- Linux: `sudo apt-get install docker-ce docker-ce-cli containerd.io`

### 1.2 Start Qdrant Vector Database

```bash
# Start Qdrant
docker-compose up -d

# Verify it's running
curl http://localhost:6333/health

# Expected output:
# {"title":"healthcheck","version":"1.7.0"}
```

### 1.3 Configure Azure OpenAI

Create `.env` file in project root:
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Phase 2: Data Ingestion (Days 2-3)

### 2.1 Obtain SPL XML Files

**Option 1**: Download from DailyMed
```bash
# Example: Download Lisinopril SPL
wget https://dailymed.nlm.nih.gov/dailymed/getFile.cfm?setid=abc123&type=zip
unzip lisinopril.zip
```

**Option 2**: FDA's NSDE (National Standard Database Establishment)
- Access via https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files

### 2.2 Parse SPL XML

```python
from ingestion.spl_xml_parser import SPLXMLParser

# Initialize parser
parser = SPLXMLParser()

# Parse single document
metadata, sections = parser.parse_document('data/lisinopril_spl.xml')

print(f"Drug: {metadata.drug_name}")
print(f"SetID: {metadata.set_id}")
print(f"Sections: {len(sections)}")
```

### 2.3 Enrich with RxNorm

```python
from normalization.rxnorm_integration import DrugNormalizer

normalizer = DrugNormalizer()

# Get RxCUI from drug name
drug_info = normalizer.normalize_drug_name(metadata.drug_name)
metadata.rxcui = drug_info['rxcui']

# Or from NDC code
for ndc in metadata.ndc_codes:
    ndc_info = normalizer.normalize_ndc(ndc)
    if ndc_info:
        metadata.rxcui = ndc_info['rxcui']
        break
```

### 2.4 Create Chunks

```python
from ingestion.chunking_strategy import DualChunker

# Initialize chunker
chunker = DualChunker(
    chunk_size=512,
    overlap=50,
    min_chunk_size=100
)

# Chunk document
chunks = chunker.chunk_document(metadata, sections)

print(f"Created {len(chunks)} chunks")
```

### 2.5 Generate Embeddings

```python
from retrieval.hybrid_retriever import DenseEmbedder
import numpy as np

# Initialize embedder (downloads model on first run)
embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")

# Generate embeddings for all chunks
semantic_texts = [chunk.semantic_text for chunk in chunks]
embeddings = embedder.embed(semantic_texts)

print(f"Embedding shape: {embeddings.shape}")  # (num_chunks, 768)
```

### 2.6 Upsert to Qdrant

```python
from vector_db.qdrant_manager import QdrantManager
from qdrant_client.models import SparseVector

# Initialize Qdrant
qm = QdrantManager(host="localhost", port=6333)

# Create collection (first time only)
qm.create_collection(dense_vector_size=768, recreate=False)

# Prepare sparse vectors (empty for now)
sparse_vectors = [SparseVector(indices=[], values=[]) for _ in chunks]

# Upsert chunks
qm.upsert_chunks(
    chunks=[chunk.to_dict() for chunk in chunks],
    dense_embeddings=embeddings,
    sparse_embeddings=sparse_vectors
)

print("Chunks uploaded to Qdrant")
```

### 2.7 Batch Processing Script

Create `scripts/ingest_spl_batch.py`:
```python
import os
import glob
from ingestion.spl_xml_parser import SPLXMLParser
from ingestion.chunking_strategy import DualChunker
from retrieval.hybrid_retriever import DenseEmbedder
from vector_db.qdrant_manager import QdrantManager
from normalization.rxnorm_integration import DrugNormalizer
from qdrant_client.models import SparseVector

def ingest_directory(spl_dir, qdrant_host="localhost", qdrant_port=6333):
    """Ingest all SPL XML files in directory"""
    
    # Initialize components
    parser = SPLXMLParser()
    chunker = DualChunker(chunk_size=512, overlap=50)
    embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
    qm = QdrantManager(host=qdrant_host, port=qdrant_port)
    normalizer = DrugNormalizer()
    
    # Find all XML files
    xml_files = glob.glob(os.path.join(spl_dir, "*.xml"))
    print(f"Found {len(xml_files)} SPL files")
    
    for i, xml_file in enumerate(xml_files):
        print(f"\n[{i+1}/{len(xml_files)}] Processing {os.path.basename(xml_file)}")
        
        try:
            # Parse
            metadata, sections = parser.parse_document(xml_file)
            
            # Normalize drug name
            drug_info = normalizer.normalize_drug_name(metadata.drug_name)
            if drug_info:
                metadata.rxcui = drug_info['rxcui']
            
            # Chunk
            chunks = chunker.chunk_document(metadata, sections)
            
            # Embed
            semantic_texts = [chunk.semantic_text for chunk in chunks]
            embeddings = embedder.embed(semantic_texts)
            
            # Upsert
            sparse_vectors = [SparseVector(indices=[], values=[]) for _ in chunks]
            qm.upsert_chunks(
                chunks=[chunk.to_dict() for chunk in chunks],
                dense_embeddings=embeddings,
                sparse_embeddings=sparse_vectors
            )
            
            print(f"✓ Uploaded {len(chunks)} chunks for {metadata.drug_name}")
            
        except Exception as e:
            print(f"✗ Error processing {xml_file}: {e}")
            continue
    
    print("\n=== Ingestion Complete ===")
    print(f"Total chunks: {qm.count_chunks()}")

if __name__ == "__main__":
    ingest_directory("data/spl_xml_files/")
```

Run batch ingestion:
```bash
python scripts/ingest_spl_batch.py
```

---

## Phase 3: Query System Setup (Day 4)

### 3.1 Initialize Components

Create `main.py`:
```python
import os
from dotenv import load_dotenv

from normalization.rxnorm_integration import DrugNormalizer
from retrieval.hybrid_retriever import (
    HybridRetriever, DenseEmbedder, CrossEncoderReranker
)
from generation.constrained_extractor import (
    ConstrainedExtractor, PostGenerationValidator, RegulatoryQAGenerator
)
from vector_db.qdrant_manager import QdrantManager
from orchestrator.qa_orchestrator import RegulatoryQAOrchestrator

# Load environment variables
load_dotenv()

# Initialize components
print("Initializing system components...")

# Drug normalization
normalizer = DrugNormalizer()

# Retrieval system
dense_embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
vector_db = QdrantManager(host="localhost", port=6333)

retriever = HybridRetriever(
    dense_embedder=dense_embedder,
    sparse_embedder=None,  # Add BM25 if needed
    reranker=reranker,
    vector_db_manager=vector_db
)

# Generation system
extractor = ConstrainedExtractor(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_api_key=os.getenv("AZURE_OPENAI_KEY"),
    model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
)
validator = PostGenerationValidator(similarity_threshold=95)
generator = RegulatoryQAGenerator(extractor=extractor, validator=validator)

# Orchestrator
orchestrator = RegulatoryQAOrchestrator(
    drug_normalizer=normalizer,
    retriever=retriever,
    generator=generator,
    audit_log_path="pharma_qa_audit.jsonl"
)

print("✓ System ready")

# Example query
if __name__ == "__main__":
    query = "What are the adverse reactions of Lisinopril?"
    print(f"\nQuery: {query}")
    
    result = orchestrator.query(query)
    
    print(f"\nStatus: {result['status']}")
    print(f"Answer: {result['answer']}")
    
    if result['status'] == 'validated':
        print(f"\nMetadata:")
        print(f"  Drug: {result['metadata']['drug_name']}")
        print(f"  Section: {result['metadata']['loinc_section']}")
        print(f"  Validation: {result['validation_score']:.1f}%")
```

Test the system:
```bash
python main.py
```

---

## Phase 4: API Development (Days 5-6)

### 4.1 Create FastAPI Backend

Create `api/server.py`:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from main import orchestrator  # Import initialized orchestrator

app = FastAPI(title="Pharmaceutical QA API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    status: str
    metadata: dict
    validation_score: float
    timestamp: str

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process pharmaceutical query"""
    
    try:
        result = orchestrator.query(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "pharma_qa"}

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    from vector_db.qdrant_manager import QdrantManager
    
    qm = QdrantManager(host="localhost", port=6333)
    info = qm.get_collection_info()
    total_chunks = qm.count_chunks()
    
    return {
        "total_chunks": total_chunks,
        "collection_info": info
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Install FastAPI:
```bash
pip install fastapi uvicorn pydantic
```

Run API server:
```bash
python api/server.py
```

Test API:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the adverse reactions of Lisinopril?"
  }'
```

---

## Phase 5: Evaluation & Testing (Day 7)

### 5.1 Create Test Dataset

Create `tests/test_queries.json`:
```json
[
  {
    "query": "What are the adverse reactions of Lisinopril?",
    "expected_drug": "Lisinopril",
    "expected_section": "ADVERSE REACTIONS",
    "ground_truth": "dizziness, headache, fatigue, cough"
  },
  {
    "query": "What are contraindications of Metformin?",
    "expected_drug": "Metformin",
    "expected_section": "CONTRAINDICATIONS",
    "ground_truth": "renal impairment, metabolic acidosis"
  }
]
```

### 5.2 Evaluation Script

Create `tests/evaluate_system.py`:
```python
import json
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision
from main import orchestrator

def run_evaluation(test_file="tests/test_queries.json"):
    """Run RAGAS evaluation on test set"""
    
    # Load test queries
    with open(test_file) as f:
        test_cases = json.load(f)
    
    results = []
    
    for test in test_cases:
        print(f"\nTesting: {test['query']}")
        
        # Run query
        result = orchestrator.query(test['query'])
        
        # Collect metrics
        results.append({
            "query": test['query'],
            "answer": result['answer'],
            "status": result['status'],
            "validation_score": result.get('validation_score', 0),
            "expected_section": test['expected_section'],
            "actual_section": result.get('metadata', {}).get('loinc_section', 'N/A')
        })
    
    # Calculate statistics
    validated = sum(1 for r in results if r['status'] == 'validated')
    avg_validation = sum(r['validation_score'] for r in results) / len(results)
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"Total queries: {len(results)}")
    print(f"Validated: {validated}/{len(results)} ({100*validated/len(results):.1f}%)")
    print(f"Avg validation score: {avg_validation:.1f}%")
    
    return results

if __name__ == "__main__":
    run_evaluation()
```

Run evaluation:
```bash
python tests/evaluate_system.py
```

---

## Phase 6: Production Hardening (Days 8-10)

### 6.1 Add Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_drug_normalization(drug_name):
    return normalizer.normalize_drug_name(drug_name)
```

### 6.2 Add Rate Limiting

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")
async def process_query(request: QueryRequest):
    ...
```

### 6.3 Add Monitoring

```python
import logging
from pythonjsonlogger import jsonlogger

# Configure JSON logging
logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)
```

### 6.4 Add Error Handling

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )
```

---

## Phase 7: Deployment (Days 11-14)

### 7.1 Containerize Application

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements_regulatory.txt .
RUN pip install --no-cache-dir -r requirements_regulatory.txt

# Copy application
COPY . .

# Download models (cache them in image)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO'); \
    from transformers import CrossEncoder; \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')"

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "api/server.py"]
```

Build image:
```bash
docker build -t pharma-qa-api:1.0 .
```

### 7.2 Update Docker Compose

Add API service to `docker-compose.yml`:
```yaml
services:
  qdrant:
    # ... existing config ...
  
  api:
    image: pharma-qa-api:1.0
    container_name: pharma_qa_api
    ports:
      - "8000:8000"
    environment:
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_KEY=${AZURE_OPENAI_KEY}
      - AZURE_OPENAI_DEPLOYMENT=${AZURE_OPENAI_DEPLOYMENT}
    depends_on:
      - qdrant
    restart: unless-stopped
```

Deploy:
```bash
docker-compose up -d
```

---

## Checklist: Pre-Production

- [ ] SPL XML files downloaded and validated
- [ ] Qdrant collection created with proper indices
- [ ] All drugs ingested and indexed
- [ ] RxNorm/RxClass connectivity verified
- [ ] System prompt finalized and version-controlled
- [ ] Validation threshold set (≥95%)
- [ ] Test dataset created and evaluation run
- [ ] API rate limiting configured
- [ ] Audit logging enabled and tested
- [ ] Error handling implemented
- [ ] Health check endpoints working
- [ ] Monitoring/alerting configured
- [ ] Documentation reviewed
- [ ] UI disclaimer approved by legal/compliance

---

## Troubleshooting

### Issue: Qdrant connection fails
```bash
# Check if container is running
docker ps | grep qdrant

# Check logs
docker logs pharma_qa_qdrant

# Restart container
docker-compose restart qdrant
```

### Issue: Model download too slow
```bash
# Pre-download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')"
```

### Issue: Low validation scores
- Check if source chunks contain the generated text
- Review LLM temperature (should be 0.0)
- Verify system prompt is loaded correctly
- Check for encoding issues in SPL XML parsing

---

## Next Steps

1. **Scale ingestion**: Process full FDA SPL corpus (~100K documents)
2. **Add caching**: Redis for drug normalization results
3. **Optimize retrieval**: Fine-tune reranking thresholds
4. **Build UI**: Chat interface with source citations
5. **Add analytics**: Track query patterns, validation failures
6. **Implement A/B testing**: Test different retrieval strategies

---

**You now have a complete regulatory-grade pharmaceutical QA system!**

Remember the core principle:  
**This system retrieves truth. It does not generate intelligence.**
