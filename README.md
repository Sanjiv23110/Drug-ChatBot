---
title: Solomind
emoji: ⚕️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Solomind.ai Regulatory QA Chatbot - Technical Architecture

This document provides an exhaustive, implementation-accurate technical overview of the Solomind.ai chatbot architecture, based strictly on the current codebase.

## 1. SYSTEM OVERVIEW

### High-Level Architecture Diagram
```ascii
                                +-------------------+
                                |  Ingestion (XML)  |
                                +---------+---------+
                                          |
                                          v
+------------------+             +--------+--------+            +------------------+
|   User Client    | <---------> | FastAPI Backend | ---------> | Azure OpenAI LLM |
+------------------+             +--------+--------+            +------------------+
                                          |
                                          v
                                 +--------+--------+
                                 |  QA Orchestrator|
                                 +--------+--------+
                                          | (Hybrid Retrieval + Cross-Encoder)
                                          v
                                 +--------+--------+
                                 |  Qdrant Cloud   |
                                 +-----------------+
                                  (spl_children)
                                  (spl_parents)
```

### End-to-End Data Flow
1. **Ingestion**: `ingest_v3.py` processes SPL XML documents via `SPLXMLParser`. Data is split by `HierarchicalChunker` into paragraphs (`ParentChunk`) and sentences (`ChildChunk`). Dense/sparse embeddings are computed and upserted into Qdrant (`spl_parents`, `spl_children`).
2. **Retrieval**: `backend_server.py` routes the query to `RegulatoryQAOrchestrator`. `IntentClassifier` gates the query (Regex + LLM Arbiter). `EntityValidator` extracts the drug name. `SectionClassifier` maps queries to LOINC codes. `HybridRetriever` executes a dense/sparse search on `spl_children`, reranks with `CrossEncoderReranker`, and fetches corresponding canonical parent paragraphs.
3. **Generation**: `RegulatoryQAGenerator` calls `ConstrainedExtractor`. Depending on query shape (`FACT`, `MANAGEMENT`, `LIST`), it either deterministically extracts spans/paragraphs or invokes the Azure LLM (Sentence Locator prompt) for ambiguous queries. `PostGenerationValidator` verifies the result using fuzzy matching.

### Request Lifecycle (User → Response)
1. User submits POST `{"query": "..."}` to `/chat`.
2. Handled via `asyncio.get_event_loop().run_in_executor` to execute sync orchestrator logic.
3. `IntentClassifier.classify()` enforces guardrails.
4. `EntityValidator.validate()` checks for a valid drug entity.
5. `SectionClassifier.classify()` resolves the LOINC section code target.
6. `HybridRetriever.retrieve()` runs vector search with `filter_conditions={"drug_name": target}`.
7. Candidate chunks are reranked via `CrossEncoderReranker` and filtered based on the section intent.
8. Parent definitions are fetched (accumulating matching child scores).
9. Hierarchical conflict resolution (`HierarchicalConflictResolver.resolve()`) is applied.
10. `ConstrainedExtractor.extract()` extracts the text (with or without LLM).
11. `PostGenerationValidator.validate()` enforces a 95% partial ratio similarity threshold.
12. Final `QueryResponse` returned to the client.

## 2. INGESTION PIPELINE (ingest_v3.py)
- **Entry point:** `scripts/ingest_v3.py` `main()`
- **Document loading logic:** Reads `.xml` files from `data/xml` via `glob.glob()`. Skips existing files by scrolling `spl_parents` for unique `set_id` (`get_already_ingested_files`).
- **File parsing methods:** `SPLXMLParser` utilizes `lxml.etree`. Tables are preserved using FDA XSLT (`spl.xsl`), transformed to HTML, and converted to Markdown via `html2text` in `TablePreserver`.
- **Chunking strategy:** `HierarchicalChunker` enforces:
  - Parent chunks: Full paragraphs (split on `\n\n`) or full tables. Atomic source of truth.
  - Child chunks: Individual sentences using `re.compile(r'(?<=[.!?])\s+(?=[A-Z])')`, len > 10.
- **Metadata structure:** `SPLMetadata` (set_id, version_number, root_id, effective_time, drug_name, ndc_codes, rxcui) and `SPLSection` (loinc_code, section_name, text_content).
- **Embedding model used:** Dense via `pritamdeka/S-PubMedBert-MS-MARCO`, Sparse via BM25 stable hashing (`zlib.crc32`) in `SparseEmbedder`.
- **Embedding dimensionality:** 768 parameters.
- **Vector normalization:** True (`normalize_embeddings=True`).
- **Batch size logic:** Processed sequentially file-by-file. Upserts are collected per document in `children_dicts` and `parents_dicts`.
- **Error handling:** Wrapped with `retry_upsert` (max 3 retries). Failures appended to `FAILED_FILES_LOG` (`ingestion_failures.log`).
- **Qdrant collection creation logic:** `HierarchicalQdrantManager.create_collections` builds `spl_children` (dense Cosine + sparse vectors) and `spl_parents` (dummy vector config `size=1`, no embeddings).
- **Upsert mechanism:** `upsert_children` and `upsert_parents` push `PointStruct` definitions.
- **Index configuration:** `PayloadSchemaType.KEYWORD` explicitly indexed for:
  - Children: `drug_name`, `rxcui`, `loinc_code`, `loinc_section`, `parent_id`.
  - Parents: `parent_id`, `drug_name`, `set_id`.
- **Payload schema:** Strict dict matching chunk properties + mapped metadata.
- **Id generation strategy:**
  - Parent ID: Deterministic string (`{drug}_v{version}_{loinc}_sec_{sec_idx}_para_{para_idx}`) hashed to unsigned int (`abs(hash(id))`).
  - Child ID: `{parent_id}_sent_{sent_idx}` hashed identically.

## 3. VECTOR DATABASE LAYER
- **Qdrant client initialization:** `QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])`
- **Local vs Cloud configuration:** Exclusively environment-variable driven (supports Hugging Face/Cloud configurations directly).
- **Collection parameters:**
  - Vector size: 768 (dense), 1 (dummy for parents).
  - Distance metric: `Distance.COSINE`
  - HNSW parameters: Default Qdrant values.
  - Quantization: None defined.
- **Filtering logic:** Dynamic `_build_filter` translates dict configs to `Filter(must=[FieldCondition(MatchValue/MatchAny)])`.
- **Payload indexing:** Payload KeyWord indices optimize subset filtering (`drug_name`, `parent_id`).
- **Storage structure:** Separates search indexing (`spl_children`) from display retrieval (`spl_parents`).
- **Persistence behavior:** Inherent to Qdrant Cloud. Points overwritten deterministically if IDs match.

## 4. RETRIEVAL PIPELINE (STRICTLY AS IMPLEMENTED)
- **Query preprocessing:** Dense embedded via query encode; sparse embedded using token hashing mapping term frequency.
- **Embedding generation:** `pritamdeka/S-PubMedBert-MS-MARCO`.
- **Search parameters:**
  - Top_k (`retrieval_limit`): 75 (product-specific) or 100 (class).
  - Score threshold: Native unconstrained, gated downstream natively.
  - Filters: Start with `{"drug_name": drug, "rxcui": rxcui}`.
- **Ranking logic:** `HybridRetriever.retrieve()` delegates hybrid search. Normalizes via `SectionIntentNormalizer`. `CrossEncoderReranker.rerank()` scores (`cross-encoder/ms-marco-MiniLM-L-12-v2`) limited to `rerank_top_k` (15 or 10).
- **Post-processing:** Uses Section-First Reranking. Parent retrieval accumulates the sum of rerank scores from matched children. Parents filtered post-fetch by `target_loinc` or intent synonyms (`SECTION_INTENT_MAP`). Fingerprint string deduping (300 chars parent, 200 chars child).
- **Context assembly:** Fact queries retrieve unique `sentence_text` strings. Section/List queries access the winning parent's `raw_text` directly (Rank 0).
- **Section stitching logic:** Employs canonical paragraph fetches (no cross-section synthetic stitching).
- **Expansion policies:** Not implemented / Removed from codebase.
- **Handling NOT FOUND cases:** Returns `"Evidence not found in source document."` with `status: refused` gracefully via orchestrator.
- **Exact text extraction behavior:** Deterministic shape routing:
  - `FACT`: Regex anchor parsing for shortest phrase.
  - `MANAGEMENT`/`LIST`: Verbatim return of canonical paragraph.
  - Generic: Invokes Azure LLM to emit indices.
- **Citation logic:** Appends `\n\n[Source: {section_name}]`.

## 5. LLM INTERACTION LAYER
- **Model used:** Azure OpenAI (`gpt-4o-agent` or `gpt-4o`).
- **Prompt template structure:** Uses `RUNTIME_SYSTEM_PROMPT` defining "SENTENCE LOCATOR" function, returning JSON index arrays. Uses `USER_PROMPT_TEMPLATE` for formatting contextual blocks. LLM arbiter system prompt governs intent switching.
- **System vs user prompts:** System dictates behavior ("do NOT write or generate", "output ONLY indices"). User prompt contains the query and indexed strings.
- **Token constraints:** `max_tokens=500`.
- **Temperature:** `0.0`.
- **Determinism configuration:** `temperature=0.0`, `top_p=0.1`.
- **Response validation logic:** `PostGenerationValidator` utilizes `rapidfuzz` computing substring matches (`fuzz.partial_ratio`). Expected threshold: 95.0.

## 6. API LAYER
- **Framework:** FastAPI with Uvicorn standard deployment.
- **Endpoints:** `/chat` (POST).
- **Request schema:** `QueryRequest(query, user_id, session_id)`.
- **Response schema:** `QueryResponse(answer, status, reason, metadata, validation_score, timestamp)`.
- **Error handling:** Top level `try-except` wrapped raising `HTTPException(status_code=500, detail=str(e))`.
- **Async vs sync behavior:** Endpoint uses `async def` but drops synchronous `orchestrator.query` into a ThreadPoolExecutor (`run_in_executor`).

## 7. CONFIGURATION MANAGEMENT
- **Environment variables:** Actively loads `.env` variables stringently expecting `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_CHAT_DEPLOYMENT`, `QDRANT_URL`, `QDRANT_API_KEY`.
- **Secret handling:** Retrieved natively using `os.getenv()` and `os.environ[]`. No hardcoded keys.
- **Runtime configuration:** Implements `MODULES_AVAILABLE` fast-fail detection if heavy models/dependencies drop, failing over to mock routing string.
- **Dev vs prod differences:** Defined solely by `.env` configurations (e.g., local Qdrant ports vs managed Cloud).

## 8. DEPLOYMENT ARCHITECTURE
- **Hugging Face Space configuration:** Designed for Gradio/FastAPI endpoint interfacing natively with Qdrant Cloud. Requires `requirements.txt`.
- **Cloudflare routing:** Historically referenced in Quick Tunnels for local staging.
- **Reverse proxy:** N/A for raw execution.
- **Docker setup:** Implied `docker-compose.yml` local orchestration available.
- **Port exposure:** Served automatically on `0.0.0.0:8000`.
- **Security configuration:** Uses FastAPI `CORSMiddleware`, granting `allow_origins=["*"]`.

## 9. PERFORMANCE CHARACTERISTICS
- **Expected memory usage:** High local memory usage for loading `S-PubMedBert-MS-MARCO` block and `CrossEncoder`. Handled efficiently by Azure and Qdrant backend delegation.
- **Vector search complexity:** Native fast lookup with HNSW indexing locally scaling sub-linearly.
- **Scaling limitations:** CrossEncoder linear scaling computation limits throughput efficiency based purely on batching latency.
- **Known bottlenecks:** Sequential model loading overhead and heavy latency on remote API requests.

## 10. FAILURE MODES
- **What breaks ingestion:** Irregular raw XML failing `etree.parse()`, skipping elements completely if `metadata.drug_name` mapping goes null.
- **What breaks retrieval:** Intent classifiers overriding exact match pathways causing timeout queries or dropping entirely. Discarding missing `parent_id` pointers in parsed subset indexes.
- **DB mismatch risks:** Dummy `VectorConfig` failing upserts with irregular mapping arrays when vector configurations flip.
- **Embedding drift risks:** Nullified logic; models are hardcoded precisely against `S-PubMedBert-MS-MARCO`.
- **Collection mismatch risks:** Independent collections mutating natively losing relational bounds (orphaning payloads).

## 11. DEVELOPMENT WORKFLOW
- **How to run ingestion:** Validate local config, execute `python scripts/ingest_v3.py`.
- **How to reset DB:** Use `qm.create_collections(recreate=True)`.
- **How to test retrieval:** Push HTTP requests against FastAPI `/chat` using defined `QueryRequest` definitions.
- **Logging behavior:** Streams explicit standard output `logging.StreamHandler(sys.stdout)` for ingestion.
- **Debugging tips:** Monitor `ingestion_failures.log` for failure patterns. Ensure `QDRANT_URL` maps exactly.

## 12. DATA FLOW DIAGRAMS

### Ingestion Pipeline
```ascii
[FDA XML] -> SPLXMLParser -> TablePreserver(XSLT) -> HierarchicalChunker -> Parent / Child
                                                                                 |
                                                                         Embedders (Dense/Sparse)
                                                                                 |
                                                                         Qdrant Upsert
                                                                         (spl_parents, spl_children)
```

### Retrieval Pipeline
```ascii
Query -> Embed Query -> Hybrid Search (spl_children) -> [Top K Candidates]
                                                              |
                                                   Section Normalizer
                                                              |
                                                   Cross-Encoder Reranker
                                                              |
                                           Fetch Parents (Aggregated Score/Dedup)
                                                              |
                                                   Conflict Resolution
                                                              |
                                                   [Filtered Context]
```

### Full RAG Loop
```ascii
User Input -> Intent/Section Classification -> Retrieval Pipeline -> Target Extractor (Determinism/LLM) -> Text Validation -> Response
```
