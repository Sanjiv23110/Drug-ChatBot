# Hugging Face Spaces Deployment Guide

This document outlines the STRICT steps to deploy the Solomind chatbot to a Hugging Face Space (CPU Basic) while adhering to all constraints (no logic changes, using Qdrant Cloud, and using `ingest_v3.py`).

## 1. Minimal Code Diffs Required

Apply these exact minimal changes to transition from local Qdrant to Qdrant Cloud.

### `vector_db/hierarchical_qdrant.py`
```diff
@@ -41,8 +41,6 @@
     def __init__(
         self,
-        host: str = "localhost",
-        port: int = 6333,
         child_collection: str = "spl_children",
         parent_collection: str = "spl_parents"
     ):
@@ -57,7 +55,11 @@
         """
-        self.client = QdrantClient(host=host, port=port)
+        import os
+        self.client = QdrantClient(
+            url=os.environ["QDRANT_URL"],
+            api_key=os.environ["QDRANT_API_KEY"]
+        )
         self.child_collection = child_collection
         self.parent_collection = parent_collection
```

### `vector_db/qdrant_manager.py`
```diff
@@ -33,8 +33,6 @@
     def __init__(
         self,
-        host: str = "localhost",
-        port: int = 6333,
         collection_name: str = "spl_chunks"
     ):
@@ -47,7 +45,11 @@
         """
-        self.client = QdrantClient(host=host, port=port)
+        import os
+        self.client = QdrantClient(
+            url=os.environ["QDRANT_URL"],
+            api_key=os.environ["QDRANT_API_KEY"]
+        )
         self.collection_name = collection_name
```

### `scripts/ingest_v3.py`
```diff
@@ -45,8 +45,6 @@
 # Qdrant Config
 load_dotenv()
-QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
-QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
 
 def get_already_ingested_files(qm: HierarchicalQdrantManager) -> Set[str]:
     try:
-        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
+        client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
@@ -98,7 +96,7 @@
-        qm = HierarchicalQdrantManager(host=QDRANT_HOST, port=QDRANT_PORT)
+        qm = HierarchicalQdrantManager()
```

### `backend_server.py`
```diff
@@ -67,9 +67,5 @@
-            # Use 127.0.0.1 to avoid IPv6 resolution issues on Windows
-            qdrant_host = os.getenv("QDRANT_HOST", "127.0.0.1")
-            qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
-            
-            print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
-            vector_db = QdrantManager(host=qdrant_host, port=qdrant_port, collection_name="spl_children")
+            print("Connecting to Qdrant Cloud...")
+            vector_db = QdrantManager(collection_name="spl_children")
```

---

## 2. Requirements additions (`requirements.txt`)
Create `requirements.txt` in the repository root:
```text
fastapi==0.128.2
uvicorn==0.40.0
qdrant-client==1.16.2
sentence-transformers==2.7.0
openai==2.17.0
python-dotenv==1.2.1
pydantic==2.12.5
lxml==6.0.2
httpx==0.28.1
```

---

## 3. Space Runtime Settings
- **Hardware Selection**: Standard CPU (Basic / `cpu-basic`)
- **SDK**: Docker (Create a `Dockerfile`) OR Gradio/FastAPI Python space if natively supported (FastAPI is supported via Docker Spaces).
- **Startup Command** (for standard Docker or basic Python setting):
  `uvicorn backend_server:app --host 0.0.0.0 --port 7860`

*If setting up as a Hugging Face Docker Space, create `Dockerfile`*:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# HF Spaces run on port 7860
CMD ["uvicorn", "backend_server:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## 4. Environment Variables List
In the Hugging Face Space Settings -> **Variables and secrets**, add:

**Secrets:**
- `QDRANT_API_KEY` (Your Qdrant cluster API Key)
- `AZURE_OPENAI_API_KEY` (Your Azure OpenAI Key)

**Variables:**
- `QDRANT_URL` (Your exact Qdrant Cloud Cluster URL: `https://<cluster-id>.us-east4-0.gcp.cloud.qdrant.io:6333`)
- `AZURE_OPENAI_ENDPOINT` (Your Azure OpenAI Endpoint)
- `AZURE_OPENAI_CHAT_DEPLOYMENT` (Deployment name, e.g. `gpt-4o-agent`)
- `AZURE_OPENAI_API_VERSION` (e.g. `2024-02-15-preview`)

*(Optional depending on embeddings)*:
- If `HUGGINGFACE_API_KEY` is needed for downloading models heavily without rate-limits, add it.

---

## 5. Deployment Steps Summary
1. Apply the **minimal code diffs** above (no logic modifications).
2. Ensure you have the XML files safely uploaded (if ingestion will happen on HF space, though ingestion can happen from your local machine to Qdrant Cloud using `scripts/ingest_v3.py` beforehand). 
   *Recommended:* Run `python scripts/ingest_v3.py` locally pointing to `QDRANT_URL` and `QDRANT_API_KEY` to pre-populate Qdrant Cloud.
3. Commit and push the code + `requirements.txt` (+ `Dockerfile` if using Docker approach) to your Hugging Face Space repository.
4. Set all the exact environment variables (Variables & Secrets) listed above in HF Space Settings.
5. Watch the HF Space Build Logs. The space will verify the Qdrant connection on startup (`startup_event` in `backend_server.py`) and be ready to accept requests on the HF interface (port 7860) without modifying Chatbot retrieval algorithms.

---

### Final Validation Checklist:
- [x] Retrieval logic unchanged (CrossEncoder, Hybrid, and Extractors untouched)
- [x] `scripts/ingest_v3.py` configured correctly for remote pipeline initialization.
- [x] No `localhost:6333` hardcodes remaining in manager code.
- [x] No refactors performed.
- [x] Model answers will remain 100% physically identical.
