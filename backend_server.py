import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure root directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from orchestrator.qa_orchestrator import RegulatoryQAOrchestrator
    from normalization.rxnorm_integration import DrugNormalizer
    from retrieval.hybrid_retriever import HybridRetriever, DenseEmbedder, CrossEncoderReranker
    from generation.constrained_extractor import ConstrainedExtractor, PostGenerationValidator, RegulatoryQAGenerator
    from vector_db.qdrant_manager import QdrantManager
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")
    MODULES_AVAILABLE = False

app = FastAPI(title="Solomind.ai API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "user"
    session_id: Optional[str] = "default"

class QueryResponse(BaseModel):
    answer: str
    status: str
    reason: Optional[str] = None
    metadata: Dict[str, Any] = {}
    validation_score: Optional[float] = None
    timestamp: Optional[str] = None


# Global orchestrator instance
orchestrator = None

@app.on_event("startup")
async def startup_event():
    global orchestrator
    if MODULES_AVAILABLE:
        print("Initializing Solomind Orchestrator...")
        try:
            # Initialize with default settings matching orchestrator main
            normalizer = DrugNormalizer()
            # Note: Verify model paths/names exist. Using validation logic from main.
            dense_embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
            reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
            vector_db = QdrantManager(host="localhost", port=6333, collection_name="spl_children")
            
            retriever = HybridRetriever(
                dense_embedder=dense_embedder,
                sparse_embedder=None, # Update if sparse used
                reranker=reranker,
                vector_db_manager=vector_db
            )
            
            extractor = ConstrainedExtractor(model_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-agent"))
            validator = PostGenerationValidator(similarity_threshold=75)
            generator = RegulatoryQAGenerator(extractor=extractor, validator=validator)
            
            orchestrator = RegulatoryQAOrchestrator(
                drug_normalizer=normalizer,
                retriever=retriever,
                generator=generator
            )
            print("Orchestrator initialized successfully.")
        except Exception as e:
            print(f"Failed to fully initialize orchestrator (Models/DB might be missing): {e}")
    else:
         print("Running in Mock Mode (Modules missing).")

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not orchestrator:
        # Fallback Mock for testing frontend content
        await asyncio.sleep(1)
        return {
            "answer": f"**Solomind Backend Status**: Indeterminate.\n\nI received your query: \"{request.query}\".\n\nHowever, the backend orchestration engine is not currently connected or initialized. Please ensure:\n1. Qdrant is running (`docker-compose up`)\n2. Python dependencies are installed\n3. `backend_server.py` is running without errors.",
            "status": "mock_response",
            "timestamp": "now"
        }
    
    try:
        # Run blocking code in threadpool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            orchestrator.query, 
            request.query, 
            request.user_id, 
            request.session_id
        )
        return result
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
