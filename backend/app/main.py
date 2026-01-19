"""
Simple FastAPI endpoint to test your RAG system.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional

# Import RAG components
from app.vectorstore.faiss_store import FAISSVectorStore
from app.vectorstore.index_manager import IndexManager
from app.metadata.sqlite_store import SQLiteMetadataStore
from app.ingestion.embedder import AzureEmbedder
from app.retrieval.retriever import retrieve_with_resolver
from app.resolver.drug_name_resolver import DrugNameResolver
from app.generation.answer_generator import AnswerGenerator

# Initialize app
app = FastAPI(title="Medical RAG API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize stores (singleton pattern)
faiss_store = None
metadata_store = None
embedder = None
resolver = None
generator = None

@app.on_event("startup")
async def startup():
    """Initialize RAG system on startup."""
    global faiss_store, metadata_store, embedder, resolver, generator
    
    print("🚀 Initializing RAG system...")
    
    # Load FAISS index
    faiss_store = FAISSVectorStore(dimension=1536)
    index_manager = IndexManager("data/faiss/medical_index")
    
    if index_manager.exists():
        loaded = index_manager.load(dimension=1536)
        if loaded:
            faiss_store.index, faiss_store.chunk_ids, _ = loaded
            print(f"✓ Loaded FAISS index with {faiss_store.count()} vectors")
    else:
        print("⚠️  No FAISS index found - run ingestion first")
    
    # Initialize stores
    metadata_store = SQLiteMetadataStore("data/metadata.db")
    embedder = AzureEmbedder()
    resolver = DrugNameResolver("data/metadata.db")
    generator = AnswerGenerator()
    
    print("✅ RAG system ready!\n")

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring."""
    from datetime import datetime
    return {
        "status": "healthy",
        "service": "solomind-backend",
        "timestamp": datetime.now().isoformat(),
        "database_loaded": faiss_store is not None and metadata_store is not None
    }

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    has_answer: bool
    chunks_retrieved: int


@app.post("/api/chat", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system.
    
    Example:
        POST /api/chat
        {
            "question": "What are the contraindications?"
        }
    """
    if faiss_store is None or faiss_store.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized or no data ingested"
        )
    
    try:
        # 1. Retrieve relevant chunks
        chunks = retrieve_with_resolver(
            query=request.question,
            faiss_store=faiss_store,
            metadata_store=metadata_store,
            embedder=embedder,
            resolver=resolver
        )
        
        if not chunks:
            return QueryResponse(
                answer="I couldn't find relevant information in the available drug monographs.",
                sources=[],
                has_answer=False,
                chunks_retrieved=0
            )
        
        # 2. Generate answer
        result = generator.generate(
            query=request.question,
            context_chunks=chunks
        )
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'],
            has_answer=result['has_answer'],
            chunks_retrieved=len(chunks)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "faiss_vectors": faiss_store.count() if faiss_store else 0
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Medical RAG API",
        "endpoints": {
            "query": "POST /query",
            "chat": "POST /api/chat",
            "health": "GET /health"
        }
    }


# Frontend-compatible chat endpoint
@app.post("/api/chat")
async def chat(request: QueryRequest):
    """
    Chat endpoint for frontend compatibility.
    
    Same as /query but matches frontend expectations.
    """
    return await query_rag(request)
