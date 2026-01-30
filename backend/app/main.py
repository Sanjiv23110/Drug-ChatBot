"""
FastAPI endpoint using PostgreSQL-based retrieval system.

Uses SQL-first retrieval with dynamic section handling and pgvector fallback.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

# NEW: PostgreSQL-based retrieval system
from app.retrieval.router import RetrievalRouter, FormattedContext
from app.generation.answer_generator import AnswerGenerator

# Initialize app
app = FastAPI(title="Medical RAG API - PostgreSQL Edition")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retrieval router (singleton pattern)
retrieval_router = None
generator = None

@app.on_event("startup")
async def startup():
    """Initialize PostgreSQL-based RAG system on startup."""
    global retrieval_router, generator
    
    print("🚀 Initializing PostgreSQL-based RAG system...")
    
    # Initialize new PostgreSQL-based retrieval router
    retrieval_router = RetrievalRouter(enable_vector_fallback=True)
    print("✓ PostgreSQL retrieval router initialized")
    
    # Initialize answer generator
    generator = AnswerGenerator()
    print("✓ Answer generator initialized")
    
    # Check database connection
    try:
        drugs = await retrieval_router.list_drugs()
        print(f"✓ Connected to PostgreSQL - {len(drugs)} drugs available")
        
        # Show available section types
        section_types = await retrieval_router.list_all_section_types()
        print(f"✓ Discovered {len(section_types)} section types")
        
    except Exception as e:
        print(f"⚠️  Warning: Database connection issue - {e}")
        print("   Make sure PostgreSQL is running and data is ingested")
    
    print("✅ PostgreSQL RAG system ready!\n")

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring."""
    from datetime import datetime
    
    # Check database connectivity
    db_healthy = False
    drug_count = 0
    section_count = 0
    
    if retrieval_router:
        try:
            drugs = await retrieval_router.list_drugs()
            drug_count = len(drugs)
            
            section_types = await retrieval_router.list_all_section_types()
            section_count = len(section_types)
            
            db_healthy = True
        except Exception as e:
            logging.error(f"Health check failed: {e}")
    
    return {
        "status": "healthy" if db_healthy else "degraded",
        "service": "solomind-backend-postgresql",
        "timestamp": datetime.now().isoformat(),
        "database": {
            "connected": db_healthy,
            "drugs_available": drug_count,
            "section_types": section_count,
            "backend": "PostgreSQL with pgvector"
        }
    }

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    has_answer: bool
    chunks_retrieved: int
    retrieval_path: Optional[str] = None  # NEW: Which path was used


@app.post("/api/chat", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the PostgreSQL-based RAG system.
    
    Uses SQL-first retrieval with dynamic section handling.
    
    Example:
        POST /api/chat
        {
            "question": "What are the contraindications for nizatidine?"
        }
    """
    if retrieval_router is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    try:
        logging.info(f"Query: {request.question}")
        
        # Step 1: Retrieve context using PostgreSQL-based router
        context, raw_result = await retrieval_router.route_with_result(request.question)
        
        logging.info(
            f"Retrieved via {context.path_used}: "
            f"{context.total_chunks} chunks for drug '{context.drug_name}'"
        )
        
        # Check if we found anything
        if not context.sources or context.total_chunks == 0:
            return QueryResponse(
                answer="I couldn't find relevant information in the available drug monographs. "
                       "Please try rephrasing your question or ask about a different drug.",
                sources=[],
                has_answer=False,
                chunks_retrieved=0,
                retrieval_path=context.path_used
            )
        
        # Step 2: Map PostgreSQL fields to format expected by answer generator
        # PostgreSQL format: {drug_name, section, header, page_start, char_count, content_text}
        # Generator expects: {chunk_text, drug_generic, page_num, section_name, file_path}
        mapped_chunks = []
        for source in context.sources:
            mapped_chunks.append({
                'chunk_text': source.get('content_text', ''),  # Map content_text → chunk_text
                'drug_generic': source.get('drug_name', 'Unknown drug'),  # Map drug_name → drug_generic
                'page_num': source.get('page_start', 'Unknown'),  # Map page_start → page_num
                'section_name': source.get('section', 'Unknown'),  # Map section → section_name
                'file_path': f"{source.get('drug_name', 'unknown')}_monograph"  # Create synthetic file_path
            })
        
        # Step 3: Generate answer from mapped context
        result = generator.generate(
            query=request.question,
            context_chunks=mapped_chunks  # Use mapped chunks
        )
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'],
            has_answer=result['has_answer'],
            chunks_retrieved=context.total_chunks,
            retrieval_path=context.path_used
        )
    
    except Exception as e:
        logging.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/drugs")
async def list_drugs():
    """Get list of all available drugs in the database."""
    if retrieval_router is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        drugs = await retrieval_router.list_drugs()
        return {"drugs": drugs, "count": len(drugs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/drugs/{drug_name}/sections")
async def list_drug_sections(drug_name: str):
    """Get available sections for a specific drug."""
    if retrieval_router is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        sections = await retrieval_router.list_sections(drug_name)
        return {"drug": drug_name, "sections": sections, "count": len(sections)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sections")
async def list_all_sections():
    """Get all known section types with usage statistics."""
    if retrieval_router is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        section_types = await retrieval_router.list_all_section_types()
        return {"section_types": section_types, "count": len(section_types)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Medical RAG API - PostgreSQL Edition",
        "backend": "PostgreSQL with pgvector and pg_trgm",
        "retrieval": "SQL-first with vector fallback",
        "endpoints": {
            "chat": "POST /api/chat",
            "drugs": "GET /api/drugs",
            "sections": "GET /api/drugs/{drug_name}/sections",
            "all_sections": "GET /api/sections",
            "health": "GET /health"
        }
    }

