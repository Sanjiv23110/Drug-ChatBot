"""
Simple FastAPI endpoint to test your RAG system.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

# Import RAG components
from app.vectorstore.faiss_store import FAISSVectorStore
from app.vectorstore.index_manager import IndexManager
from app.metadata.sqlite_store import SQLiteMetadataStore
from app.ingestion.embedder import AzureEmbedder
from app.retrieval.retriever import retrieve_with_resolver
from app.resolver.drug_name_resolver import DrugNameResolver
from app.generation.answer_generator import AnswerGenerator

# Component 2: Multi-Query Expansion Agent
from app.agents.query_expander import QueryExpander
from app.utils.deduplication import deduplicate_chunks

# Production: Exhaustive Section Retrieval System
from app.retrieval.section_retrieval import SectionDetector, ExhaustiveSectionRetriever

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
query_expander = None  # Component 2: Query expansion agent
section_detector = None  # Production: Section detection
exhaustive_retriever = None  # Production: Exhaustive section retrieval

@app.on_event("startup")
async def startup():
    """Initialize RAG system on startup."""
    global faiss_store, metadata_store, embedder, resolver, generator, query_expander
    global section_detector, exhaustive_retriever
    
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
    
    # Component 2: Initialize query expander agent
    query_expander = QueryExpander(drug_resolver=resolver)
    print("✓ Query expansion agent initialized")
    
    # Production: Initialize exhaustive section retrieval
    section_detector = SectionDetector()
    exhaustive_retriever = ExhaustiveSectionRetriever(metadata_store)
    print("✓ Exhaustive section retrieval initialized")
    
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
        # PRODUCTION: Hybrid Intelligent Routing
        # Step 0: Detect if this is a section query (e.g., adverse reactions)
        section_detection = section_detector.detect_section(request.question)
        drug_name = resolver.extract_drug_names(request.question)
        
        final_chunks = None  # Initialize to prevent UnboundLocalError
        
        if section_detection and drug_name:
            # EXHAUSTIVE MODE: Get ALL chunks from the section
            section_type, confidence = section_detection
            detected_drug = drug_name[0] if drug_name else None
            
            if detected_drug:
                logging.info(
                    f"🎯 EXHAUSTIVE retrieval triggered: {section_type} "
                    f"for {detected_drug} (confidence: {confidence:.2f})"
                )
                
                # Get ALL chunks from the section
                final_chunks = exhaustive_retriever.retrieve_section_exhaustive(
                    drug_name=detected_drug,
                    section_type=section_type
                )
                
                if not final_chunks:
                    # Fall back to standard retrieval if section not found
                    logging.warning("Exhaustive retrieval returned no results, falling back")
                    section_detection = None
                else:
                    logging.info(f"✓ Retrieved {len(final_chunks)} chunks from section exhaustively")        
        
        if not section_detection or not final_chunks:
            # STANDARD MODE: Multi-query expansion + top-K retrieval
            # Step 1: Expand query into variants (Component 2)
            query_variants = query_expander.expand_query(request.question)
            
            # Step 2: Retrieve for ALL variants
            all_chunks = []
            for variant in query_variants:
                chunks = retrieve_with_resolver(
                    query=variant,
                    faiss_store=faiss_store,
                    metadata_store=metadata_store,
                    embedder=embedder,
                    resolver=resolver
                )
                all_chunks.extend(chunks)
            
            # Step 3: Deduplicate chunks
            unique_chunks = deduplicate_chunks(all_chunks)
            
            # Step 4: Select top chunks (max 60 from Component 1)
            final_chunks = unique_chunks[:60]
        
        if not final_chunks:
            return QueryResponse(
                answer="I couldn't find relevant information in the available drug monographs.",
                sources=[],
                has_answer=False,
                chunks_retrieved=0
            )
        
        # Step 5: Generate answer from comprehensive context
        result = generator.generate(
            query=request.question,  # Use original query for answer
            context_chunks=final_chunks
        )
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'],
            has_answer=result['has_answer'],
            chunks_retrieved=len(final_chunks)
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
