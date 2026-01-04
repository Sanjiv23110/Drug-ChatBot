from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List
import os
import shutil
from app.services.rag_service import RagService
from app.services.ingestion_service import IngestionService
from app.services.vector_store import VectorStoreService
from app.core.config import settings

router = APIRouter()
rag_service = RagService()
ingestion_service = IngestionService()
vector_store = VectorStoreService()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = rag_service.ask(request.message)
        sources = [m.get('filename', 'Unknown') for m in result['metadatas']]
        sources = list(set(sources))
        return ChatResponse(answer=result['answer'], sources=sources)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\n\n=== CHAT ERROR ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Full traceback:\n{error_details}")
        print("==================\n")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

@router.post("/ingest")
async def ingest_documents(background_tasks: BackgroundTasks):
    """
    Trigger ingestion of documents from the configured directory.
    """
    background_tasks.add_task(run_ingestion)
    return {"status": "Ingestion started in background"}

def run_ingestion():
    print("Starting ingestion...")
    docs = ingestion_service.process_directory(settings.DOCUMENTS_DIR)
    vector_store.add_documents(docs)
    print("Ingestion complete.")

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not os.path.exists(settings.DOCUMENTS_DIR):
        os.makedirs(settings.DOCUMENTS_DIR)
    
    file_location = os.path.join(settings.DOCUMENTS_DIR, file.filename)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}

@router.get("/stats")
async def get_stats():
    try:
        count = vector_store.collection.count()
        return {"document_count": count, "data_path": settings.DOCUMENTS_DIR}
    except Exception as e:
        return {"error": str(e)}
