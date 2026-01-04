import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.ingestion_service import IngestionService
from app.services.vector_store import VectorStoreService
from app.core.config import settings

def run_ingestion():
    print(f"Ingesting from: {settings.DOCUMENTS_DIR}")
    ingestion = IngestionService()
    vector_store = VectorStoreService()
    
    docs = ingestion.process_directory(settings.DOCUMENTS_DIR)
    print(f"Found {len(docs)} document chunks.")
    
    if docs:
        vector_store.add_documents(docs)
        print("Ingestion complete.")
    else:
        print("No documents found or processed.")

if __name__ == "__main__":
    run_ingestion()
