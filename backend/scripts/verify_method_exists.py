
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.ingestion.ingest import IngestionPipeline

print(f"Method exists: {hasattr(IngestionPipeline, '_convert_to_chunked_sections')}")
