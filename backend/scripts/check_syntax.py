
try:
    from app.ingestion.factspan_extractor import FactSpanExtractor
    from app.retrieval.retrieve import RetrievalEngine
    print("Syntax check passed.")
except Exception as e:
    print(f"Syntax error: {e}")
