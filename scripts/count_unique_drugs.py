import sys
import os
from qdrant_client import QdrantClient

# Add ROOT_DIR to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

def count_unique_drugs():
    client = QdrantClient(host="127.0.0.1", port=6333)
    
    # Scroll through spl_parents to get unique drug names
    # limit=10000 should cover our current small DB
    result = client.scroll(
        collection_name="spl_parents",
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    
    unique_drugs = set()
    for point in result[0]:
        drug = point.payload.get('drug_name')
        if drug:
            unique_drugs.add(drug)
            
    print(f"Unique drugs found in Qdrant (spl_parents): {len(unique_drugs)}")
    for drug in sorted(list(unique_drugs)):
        print(f" - {drug}")

if __name__ == "__main__":
    count_unique_drugs()
