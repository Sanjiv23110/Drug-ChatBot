
import sys
import os
from qdrant_client import QdrantClient

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    try:
        client = QdrantClient(host="localhost", port=6333)
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        print(f"Collections found: {collection_names}")
        
        if "spl_parents" not in collection_names:
            print("Error: 'spl_parents' collection not found.")
            return

        # Get stats
        stats = client.get_collection("spl_parents")
        print(f"Total Parent Chunks: {stats.points_count}")
        
        # Scroll to find unique drugs
        unique_drugs = set()
        next_offset = None
        
        print("Scanning for unique drugs...")
        while True:
            records, next_offset = client.scroll(
                collection_name="spl_parents",
                scroll_filter=None,
                limit=100,
                with_payload=["drug_name"],
                with_vectors=False,
                offset=next_offset
            )
            
            for record in records:
                if record.payload and "drug_name" in record.payload:
                    unique_drugs.add(record.payload["drug_name"])
            
            if next_offset is None:
                break
                
        print(f"\nUnique Drugs Found in DB ({len(unique_drugs)}):")
        for drug in sorted(unique_drugs):
            print(f"- {drug}")

    except Exception as e:
        print(f"Error querying Qdrant: {e}")

if __name__ == "__main__":
    main()
