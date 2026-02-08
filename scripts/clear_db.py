
"""
Database Wipe Script
-------------------
This script empties the Qdrant Vector Database by deleting the 'spl_children' and 'spl_parents'
collections entirely. This is a destructive operation.
"""
from qdrant_client import QdrantClient
import os

def wipe_database(host="localhost", port=6333):
    client = QdrantClient(host=host, port=port)
    collections = ["spl_children", "spl_parents"]
    
    print(f"Connecting to Qdrant at {host}:{port}...")
    
    for collection_name in collections:
        try:
            # Check if exists first
            exists = client.collection_exists(collection_name)
            if exists:
                print(f"Applying DELETE to collection: {collection_name}")
                client.delete_collection(collection_name)
                print(f" -> DELETED: {collection_name}")
            else:
                print(f" -> SKIPPED: {collection_name} (does not exist)")
                
        except Exception as e:
            print(f"Error deleting {collection_name}: {e}")

if __name__ == "__main__":
    # Safety Check: Require explicit user confirmation in terminal (optional for automation, but good for interactive)
    print("WARNING: This will permanently delete all data in 'spl_children' and 'spl_parents'.")
    resp = input("Type 'yes' to proceed: ")
    if resp.lower() == 'yes':
        wipe_database()
    else:
        print("Aborted.")
