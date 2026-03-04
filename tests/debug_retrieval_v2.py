"""
Simplified Retrieval Debug (Bypassing Orchestrator dependencies)
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

def debug_direct_qdrant():
    load_dotenv()
    
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    
    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    drugs_to_test = ["dantrium"]
    loinc_code = "34088-5" # OVERDOSAGE
    
    print("\n" + "="*50)
    print("Checking Case Sensitivity for Multiple Drugs")
    print("="*50 + "\n")
    
    for drug in drugs_to_test:
        print(f"--- Diagnosing: {drug} ---")
        found_casing = []
        
        # Check Lowercase
        filter_lower = Filter(must=[FieldCondition(key="drug_name", match=MatchValue(value=drug.lower()))])
        matches, _ = client.scroll(collection_name="spl_children", scroll_filter=filter_lower, limit=1)
        if matches:
            found_casing.append("lowercase")
            for i, point in enumerate(matches): # Iterate through matches (will be at most 1 due to limit=1)
                raw_text = point.payload.get('raw_text', '') or ""
                rxcui = point.payload.get('rxcui', 'MISSING')
                print(f"    [LOWERCASE] RxCUI: {rxcui} | Text: {raw_text[:100]}...")
                print(f"        ParentID: {point.payload.get('parent_id')}")
        
        # Check Title Case
        filter_title = Filter(must=[FieldCondition(key="drug_name", match=MatchValue(value=drug.capitalize()))])
        matches, _ = client.scroll(collection_name="spl_children", scroll_filter=filter_title, limit=1)
        if matches: found_casing.append("Title Case")
        
        # Check UPPER CASE
        filter_upper = Filter(must=[FieldCondition(key="drug_name", match=MatchValue(value=drug.upper()))])
        matches, _ = client.scroll(collection_name="spl_children", scroll_filter=filter_upper, limit=1)
        if matches: found_casing.append("UPPER CASE")
        
        if not found_casing:
             print(f"[FAIL] No hits for ANY casing of '{drug}'. Drug might be missing.")
        else:
             print(f"[INFO] Found casing: {found_casing}")
             if "lowercase" not in found_casing:
                 print(f"[ALERT] Lowercase query '{drug}' will FAIL (Case Sensitivity Confirmed)")

if __name__ == "__main__":
    debug_direct_qdrant()
