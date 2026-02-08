
import httpx
import json

def count_files_in_db():
    url = "http://localhost:6333/collections/spl_parents/points/scroll"
    # We want to iterate through all points and count unique 'xml_file' or 'drug_name'
    # Actually, simpler: just count total points first to see volume
    
    # 1. Get Collection Stats
    try:
        stats_url = "http://localhost:6333/collections/spl_parents"
        resp = httpx.get(stats_url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        print(f"Collection 'spl_parents' status: {data['result']['status']}")
        print(f"Total Parent Chunks (Approx): {data['result']['points_count']}")
    except Exception as e:
        print(f"Error getting stats: {e}")
        return

    # 2. Scroll to find unique files (Scan first 500 points to sample)
    try:
        payload = {
            "limit": 1000,
            "with_payload": True
        }
        resp = httpx.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        points = data.get("result", {}).get("points", [])
        unique_files = set()
        unique_drugs = set()
        
        for p in points:
            pl = p.get('payload', {})
            # Depending on what we stored. run_ingestion.py stores:
            # metadata.drug_name, set_id, spl_version etc.
            # It might NOT store the filename explicitly in payload unless we added it.
            # Let's check 'drug_name' which is definitely there.
            if 'drug_name' in pl:
                unique_drugs.add(pl['drug_name'])
                
            if 'set_id' in pl:
                unique_files.add(pl['set_id'])
                
        print("\n--- Content Summary (Sampled 1000 chunks) ---")
        print(f"Unique Drug Names Found: {len(unique_drugs)}")
        print(f"Unique Set IDs Found: {len(unique_files)}")
        print("\nDrugs Present:")
        for d in sorted(list(unique_drugs)):
            print(f" - {d}")
            
    except Exception as e:
        print(f"Error scrolling points: {e}")

if __name__ == "__main__":
    count_files_in_db()
