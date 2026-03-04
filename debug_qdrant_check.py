
import httpx
import json

def check_rxcui(rxcui):
    url = "http://localhost:6333/collections/spl_children/points/scroll"
    payload = {
        "filter": {
            "must": [
                {
                    "key": "rxcui",
                    "match": {
                        "value": rxcui
                    }
                }
            ]
        },
        "limit": 1,
        "with_payload": True
    }
    
    try:
        response = httpx.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        points = data.get("result", {}).get("points", [])
        if points:
            print(f"FOUND: {len(points)} points for RxCUI {rxcui}")
            print("Sample payload:")
            print(json.dumps(points[0]['payload'], indent=2))
        else:
            print(f"NOT FOUND: No points for RxCUI {rxcui}")
            
    except Exception as e:
        print(f"Error querying Qdrant: {e}")

if __name__ == "__main__":
    print("Checking for 'Meperidine' in drug_name...")
    
    url = "http://localhost:6333/collections/spl_children/points/scroll"
    payload = {
        "filter": {
            "must": [
                {
                    "key": "drug_name",
                    "match": {
                        "text": "Meperidine"
                    }
                }
            ]
        },
        "limit": 1,
        "with_payload": True
    }
    
    try:
        response = httpx.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        points = data.get("result", {}).get("points", [])
        if points:
            print(f"FOUND: {len(points)} points payload for Meperidine")
            print(json.dumps(points[0]['payload'], indent=2))
        else:
            print("NOT FOUND: No points with drug_name 'Meperidine'")
            
    except Exception as e:
        print(f"Error querying Qdrant: {e}")
