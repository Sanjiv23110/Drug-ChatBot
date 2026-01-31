"""Test Enhanced Fuzzy Matching (Solution 3C)"""
import requests

BASE_URL = "http://localhost:8000/api/chat"

tests = [
    ("what is Axid used for", "Should match 'what is axid used for'"),
    ("indications of axid", "Should match via keyword fallback"),
    ("contraindications of Axid", "Should work"),
    ("side effects of axid", "Should match"),
]

print("Testing Enhanced Fuzzy Matching...\n")

for query, desc in tests:
    print(f"Query: '{query}'")
    print(f"Expected: {desc}")
    
    try:
        resp = requests.post(BASE_URL, json={"question": query}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("has_answer"):
                print(f"✅ SUCCESS - {data.get('chunks_retrieved')} chunks, path: {data.get('retrieval_path')}\n")
            else:
                print(f"❌ FAIL - No answer\n")
        else:
            print(f"❌ HTTP {resp.status_code}\n")
    except Exception as e:
        print(f"❌ ERROR: {e}\n")
