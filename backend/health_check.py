"""
System Health Check for Sunday Demo Preparation
Day 1, Task 1.1 from Implementation Plan
"""
import sys
import requests

print("="*60)
print("SOLOMIND.AI SYSTEM HEALTH CHECK")
print("="*60)

# Check 1: Database
print("\n✓ Check 1: Database")
try:
    from app.metadata.sqlite_store import SQLiteMetadataStore
    store = SQLiteMetadataStore('data/metadata.db')
    count = store.count_chunks()
    print(f"  ✅ Database accessible")
    print(f"  ✅ Total chunks: {count:,}")
except Exception as e:
    print(f"  ❌ Database error: {e}")
    sys.exit(1)

# Check 2: FAISS Index
print("\n✓ Check 2: FAISS Vector Store")
try:
    from app.vectorstore.index_manager import IndexManager
    mgr = IndexManager('data/faiss/medical_index')
    exists = mgr.exists()
    print(f"  ✅ FAISS index exists: {exists}")
    if not exists:
        print("  ❌ FAISS index not found!")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ FAISS error: {e}")
    sys.exit(1)

# Check 3: API Endpoint
print("\n✓ Check 3: API Endpoint")
try:
    response = requests.post(
        'http://localhost:8000/api/chat',
        json={'question': 'What is Gravol?'},
        timeout=10
    )
    
    if response.status_code == 200:
        result = response.json()
        has_answer = result.get('has_answer', False)
        chunks = result.get('chunks_retrieved', 0)
        
        print(f"  ✅ API Status: {response.status_code} OK")
        print(f"  ✅ Answer generated: {has_answer}")
        print(f"  ✅ Chunks retrieved: {chunks}")
        
        if not has_answer:
            print("  ⚠️  Warning: Query returned no answer")
    else:
        print(f"  ❌ API returned status: {response.status_code}")
        sys.exit(1)
        
except requests.exceptions.ConnectionError:
    print("  ❌ Cannot connect to API - is backend running?")
    print("  Run: uvicorn app.main:app --reload")
    sys.exit(1)
except Exception as e:
    print(f"  ❌ API error: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("✅ ALL SYSTEMS GREEN - READY FOR DEMO PREP!")
print("="*60)
print("\nNext steps from implementation plan:")
print("  1. Create test query list (20 queries)")
print("  2. Test Tier 1 queries (5 critical queries)")
print("  3. Take screenshots of good results")
print("\nTime: ~2 hours remaining for Day 1 tasks")
print("="*60)
