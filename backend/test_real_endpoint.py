"""
Test the ACTUAL /api/chat endpoint (not the router directly).
This simulates what the frontend does.
"""
import requests
import json
from datetime import datetime

# Test questions subset
TEST_QUESTIONS = [
    "What is the proper name and therapeutic class of AXID?",
    "What dosage forms and strengths are available for AXID?",
    "What conditions is AXID indicated for?",
    "According to the AXID product monograph, which patients are contraindicated from receiving AXID?",
    "What is the recommended adult dose for acute duodenal ulcer?",
    "Why should malignancy be excluded before initiating AXID therapy?",
    "Is AXID safe for use during pregnancy?",
    "What are the most commonly reported adverse reactions to AXID?",
    "Which drugs have no observed interactions with AXID?",
    "What is the mechanism of action of nizatidine?",
    "What is the approximate elimination half-life of nizatidine?",
    "What are the recommended storage conditions for AXID?",
   "What symptoms are associated with AXID overdose?",
    "How is AXID explained to patients in the Patient Medication Information?",
    "Is AXID approved for treating bacterial infections?",
    "Why does renal impairment require AXID dose adjustment based on its pharmacokinetics?",
]

API_URL = "http://127.0.0.1:8000/api/chat"

print("\n" + "="*80)
print("REAL ENDPOINT TEST (via /api/chat)")
print("="*80)
print(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")

results = {
    "has_answer": 0,
    "not_found": 0,
    "try_rephrasing": 0,
    "errors": 0,
    "total": len(TEST_QUESTIONS)
}

for i, question in enumerate(TEST_QUESTIONS, 1):
    print(f"[{i}/{len(TEST_QUESTIONS)}] {question[:60]}...")
    
    try:
        response = requests.post(API_URL, json={"question": question}, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        answer = data['answer']
        has_answer = data['has_answer']
        path = data.get('retrieval_path', 'Unknown')
        chunks = data.get('chunks_retrieved', 0)
        
        # Classify response
        if has_answer:
            results["has_answer"] += 1
            status = "✅ HAS ANSWER"
        elif "not found in available monographs" in answer.lower():
            results["not_found"] += 1
            status = "⚠️  NOT FOUND"
        elif "try rephrasing" in answer.lower():
            results["try_rephrasing"] += 1
            status = "❌ TRY REPHRASING"
        else:
            status = "❓ UNKNOWN"
        
        print(f"   {status} | Path: {path} | Chunks: {chunks}")
        print(f"   Answer: {answer[:100]}...")
        
    except Exception as e:
        results["errors"] += 1
        print(f"   ❌ ERROR: {e}")

print(f"\n{'='*80}")
print("REAL ENDPOINT RESULTS")
print(f"{'='*80}\n")

print(f"Total Questions: {results['total']}")
print(f"Has Answer: {results['has_answer']} ({results['has_answer']/results['total']*100:.1f}%)")
print(f"'Not Found in Monographs': {results['not_found']} ({results['not_found']/results['total']*100:.1f}%)")
print(f"'Try Rephrasing': {results['try_rephrasing']} ({results['try_rephrasing']/results['total']*100:.1f}%)")
print(f"Errors: {results['errors']}")

print(f"\n✅ Success Rate: {results['has_answer']}/{results['total']} ({results['has_answer']/results['total']*100:.1f}%)")
print(f"❌ Failure Rate: {results['total'] - results['has_answer']}/{results['total']} ({(results['total'] - results['has_answer'])/results['total']*100:.1f}%)")
