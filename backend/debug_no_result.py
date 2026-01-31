"""
Debug why NO_RESULT is happening when all-sections fallback exists.
"""
import sys
sys.path.insert(0, r'C:\G\Maclens chatbot w api\backend')

import asyncio
import json
from app.retrieval.intent_classifier import IntentClassifier
from app.retrieval.retrieve import RetrievalEngine

# Load the 16 failing questions
with open('axid_test_results.json', 'r') as f:
    data = json.load(f)

fails = [(cat, r['question']) for cat, results in data.items() for r in results if r.get('path_used') == 'NO_RESULT']

print("\n" + "="*80)
print(f"DEBUGGING {len(fails)} NO_RESULT FAILURES")
print("="*80)

async def debug():
    classifier = IntentClassifier(use_llm_fallback=False)  # Rule-based only
    engine = RetrievalEngine(enable_vector_fallback=False)
    
    for i, (cat, question) in enumerate(fails[:5], 1):  # Test first 5
        print(f"\n{i}. [{cat}] {question}")
        print("-" * 80)
        
        # Step 1: Intent classification
        intent= classifier.classify(question)
        print(f"Intent Classification:")
        print(f"  Drug: '{intent.target_drug}' (conf: {intent.drug_confidence:.2f})")
        print(f"  Section: '{intent.target_section}' (conf: {intent.section_confidence:.2f})")
        print(f"  Attribute: '{intent.target_attribute}' (conf: {intent.attribute_confidence:.2f})")
        
        # Step 2: Check retrieval routing logic
        print(f"\nRetrieval Routing Analysis:")
        if intent.target_drug and intent.target_section and not intent.target_attribute:
            print(f"  → Should use Path A (SQL_MATCH): drug + section")
        elif intent.target_drug and intent.target_attribute:
            print(f"  → Should use Path A++ (ATTRIBUTE_LOOKUP): drug + attribute")
        elif intent.target_drug and not intent.target_section and not intent.target_attribute:
            print(f"  → Should use Path A (partial): ALL SECTIONS for drug")
        elif not intent.target_drug:
            print(f"  → NO_RESULT: No drug identified")
        
        # Step 3: Actual retrieval
        result = await engine.retrieve(question)
        print(f"\nActual Retrieval Result:")
        print(f"  Path Used: {result.path_used.value}")
        print(f"  Sections Retrieved: {result.total_results}")
        if result.sections:
            print(f"  Section names: {[s['section_name'] for s in result.sections[:3]]}")

if __name__ == "__main__":
    asyncio.run(debug())
