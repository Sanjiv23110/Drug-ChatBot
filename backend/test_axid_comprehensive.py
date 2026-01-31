"""
Comprehensive AXID RAG Test Suite

Tests 50+ questions across 14 categories to identify systemic failures.
"""
import sys
sys.path.insert(0, r'C:\G\Maclens chatbot w api\backend')

import asyncio
import json
from datetime import datetime
from app.retrieval.router import RetrievalRouter
from app.generation.answer_generator import AnswerGenerator

# Test suite organized by category
TEST_SUITE = {
    "1_BASIC_FACTS": [
        "What is the proper name and therapeutic class of AXID?",
        "What dosage forms and strengths are available for AXID?",
        "What is the route of administration for AXID?",
        "Who is the manufacturer of AXID?",
        "What is the date of last revision of the AXID product monograph?",
    ],
    "2_INDICATIONS": [
        "What conditions is AXID indicated for?",
        "Is AXID indicated for gastroesophageal reflux disease?",
        "Is AXID approved for pediatric use?",
        "What is AXID used for in maintenance therapy?",
        "Is AXID indicated for prophylactic use in duodenal ulcers?",
    ],
    "3_CONTRAINDICATIONS": [
        "According to the AXID product monograph, which patients are contraindicated from receiving AXID?",
        "Is AXID contraindicated in patients with hypersensitivity to other H2-receptor antagonists?",
        "Are renal impairment or hepatic impairment listed as contraindications?",
        "Does AXID have any container-component related contraindications?",
    ],
    "4_DOSAGE": [
        "What is the recommended adult dose for acute duodenal ulcer?",
        "What dosing regimen is recommended for GERD?",
        "How should AXID dosing be adjusted in patients with severe renal impairment?",
        "What should a patient do if a dose of AXID is missed?",
        "Can AXID be taken with antacids?",
    ],
    "5_WARNINGS": [
        "Why should malignancy be excluded before initiating AXID therapy?",
        "What vitamin deficiency may occur with long-term AXID use?",
        "What laboratory test interference is associated with AXID?",
        "Why should AXID dosage be adjusted in elderly patients?",
        "What precautions are advised for patients with renal impairment?",
    ],
    "6_SPECIAL_POPULATIONS": [
        "Is AXID safe for use during pregnancy?",
        "Should AXID be used during breastfeeding?",
        "What is Health Canada's position on pediatric use of AXID?",
        "Are there safety differences in geriatric patients?",
    ],
    "7_ADVERSE_REACTIONS": [
        "What are the most commonly reported adverse reactions to AXID?",
        "Which adverse reactions occurred at a frequency of ≥1% in clinical trials?",
        "What serious hematologic adverse effects have been reported?",
        "What rare hypersensitivity reactions are associated with AXID?",
        "Which adverse reactions may affect the nervous system?",
    ],
    "8_DRUG_INTERACTIONS": [
        "Which drugs have no observed interactions with AXID?",
        "Does AXID inhibit cytochrome P450 enzymes?",
        "How does high-dose aspirin interact with AXID?",
        "Are food interactions clinically significant for AXID?",
        "Are herbal drug interactions established for AXID?",
    ],
    "9_PHARMACOLOGY": [
        "What is the mechanism of action of nizatidine?",
        "How does AXID affect nocturnal gastric acid secretion?",
        "What is the approximate elimination half-life of nizatidine?",
        "How is nizatidine primarily eliminated from the body?",
        "What percentage of nizatidine is excreted unchanged in urine?",
    ],
    "10_STORAGE": [
        "What are the recommended storage conditions for AXID?",
        "How should unused AXID capsules be disposed of?",
    ],
    "11_OVERDOSAGE": [
        "What symptoms are associated with AXID overdose?",
        "Is AXID effectively removed by renal dialysis?",
        "What is the recommended management for suspected AXID overdose?",
    ],
    "12_PATIENT_INFO": [
        "How is AXID explained to patients in the Patient Medication Information?",
        "What side effects should patients report immediately?",
        "What should patients tell their healthcare provider before taking AXID?",
    ],
    "13_HALLUCINATION_TESTS": [
        "Is AXID approved for treating bacterial infections?",
        "Does AXID contain any opioid ingredients?",
        "Is AXID indicated for cancer treatment?",
        "Does AXID have any FDA black-box warnings?",
        "Is AXID approved for intravenous administration?",
    ],
    "14_CROSS_SECTION": [
        "Why does renal impairment require AXID dose adjustment based on its pharmacokinetics?",
        "How do AXID's elimination pathway and geriatric warnings relate?",
        "Why can long-term AXID use lead to vitamin B12 deficiency?",
        "Why is dialysis ineffective in clearing AXID overdose?",
    ],
}

async def run_comprehensive_test():
    """Run all tests and generate report."""
    
    print("\n" + "="*80)
    print("AXID COMPREHENSIVE RAG TEST SUITE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    router = RetrievalRouter()
    generator = AnswerGenerator()
    
    results = {}
    total_questions = sum(len(qs) for qs in TEST_SUITE.values())
    current = 0
    
    for category, questions in TEST_SUITE.items():
        print(f"\n{'='*80}")
        print(f"Category: {category}")
        print(f"{'='*80}\n")
        
        category_results = []
        
        for question in questions:
            current += 1
            print(f"[{current}/{total_questions}] Testing: {question[:60]}...")
            
            try:
                # Retrieve
                context, raw_result = await router.route_with_result(question)
                
                # Generate answer
                if context.sources:
                    answer_result = generator.generate(question, context.sources)
                    answer = answer_result['answer']
                    has_answer = answer_result['has_answer']
                else:
                    answer = "No sections retrieved"
                    has_answer = False
                
                # Analyze result
                result = {
                    "question": question,
                    "path_used": context.path_used,
                    "sections_count": context.total_chunks,
                    "sections_found": context.sections_found,
                    "has_answer": has_answer,
                    "answer_preview": answer[:200] if answer else "",
                    "success": context.total_chunks > 0 and has_answer
                }
                
                category_results.append(result)
                
                # Print summary
                status = "✓" if result['success'] else "✗"
                print(f"   {status} Path: {context.path_used} | Sections: {context.total_chunks} | Answer: {has_answer}")
                
            except Exception as e:
                print(f"   ✗ ERROR: {e}")
                category_results.append({
                    "question": question,
                    "error": str(e),
                    "success": False
                })
        
        results[category] = category_results
        
        # Category summary
        success_count = sum(1 for r in category_results if r.get('success', False))
        print(f"\n{category} Summary: {success_count}/{len(questions)} passed ({success_count/len(questions)*100:.1f}%)")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}\n")
    
    all_results = [r for cat in results.values() for r in cat]
    total_success = sum(1 for r in all_results if r.get('success', False))
    
    print(f"Total Questions: {total_questions}")
    print(f"Passed: {total_success} ({total_success/total_questions*100:.1f}%)")
    print(f"Failed: {total_questions - total_success} ({(total_questions - total_success)/total_questions*100:.1f}%)")
    
    # Failure analysis
    print(f"\n{'='*80}")
    print("FAILURE ANALYSIS")
    print(f"{'='*80}\n")
    
    no_retrieval = [r for r in all_results if r.get('sections_count', 0) == 0]
    wrong_answer = [r for r in all_results if r.get('sections_count', 0) > 0 and not r.get('has_answer', False)]
    
    print(f"No Sections Retrieved: {len(no_retrieval)}")
    print(f"Sections Retrieved but No Answer: {len(wrong_answer)}")
    
    # Path usage statistics
    path_stats = {}
    for r in all_results:
        path = r.get('path_used', 'ERROR')
        path_stats[path] = path_stats.get(path, 0) + 1
    
    print(f"\nPath Usage:")
    for path, count in sorted(path_stats.items(), key=lambda x: -x[1]):
        print(f"  {path}: {count} ({count/total_questions*100:.1f}%)")
    
    # Save detailed results
    with open('axid_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Detailed results saved to: axid_test_results.json")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())
