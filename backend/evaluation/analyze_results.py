"""
Automated RAGas Results Analyzer

Simply run: python evaluation/analyze_results.py

This script automatically:
1. Loads RAGas evaluation results
2. Compares to medical-grade targets
3. Shows pass/fail status
4. Highlights problematic questions
5. Generates actionable recommendations
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Medical system targets
TARGETS = {
    'faithfulness': 0.90,
    'answer_relevancy': 0.85,
    'context_precision': 0.80,
    'context_recall': 0.75
}

def load_results():
    """Load evaluation results and test set."""
    results_path = Path("evaluation/rag_evaluation_results.json")
    test_set_path = Path("evaluation/medical_test_set.csv")
    
    if not results_path.exists():
        print("❌ No results found. Run 'python evaluation/evaluate_rag.py' first.")
        return None, None
    
    with open(results_path) as f:
        results = json.load(f)
    
    test_set = pd.read_csv(test_set_path) if test_set_path.exists() else None
    
    return results, test_set

def analyze_scores(results):
    """Analyze overall scores vs targets."""
    summary = results.get('summary', {})
    
    print("="*70)
    print("RAG SYSTEM EVALUATION REPORT")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Overall status
    all_pass = all(summary.get(metric, 0) >= target for metric, target in TARGETS.items())
    
    print("OVERALL STATUS:")
    print("-"*70)
    if all_pass:
        status = "✅ PASS - System meets medical-grade quality standards"
        status_emoji = "🟢"
    else:
        status = "❌ FAIL - System needs improvement before deployment"
        status_emoji = "🔴"
    
    print(f"{status_emoji} {status}")
    print()
    
    # Individual metrics
    print("METRIC SCORES:")
    print("-"*70)
    print(f"{'Metric':<25} {'Score':<10} {'Target':<10} {'Status':<10}")
    print("-"*70)
    
    failed_metrics = []
    
    for metric, target in TARGETS.items():
        score = summary.get(metric, 0)
        passed = score >= target
        status_icon = "✓" if passed else "✗"
        
        print(f"{metric.replace('_', ' ').title():<25} {score:<10.3f} {target:<10.2f} {status_icon}")
        
        if not passed:
            failed_metrics.append((metric, score, target))
    
    print("-"*70)
    print()
    
    return all_pass, failed_metrics, summary

def analyze_failed_questions(results, test_set, failed_metrics):
    """Find questions with lowest scores for failed metrics."""
    if not failed_metrics or test_set is None:
        return
    
    detailed = pd.DataFrame(results.get('detailed_scores', []))
    
    if detailed.empty:
        return
    
    print("PROBLEMATIC QUESTIONS:")
    print("-"*70)
    
    for metric, score, target in failed_metrics:
        gap = target - score
        print(f"\n{metric.replace('_', ' ').title()} (Score: {score:.3f}, Gap: {gap:.3f})")
        print("-"*70)
        
        if metric in detailed.columns:
            # Get bottom 3 questions for this metric
            low_scores = detailed.nsmallest(3, metric)
            
            for idx, row in low_scores.iterrows():
                if idx < len(test_set):
                    question = test_set.iloc[idx]['question']
                    q_score = row[metric]
                    
                    print(f"\n  Question #{idx+1}: {question[:60]}...")
                    print(f"  Score: {q_score:.3f}")
                    
                    # Diagnosis
                    if metric == 'faithfulness' and q_score < 0.90:
                        print("  ⚠️  Issue: Potential hallucination or incorrect information")
                    elif metric == 'answer_relevancy' and q_score < 0.85:
                        print("  ⚠️  Issue: Answer doesn't directly address the question")
                    elif metric == 'context_precision' and q_score < 0.80:
                        print("  ⚠️  Issue: Wrong chunks retrieved or ranked poorly")
                    elif metric == 'context_recall' and q_score < 0.75:
                        print("  ⚠️  Issue: Missing relevant information")
    
    print()

def generate_recommendations(failed_metrics, summary):
    """Generate actionable recommendations."""
    if not failed_metrics:
        print("RECOMMENDATIONS:")
        print("-"*70)
        print("✅ No action needed - all metrics meet targets!")
        print()
        print("Next steps:")
        print("  • Save this as baseline for regression testing")
        print("  • Re-run evaluation after major changes")
        print("  • Use for validation when scaling to 1000 PDFs")
        print()
        return
    
    print("RECOMMENDATIONS:")
    print("-"*70)
    
    for metric, score, target in failed_metrics:
        gap = target - score
        
        print(f"\n{metric.replace('_', ' ').title()} (Gap: {gap:.3f}):")
        
        if metric == 'faithfulness':
            print("  1. Review prompts - may be too permissive")
            print("  2. Re-enable stricter validation gates")
            print("  3. Check if chunks contain sufficient context")
            print("  4. Verify citations are accurate")
        
        elif metric == 'answer_relevancy':
            print("  1. Improve system prompt to focus on question")
            print("  2. Review if questions are too ambiguous")
            print("  3. Check if retrieval is finding right topics")
            print("  4. Adjust temperature (currently 0.0)")
        
        elif metric == 'context_precision':
            print("  1. Tune embedding model parameters")
            print("  2. Adjust reranking weights")
            print("  3. Review section priority settings")
            print("  4. Check if chunk size is appropriate")
        
        elif metric == 'context_recall':
            print("  1. Increase context window (currently 10-20 chunks)")
            print("  2. Improve chunking strategy")
            print("  3. Check for cross-referenced information")
            print("  4. Verify section detection during ingestion")
    
    print()

def main():
    print()
    
    # Load data
    results, test_set = load_results()
    
    if results is None:
        return
    
    # Analyze scores
    all_pass, failed_metrics, summary = analyze_scores(results)
    
    # Show problematic questions if any
    if failed_metrics and test_set is not None:
        analyze_failed_questions(results, test_set, failed_metrics)
    
    # Generate recommendations
    generate_recommendations(failed_metrics, summary)
    
    # Final summary
    print("="*70)
    if all_pass:
        print("🎉 READY FOR DEPLOYMENT")
    else:
        print("⚠️  NEEDS IMPROVEMENT - Address recommendations above")
    print("="*70)
    print()

if __name__ == "__main__":
    main()
