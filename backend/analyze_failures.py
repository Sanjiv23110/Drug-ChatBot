"""
Analyze the 16 NO_RESULT failures to identify patterns.
"""
import json

# Load results
with open('axid_test_results.json', 'r') as f:
    data = json.load(f)

# Extract NO_RESULT failures
no_result_failures = []
for category, results in data.items():
    for result in results:
        if result.get('path_used') == 'NO_RESULT':
            no_result_failures.append({
                'category': category,
                'question': result['question']
            })

print("\n" + "="*80)
print(f"NO_RESULT FAILURE ANALYSIS ({len(no_result_failures)} failures)")
print("="*80 + "\n")

# Group by category
category_stats = {}
for failure in no_result_failures:
    cat = failure['category']
    category_stats[cat] = category_stats.get(cat, 0) + 1

print("Failures by Category:")
for cat, count in sorted(category_stats.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")

print("\n" + "="*80)
print("DETAILED FAILURE LIST")
print("="*80 + "\n")

for failure in no_result_failures:
    print(f"[{failure['category']}]")
    print(f"  Q: {failure['question']}")
    print()

# Pattern analysis
print("="*80)
print("PATTERN ANALYSIS")
print("="*80 + "\n")

# Check for common query patterns
patterns = {
    "proper name": [],
    "therapeutic": [],
    "clinical trial": [],
    "frequency": [],
    "serious": [],
    "rare": [],
    "herbal": [],
    "food": [],
    "elderly": [],
    "geriatric": [],
    "renal": [],
    "report": [],
    "general info": []  # catch-all
}

for failure in no_result_failures:
    q = failure['question'].lower()
    matched = False
    
    if 'proper name' in q or 'therapeutic class' in q:
        patterns['proper name'].append(failure)
        matched = True
    if 'therapeutic' in q:
        patterns['therapeutic'].append(failure)
        matched = True
    if 'frequency' in q or '≥1%' in failure['question']:
        patterns['frequency'].append(failure)
        matched = True
    if 'serious' in q or 'hematologic' in q:
        patterns['serious'].append(failure)
        matched = True
    if 'rare' in q or 'hypersensitivity' in q:
        patterns['rare'].append(failure)
        matched = True
    if 'nervous system' in q:
        patterns['serious'].append(failure)
        matched = True
    if 'herbal' in q:
        patterns['herbal'].append(failure)
        matched = True
    if 'food' in q:
        patterns['food'].append(failure)
        matched = True
    if 'elderly' in q or 'geriatric' in q:
        patterns['elderly'].append(failure)
        matched = True
    if 'renal impairment' in q:
        patterns['renal'].append(failure)
        matched = True
    if 'report' in q:
        patterns['report'].append(failure)
        matched = True
    if not matched:
        patterns['general info'].append(failure)

print("Query Patterns:")
for pattern, failures in patterns.items():
    if failures:
        print(f"\n{pattern.upper()} ({len(failures)} queries):")
        for f in failures:
            print(f"  - {f['question']}")

print("\n" + "="*80)
print("ROOT CAUSE HYPOTHESES")
print("="*80 + "\n")

print("""
1. GENERIC QUERIES (no clear section mapping):
   - "proper name and therapeutic class" 
   - "report immediately"
   
   These queries don't match any standard section patterns.
   FIX: Add fallback to retrieve ALL sections or map to "description"/"summary" sections.

2. SUBSECTION QUERIES (too specific):
   - "frequency ≥1%" 
   - "serious hematologic"
   - "nervous system" effects
   
   These are asking about SUBSECTIONS within "adverse effects".
   FIX: Map these to parent section "adverse effects" or "adverse reactions".

3. MISSING PATTERN COVERAGE:
   - "elderly patients" / "geriatric" 
   - "renal impairment precautions"
   - "herbal interactions"
   - "food interactions"
   
   These patterns exist but aren't in SECTION_PATTERNS.
   FIX: Add these to the IntentClassifier SECTION_PATTERNS.

4. NEGATION/ABSENCE QUERIES:
   - "Which drugs have NO interactions"
   - "Are herbal interactions ESTABLISHED"
   
   These are asking about ABSENCE of information.
   FIX: Still route to "interactions" section, let LLM handle the negation.
""")
