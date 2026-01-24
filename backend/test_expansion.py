"""
Test query expansion to see why different word orders produce different results.
"""
from app.agents.query_expander import QueryExpander

expander = QueryExpander()

test_queries = [
    "what is gravol used for",
    "what is the use of gravol",
    "what is gravol",
    "whats gravol used for",
]

print("="*80)
print("QUERY EXPANSION COMPARISON")
print("="*80)

for query in test_queries:
    print(f"\n📝 Original: '{query}'")
    variants = expander.expand_query(query)
    print(f"   Variants generated ({len(variants)}):")
    for i, v in enumerate(variants, 1):
        print(f"   {i}. {v}")
    print()
