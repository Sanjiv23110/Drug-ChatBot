"""
Debug why "what is gravol used for" returns no information.
"""
import logging
logging.basicConfig(level=logging.INFO)

from app.metadata.sqlite_store import SQLiteMetadataStore
from app.agents.query_expander import QueryExpander
from app.resolver.drug_name_resolver import DrugNameResolver

# Setup
metadata_store = SQLiteMetadataStore('data/metadata.db')
resolver = DrugNameResolver('data/metadata.db')
expander = QueryExpander(drug_resolver=resolver)

# Test queries
test_queries = [
    "what is the use of gravol",  # This works
    "what is gravol used for",    # This doesn't work
    "what are the uses of gravol", # This doesn't work
]

print("="*80)
print("COMPREHENSIVE DEBUG: Why different queries fail")
print("="*80)

for query in test_queries:
    print(f"\n{'='*80}")
    print(f"📝 Original Query: '{query}'")
    print("="*80)
    
    # Step 1: Normalization
    normalized = query.lower().strip().rstrip('?.!,;:')
    normalized = ' '.join(normalized.split())
    print(f"  ✓ Normalized: '{normalized}'")
    
    # Step 2: Drug extraction
    drugs = resolver.extract_drug_names(normalized)
    print(f"  ✓ Drugs extracted: {drugs}")
    
    # Step 3: Check if it matches general info pattern
    patterns = ['what is', 'tell me about', 'information', 'describe', 
                'used for', 'use of', 'indication', 'purpose']
    matches = [p for p in patterns if p in normalized]
    print(f"  ✓ Pattern matches: {matches}")
    
    # Step 4: Query expansion
    variants = expander.expand_query(normalized)
    print(f"  ✓ Expanded to {len(variants)} variants:")
    for i, v in enumerate(variants, 1):
        print(f"     {i}. {v}")
    
    # Step 5: Check database for "indications" 
    if drugs:
        drug = drugs[0]
        print(f"\n  🔍 Database check for '{drug}':")
        
        # Try to find chunks about indications
        import sqlite3
        conn = sqlite3.connect('data/metadata.db')
        
        # Check what sections exist for this drug
        cursor = conn.execute("""
            SELECT DISTINCT section_name 
            FROM chunks 
            WHERE file_path LIKE ? OR file_path LIKE ?
        """, (f'%{drug}%', f'%{drug.upper()}%'))
        sections = [row[0] for row in cursor.fetchall()]
        print(f"     Sections in DB: {sections[:5]}...")
        
        # Check for indication-related chunks
        cursor = conn.execute("""
            SELECT COUNT(*) 
            FROM chunks 
            WHERE (file_path LIKE ? OR file_path LIKE ?)
            AND (chunk_text LIKE '%indication%' OR chunk_text LIKE '%use%' 
                 OR section_name LIKE '%INDICATION%')
        """, (f'%{drug}%', f'%{drug.upper()}%'))
        count = cursor.fetchone()[0]
        print(f"     Chunks with 'indication/use': {count}")
        
        conn.close()
    
    print()

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
