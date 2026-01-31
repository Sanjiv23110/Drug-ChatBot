
import asyncio
import re
from app.retrieval.intent_classifier import IntentClassifier
from app.db.session import get_session
from sqlalchemy import text

async def research_issues():
    print("--- 1. RESEARCHING 'ALL CONTRAINDICATIONS' FAILURE ---")
    classifier = IntentClassifier()
    
    # Test Query 1: The one that failed
    query_failed = "all contraindications of APO-METOPROLOL"
    intent_failed = classifier.classify(query_failed)
    print(f"Query: '{query_failed}'")
    print(f"Extracted Drug: '{intent_failed.target_drug}'")
    print(f"Extracted Section: '{intent_failed.target_section}'")
    
    # Test Query 2: The one that worked (partially)
    query_worked = "contraindications of APO-METOPROLOL"
    intent_worked = classifier.classify(query_worked)
    print(f"\nQuery: '{query_worked}'")
    print(f"Extracted Drug: '{intent_worked.target_drug}'")
    print(f"Extracted Section: '{intent_worked.target_section}'")

    print("\n--- 2. RESEARCHING 'PARTIAL ANSWER' (DATABASE INSPECTION) ---")
    # We need to see all sections for this drug to see if "Contraindications" is split
    # Note: We'll search for 'apo-metoprolol' in brand_name or generic_name
    async with get_session() as session:
        # Find the drug first to get the exact name used in DB
        stmt = text("""
            SELECT DISTINCT drug_name, brand_name, generic_name 
            FROM monograph_sections 
            WHERE brand_name ILIKE '%metoprolol%' OR drug_name ILIKE '%metoprolol%'
        """)
        result = await session.execute(stmt)
        drugs = result.fetchall()
        print(f"Found related drugs in DB: {drugs}")

        if drugs:
            target_drug = drugs[0].drug_name # Use the first match's canonical name
            print(f"\nInspecting sections for drug: '{target_drug}'")
            
            # Fetch all sections that might be related to contraindications
            stmt = text("""
                SELECT section_name, original_header, LEFT(content_text, 100) as preview, length(content_text) as len
                FROM monograph_sections 
                WHERE drug_name = :drug
                ORDER BY id ASC
            """)
            result = await session.execute(stmt, {"drug": target_drug})
            rows = result.fetchall()
            
            print(f"\nTotal Sections Found: {len(rows)}")
            print("Sections sequence:")
            for row in rows:
                # Flag potential contraindication parts
                pointer = "  "
                if "contra" in str(row.section_name).lower() or "contra" in str(row.original_header).lower():
                    pointer = "->"
                print(f"{pointer} Header: '{row.original_header}' | Norm: '{row.section_name}' | Len: {row.len} | Preview: {row.preview}")

if __name__ == "__main__":
    asyncio.run(research_issues())
