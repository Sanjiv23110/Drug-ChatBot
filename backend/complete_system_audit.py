"""
COMPLETE SYSTEM AUDIT
=====================
Comprehensive check of the entire RAG pipeline.
"""
import sys
sys.path.insert(0, r'C:\G\Maclens chatbot w api\backend')

import asyncio
from app.database.session import get_session
from app.database.models import MonographSection
from sqlalchemy import select, func, or_

async def audit():
    print("\n" + "="*80)
    print("COMPLETE SYSTEM AUDIT")
    print("="*80 + "\n")
    
    # ===== 1. DATABASE CONTENT CHECK =====
    print("1. DATABASE CONTENT CHECK")
    print("-" * 80)
    
    async with get_session() as session:
        # Check total drugs
        drug_count_stmt = select(func.count(func.distinct(MonographSection.drug_name)))
        result = await session.execute(drug_count_stmt)
        total_drugs = result.scalar()
        print(f"✓ Total unique drugs: {total_drugs}")
        
        # List all drugs
        drugs_stmt = select(MonographSection.drug_name).distinct().limit(10)
        result = await session.execute(drugs_stmt)
        drugs = [r[0] for r in result.fetchall()]
        print(f"✓ Drugs: {drugs}")
        
        # Check AXID specifically
        print(f"\n📊 AXID Data Audit:")
        axid_count_stmt = select(func.count()).where(
            or_(
                MonographSection.drug_name.ilike('%axid%'),
                MonographSection.brand_name.ilike('%axid%'),
                MonographSection.generic_name.ilike('%niz%')  # nizatidine
            )
        )
        result = await session.execute(axid_count_stmt)
        axid_sections = result.scalar()
        print(f"   Total AXID sections: {axid_sections}")
        
        if axid_sections == 0:
            print("   ❌ CRITICAL: NO AXID DATA FOUND!")
            print("   This explains ALL failures.")
            return
        
        # Get section names for AXID
        sections_stmt = select(MonographSection.section_name).where(
            or_(
                MonographSection.drug_name.ilike('%axid%'),
                MonographSection.brand_name.ilike('%axid%')
            )
        ).distinct()
        result = await session.execute(sections_stmt)
        section_names = [r[0] for r in result.fetchall()]
        print(f"   AXID sections ({len(section_names)} unique):")
        for s in sorted(section_names)[:20]:  # Show first 20
            print(f"     - {s}")
        
        # Check for specific important sections
        print(f"\n🔍 Critical Section Check:")
        critical_sections = {
            'indications': ['indication', 'use', 'therapeutic'],
            'dosage': ['dosage', 'dose', 'administration'],
            'contraindications': ['contraindication'],
            'warnings': ['warning', 'precaution'],
            'adverse': ['adverse', 'side effect'],
            'pharmacology': ['pharmacology', 'pharmacokinetic'],
            'interactions': ['interaction'],
        }
        
        for key, keywords in critical_sections.items():
            found = []
            for s in section_names:
                if any(kw in s.lower() for kw in keywords):
                    found.append(s)
            status = "✅" if found else "❌"
            print(f"   {status} {key}: {len(found)} sections")
            if found:
                print(f"       {found[:3]}")  # Show first 3
        
        # Check for half-life data
        print(f"\n💊 Half-Life Data Check:")
        pharmacology_stmt = select(MonographSection.content_text).where(
            or_(
                MonographSection.drug_name.ilike('%axid%'),
                MonographSection.brand_name.ilike('%axid%')
            )
        ).where(
            MonographSection.section_name.ilike('%pharmaco%')
        ).limit(5)
        result = await session.execute(pharmacology_stmt)
        pharm_contents = [r[0] for r in result.fetchall()]
        
        has_half_life = False
        for content in pharm_contents:
            if content and 'half' in content.lower():
                has_half_life = True
                # Extract snippet
                idx = content.lower().index('half')
                snippet = content[max(0, idx-50):idx+150]
                print(f"   ✅ Found 'half-life': ...{snippet}...")
                break
        
        if not has_half_life:
            print(f"   ❌ NO 'half-life' found in pharmacology sections!")
            print(f"   This explains pharmacology query failures.")
    
    # ===== 2. INTENT CLASSIFICATION TEST =====
    print(f"\n2. INTENT CLASSIFICATION TEST")
    print("-" * 80)
    
    from app.retrieval.intent_classifier import IntentClassifier
    classifier = IntentClassifier(use_llm_fallback=False)  # Rule-based only
    
    test_queries = [
        "What is the half-life of axid?",
        "What are the contraindications of axid?",
        "What is axid used for?",
        "Can axid be taken with food?",
    ]
    
    for query in test_queries:
        intent = classifier.classify(query)
        print(f"\nQuery: '{query}'")
        print(f"  Drug: {intent.target_drug} (conf: {intent.drug_confidence:.2f})")
        print(f"  Section: {intent.target_section} (conf: {intent.section_confidence:.2f})")
        print(f"  Attribute: {intent.target_attribute} (conf: {intent.attribute_confidence:.2f})")
    
    # ===== 3. RETRIEVAL TEST =====
    print(f"\n3. RETRIEVAL ENGINE TEST")
    print("-" * 80)
    
    from app.retrieval.retrieve import RetrievalEngine
    engine = RetrievalEngine(enable_vector_fallback=False)
    
    for query in test_queries[:2]:  # Test first 2
        print(f"\nQuery: '{query}'")
        result = await engine.retrieve(query)
        print(f"  Path: {result.path_used.value}")
        print(f"  Sections: {result.total_results}")
        if result.sections:
            print(f"  Section names: {[s['section_name'] for s in result.sections[:3]]}")
    
    # ===== 4. DRUG NAME VARIATIONS =====
    print(f"\n4. DRUG NAME VARIATIONS IN DB")
    print("-" * 80)
    
    async with get_session() as session:
        # Check all name fields for AXID-related entries
        for field in ['drug_name', 'brand_name', 'generic_name']:
            stmt = select(getattr(MonographSection, field)).distinct().limit(20)
            result = await session.execute(stmt)
            values = [r[0] for r in result.fetchall() if r[0]]
            print(f"\n{field}: {len(values)} unique values")
            for v in sorted(values)[:10]:
                print(f"  - {v}")
    
    print(f"\n{'='*80}")
    print("AUDIT COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(audit())
