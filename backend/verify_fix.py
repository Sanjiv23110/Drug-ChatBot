"""
Script to verify that the APO-METOPROLOL contraindications section is correctly ingested.
"""
import asyncio
from sqlalchemy import text
from app.db.session import get_session

async def verify_fix():
    print("running verification...")
    async with get_session() as session:
        # Find the contraindications section for Metoprolol
        stmt = text("""
            SELECT content_text, section_name
            FROM monograph_sections 
            WHERE drug_name ILIKE '%metoprolol%' 
            AND (section_name = 'contraindications' OR original_header ILIKE '%contraindications%')
        """)
        result = await session.execute(stmt)
        rows = result.fetchall()
        
        if not rows:
            print("❌ No contraindications section found for Metoprolol")
            return

        print(f"Found {len(rows)} contraindications sections")
        
        for row in rows:
            content = row.content_text
            print(f"\n--- Section: {row.section_name} ---")
            print(f"Length: {len(content)} chars")
            
            # Check for bullet points that were previously missing
            missing_items = [
                "Sinus bradycardia",
                "Sick sinus syndrome",
                "Second and third degree A-V block",
                "Right ventricular failure",
                "severe hypotension"
            ]
            
            found_count = 0
            for item in missing_items:
                if item.lower() in content.lower():
                    print(f"✅ Found: {item}")
                    found_count += 1
                else:
                    print(f"❌ Missing: {item}")
            
            if found_count == len(missing_items):
                print("\nSUCCESS: All missing items are now present! Fix verified.")
            else:
                print(f"\nPARTIAL: Found {found_count}/{len(missing_items)} items.")
                print("Content:")
                print(content)
                print("-" * 50)

if __name__ == "__main__":
    asyncio.run(verify_fix())
