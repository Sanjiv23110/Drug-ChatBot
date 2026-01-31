"""
Script to clean the database by truncating content tables.
Does NOT drop tables, only removes data to allow fresh ingestion.
"""
import asyncio
import sys
from sqlalchemy import text
from app.db.session import get_session

async def clean_database():
    force = len(sys.argv) > 1 and sys.argv[1] == "--force"
    
    if not force:
        print("⚠️  WARNING: This will delete ALL data from the following tables:")
        print("   - monograph_sections")
        print("   - section_mappings")
        print("   - image_classifications")
        print("   - drug_metadata")
        print("   - ingestion_logs")
        
        confirm = input("\nType 'DELETE' to confirm: ")
        if confirm != "DELETE":
            print("❌ Operation cancelled.")
            return

    async with get_session() as session:
        print("\n🗑️  Cleaning database...")
        
        # Disable foreign key checks temporarily to allow truncation
        await session.execute(text("SET session_replication_role = 'replica';"))
        
        try:
            # Truncate tables
            # Using CASCADE to handle dependent tables if any
            tables = [
                "monograph_sections",
                "section_mappings",
                "image_classifications",
                "drug_metadata", 
                "ingestion_logs"
            ]
            
            for table in tables:
                try:
                    await session.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
                    print(f"   - Truncated {table}")
                except Exception as table_err:
                    print(f"   - Note: Could not truncate {table} (might not exist): {table_err}")

            # Commit changes
            await session.commit()
            print("✅ Database cleaned successfully.")
            
        except Exception as e:
            print(f"❌ Error cleaning database: {e}")
            await session.rollback()
        finally:
            # Re-enable foreign key checks
            await session.execute(text("SET session_replication_role = 'origin';"))

if __name__ == "__main__":
    asyncio.run(clean_database())
