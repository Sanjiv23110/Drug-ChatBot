#!/usr/bin/env python3
"""
CLI script for ingesting drug monograph PDFs.

Usage:
    python run_ingest.py              # Ingest all PDFs in data/pdfs
    python run_ingest.py /path/to/pdfs  # Ingest from specific directory
    python run_ingest.py --init-db     # Initialize database only
"""
import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from app.db.session import init_db, check_connection, get_db_status
from app.ingestion.ingest import IngestionPipeline, IngestionResult
from app.utils.logging import setup_logging

# Setup logging
logger = setup_logging(level="INFO", json_format=False)


async def initialize_database():
    """Initialize PostgreSQL database with all tables."""
    print("\n🔧 Initializing PostgreSQL database...")
    print("-" * 50)
    
    try:
        # Check connection first
        status = await get_db_status()
        print(f"Database: {status['database']}@{status['host']}")
        
        if not status['connected']:
            print("❌ Cannot connect to PostgreSQL!")
            print("\nMake sure:")
            print("  1. PostgreSQL is running")
            print("  2. Database exists: CREATE DATABASE medical_rag;")
            print("  3. .env has correct credentials")
            print("  4. pgvector extension is installed")
            return False
        
        # Initialize tables
        await init_db()
        print("✅ Database initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False


async def run_ingestion(pdf_dir: str, skip_vision: bool = False):
    """Run the ingestion pipeline on a directory of PDFs."""
    print(f"\n📚 Starting PDF Ingestion")
    print("-" * 50)
    print(f"Source: {pdf_dir}")
    print(f"Vision: {'disabled' if skip_vision else 'enabled (GPT-4o)'}")
    print()
    
    # Check directory exists
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"❌ Directory not found: {pdf_dir}")
        return []
    
    # Count PDFs
    pdf_files = list(pdf_path.glob("*.pdf")) + list(pdf_path.glob("*.PDF"))
    print(f"Found {len(pdf_files)} PDF files")
    print()
    
    if len(pdf_files) == 0:
        print("No PDF files to process.")
        return []
    
    # Run pipeline
    start_time = datetime.now()
    
    pipeline = IngestionPipeline(
        image_output_dir="./data/images",
        skip_vision=skip_vision,
        skip_existing=True
    )
    
    results = await pipeline.ingest_directory(pdf_dir)
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print()
    print("=" * 50)
    print("📊 INGESTION SUMMARY")
    print("=" * 50)
    
    success = sum(1 for r in results if r.success)
    failed = len(results) - success
    total_sections = sum(r.sections_created for r in results)
    total_images = sum(r.images_extracted for r in results)
    total_structures = sum(r.structures_detected for r in results)
    
    print(f"✅ Successful: {success}")
    print(f"❌ Failed: {failed}")
    print(f"📄 Sections created: {total_sections}")
    print(f"🖼️  Images extracted: {total_images}")
    print(f"🧬 Structures detected: {total_structures}")
    print(f"⏱️  Time elapsed: {elapsed:.1f}s")
    
    if failed > 0:
        print("\n⚠️ Failed files:")
        for r in results:
            if not r.success:
                print(f"  - {r.file_name}: {r.error_message}")
    
    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Ingest drug monograph PDFs into PostgreSQL"
    )
    parser.add_argument(
        "pdf_dir",
        nargs="?",
        default="data/pdfs",
        help="Directory containing PDF files (default: data/pdfs)"
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize database only (don't ingest)"
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip chemical structure detection (faster)"
    )
    parser.add_argument(
        "--check-db",
        action="store_true",
        help="Check database connection only"
    )
    
    args = parser.parse_args()
    
    print()
    print("🏥 Medical Drug Monograph Ingestion System")
    print("=" * 50)
    
    # Check database connection
    if args.check_db:
        status = await get_db_status()
        print(f"\nDatabase Status:")
        print(f"  Host: {status['host']}")
        print(f"  Database: {status['database']}")
        print(f"  Connected: {'✅ Yes' if status['connected'] else '❌ No'}")
        print(f"  Pool Size: {status['pool_size']}")
        return
    
    # Initialize database
    if args.init_db or True:  # Always check DB first
        db_ok = await initialize_database()
        if not db_ok:
            print("\n❌ Cannot proceed without database connection.")
            sys.exit(1)
    
    if args.init_db:
        print("\n✅ Database initialization complete.")
        return
    
    # Run ingestion
    results = await run_ingestion(args.pdf_dir, skip_vision=args.skip_vision)
    
    # Exit code based on results
    if all(r.success for r in results):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
