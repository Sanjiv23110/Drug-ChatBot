"""
Simple PDF ingestion script.

Usage:
  python ingest.py

Place PDFs in data/pdfs/ directory.
"""
from app.ingestion.ingest_pipeline import ingest_directory
from pathlib import Path

# Create directories if they don't exist
Path("data/pdfs").mkdir(parents=True, exist_ok=True)
Path("data/faiss").mkdir(parents=True, exist_ok=True)

print("=== PDF Ingestion ===\n")
print("Looking for PDFs in: data/pdfs/\n")

# Check if PDFs exist
pdf_files = list(Path("data/pdfs").glob("*.pdf"))

if not pdf_files:
    print("❌ No PDF files found in data/pdfs/")
    print("\nTo add PDFs:")
    print("  1. Copy your drug monograph PDFs to: data/pdfs/")
    print("  2. Run this script again")
    exit(1)

print(f"Found {len(pdf_files)} PDF(s):\n")
for pdf in pdf_files:
    print(f"  • {pdf.name}")

print(f"\nStarting ingestion...\n")

# Initialize stores
try:
    from app.vectorstore.faiss_store import FAISSVectorStore
    from app.vectorstore.index_manager import IndexManager
    from app.metadata.sqlite_store import SQLiteMetadataStore
    
    # Create store instances
    faiss_store = FAISSVectorStore(dimension=1536)
    metadata_store = SQLiteMetadataStore(db_path="data/metadata.db")
    index_manager = IndexManager(index_dir="data/faiss/medical_index")
    
    # Load existing index if it exists
    if index_manager.exists():
        print("Loading existing index...")
        loaded = index_manager.load(dimension=1536)
        if loaded:
            faiss_store.index, faiss_store.chunk_ids, _ = loaded
            print(f"✓ Loaded existing index with {faiss_store.count()} vectors\n")
    
    # Run ingestion and capture stats
    stats = ingest_directory(
        pdf_dir="data/pdfs",
        faiss_store=faiss_store,
        metadata_store=metadata_store,
        index_manager=index_manager
    )
    
    print("\n=== Ingestion Complete ===")
    
    # Show what happened this run
    if stats['skipped_files'] > 0:
        print(f"⏭️  Skipped {stats['skipped_files']} unchanged file(s) (fingerprint match)")
    
    if stats['files_processed'] > 0:
        print(f"✓ Processed {stats['files_processed']} file(s)")
        print(f"✓ Added {stats['chunks_added']} new chunks")
    
    if stats['files_processed'] == 0 and stats['skipped_files'] > 0:
        print("✓ No processing needed - all files unchanged!")
    
    # Show database totals
    import sqlite3
    conn = sqlite3.connect("data/metadata.db")
    cursor = conn.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM files")
    total_files = cursor.fetchone()[0]
    conn.close()
    
    print(f"\nDatabase Totals:")
    print(f"  Total Files: {total_files}")
    print(f"  Total Chunks: {total_chunks}")
    print(f"  Avg chunks/file: {total_chunks // total_files if total_files > 0 else 0}")

except Exception as e:
    print(f"\n❌ Ingestion failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
