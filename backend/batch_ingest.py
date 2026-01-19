"""
Batch ingestion with progress tracking and checkpoints.

Usage:
    python batch_ingest.py

Features:
- Progress tracking
- Time estimates
- Memory monitoring
- Automatic checkpoints
"""
import time
import psutil
from pathlib import Path
from app.vectorstore.faiss_store import FAISSVectorStore
from app.vectorstore.index_manager import IndexManager
from app.metadata.sqlite_store import SQLiteMetadataStore
from app.ingestion.ingest_pipeline import ingest_directory


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def batch_ingest_with_monitoring(pdf_dir: str = "data/pdfs"):
    """
    Ingest PDFs with comprehensive monitoring.
    """
    print("=" * 60)
    print("BATCH PDF INGESTION WITH MONITORING")
    print("=" * 60)
    
    # Initialize stores
    print("\nInitializing stores...")
    faiss_store = FAISSVectorStore(dimension=1536)
    metadata_store = SQLiteMetadataStore("data/metadata.db")
    index_manager = IndexManager("data/faiss/medical_index")
    
    # Load existing
    existing_count = 0
    if index_manager.exists():
        print("Loading existing index...")
        loaded = index_manager.load(dimension=1536)
        if loaded:
            faiss_store.index, faiss_store.chunk_ids, _ = loaded
            existing_count = faiss_store.count()
            print(f"✓ Loaded index with {existing_count} existing vectors")
    
    # Count PDFs
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    total_pdfs = len(pdf_files)
    
    print(f"\nFound {total_pdfs} PDFs in {pdf_dir}")
    
    if total_pdfs == 0:
        print("❌ No PDFs found. Add PDFs to data/pdfs/ directory.")
        return
    
    # Memory check
    mem_before = get_memory_usage()
    print(f"Memory usage before: {mem_before:.1f} MB")
    
    # Confirm
    print(f"\n{'=' * 60}")
    print(f"Ready to ingest {total_pdfs} PDFs")
    print(f"Estimated time: {total_pdfs * 12 / 60:.1f} minutes")
    print(f"{'=' * 60}")
    
    input("\nPress Enter to start ingestion...")
    
    print("\nStarting ingestion...\n")
    start_time = time.time()
    
    # Ingest
    try:
        stats = ingest_directory(
            pdf_dir=pdf_dir,
            faiss_store=faiss_store,
            metadata_store=metadata_store,
            index_manager=index_manager
        )
        
        elapsed = time.time() - start_time
        mem_after = get_memory_usage()
        
        print(f"\n{'=' * 60}")
        print("INGESTION COMPLETE")
        print(f"{'=' * 60}")
        
        print(f"\nTime Statistics:")
        print(f"  Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
        print(f"  Avg per file: {elapsed/max(stats['files_processed'], 1):.1f} seconds")
        
        if stats['files_processed'] > 0:
            print(f"  Processing rate: {stats['files_processed']/(elapsed/60):.1f} files/minute")
        
        print(f"\nIngestion Statistics:")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Files skipped: {stats['skipped_files']} (unchanged)")
        print(f"  Chunks added: {stats['chunks_added']}")
        
        if stats['errors']:
            print(f"  Errors: {len(stats['errors'])}")
            for error in stats['errors'][:5]:  # Show first 5
                print(f"    - {error}")
        
        print(f"\nMemory Usage:")
        print(f"  Before: {mem_before:.1f} MB")
        print(f"  After: {mem_after:.1f} MB")
        print(f"  Increase: {mem_after - mem_before:.1f} MB")
        
        # Database stats
        import sqlite3
        conn = sqlite3.connect("data/metadata.db")
        cursor = conn.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]
        cursor = conn.execute("SELECT COUNT(*) FROM files")
        total_files = cursor.fetchone()[0]
        conn.close()
        
        print(f"\nDatabase Totals:")
        print(f"  Total files: {total_files}")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Avg chunks/file: {total_chunks // total_files if total_files > 0 else 0}")
        print(f"  FAISS vectors: {faiss_store.count()}")
        
        if total_chunks != faiss_store.count():
            print(f"\n⚠️  WARNING: Chunk count mismatch!")
            print(f"  SQLite: {total_chunks}, FAISS: {faiss_store.count()}")
        else:
            print(f"\n✓ Database consistency verified")
        
        print(f"\n{'=' * 60}")
        print("Ready for queries!")
        print(f"{'=' * 60}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Ingestion interrupted by user")
        print(f"Progress saved. Re-run to continue from where you left off.")
    
    except Exception as e:
        print(f"\n\n❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    batch_ingest_with_monitoring(pdf_dir="data/pdfs")
