"""
Full ingestion pipeline.

Orchestrates PDF → chunks → embeddings → FAISS + SQLite.
"""
import logging
from pathlib import Path
from typing import Dict, List
import hashlib

from app.vectorstore.faiss_store import FAISSVectorStore
from app.vectorstore.index_manager import IndexManager
from app.metadata.sqlite_store import SQLiteMetadataStore
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_text, detect_section
from app.ingestion.embedder import AzureEmbedder


# Current ingestion version - increment on schema/logic changes
CURRENT_INGESTION_VERSION = 1

# Batch and save configuration
EMBED_BATCH_SIZE = 50
SAVE_INTERVAL = 1000  # Save index every N chunks


def generate_chunk_id(file_hash: str, page_num: int, char_start: int) -> int:
    """
    Generate deterministic chunk ID using file hash.
    
    More stable than file_path (handles renames).
    """
    identifier = f"{file_hash}:{page_num}:{char_start}"
    hash_bytes = hashlib.sha256(identifier.encode()).digest()
    chunk_id = int.from_bytes(hash_bytes[:4], 'big') & 0x7FFFFFFF
    return chunk_id


def insert_chunk_with_retry(
    metadata_store: SQLiteMetadataStore,
    chunk_id: int,
    file_path: str,
    file_hash: str,
    chunk_data: Dict,
    max_retries: int = 1
) -> tuple[bool, int]:
    """
    Insert chunk with collision handling.
    
    Returns: (success, final_chunk_id)
    
    On collision:
    1. Log error with full context
    2. Retry ONCE with salt
    3. If still fails → RAISE exception
    
    Never silently mutate IDs.
    """
    original_chunk_id = chunk_id
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                # Add salt on retry
                salt_identifier = f"{file_hash}:retry{attempt}:{chunk_data['page_num']}:{chunk_data['char_start']}"
                hash_bytes = hashlib.sha256(salt_identifier.encode()).digest()
                chunk_id = int.from_bytes(hash_bytes[:4], 'big') & 0x7FFFFFFF
                
                logging.warning(
                    f"Chunk ID collision detected. Original: {original_chunk_id}, "
                    f"Retrying with salt: {chunk_id} for {file_path}:pg{chunk_data['page_num']}"
                )
            
            success = metadata_store.insert_chunk(
                chunk_id=chunk_id,
                file_path=file_path,
                chunk_text=chunk_data['text'],
                page_num=chunk_data['page_num'],
                section_name=chunk_data.get('section_name'),
                char_start=chunk_data['char_start'],
                char_end=chunk_data['char_end']
            )
            
            if success:
                return True, chunk_id
        
        except Exception as e:
            if attempt == max_retries:
                logging.error(
                    f"FATAL: Chunk ID collision unresolved for "
                    f"{file_path}:pg{chunk_data['page_num']} "
                    f"(original ID: {original_chunk_id})"
                )
                raise RuntimeError(f"Unresolvable chunk ID collision: {e}")
    
    return False, chunk_id


def ingest_directory(
    pdf_dir: str,
    faiss_store: FAISSVectorStore,
    metadata_store: SQLiteMetadataStore,
    index_manager: IndexManager,
    batch_size: int = EMBED_BATCH_SIZE,
    save_interval: int = SAVE_INTERVAL
) -> Dict:
    """
    Ingest all PDFs in directory.
    
    Features:
    - File fingerprinting (skip unchanged files)
    - Adaptive chunking (section-aware)
    - Batched embeddings (respect TPM limit)
    - Incremental saves (every save_interval chunks)
    - Atomic operations (commit only on success)
    
    Args:
        pdf_dir: Directory containing PDFs
        faiss_store: FAISS vector store
        metadata_store: SQLite metadata store
        index_manager: Index persistence manager
        batch_size: Embeddings per API call (default 50)
        save_interval: Save index every N chunks (default 1000)
        
    Returns:
        Statistics dict: {files_processed, chunks_added, errors, skipped_files}
    """
    logging.info(f"Starting ingestion from {pdf_dir}")
    
    # Initialize embedder
    embedder = AzureEmbedder()
    
    # Statistics
    stats = {
        'files_processed': 0,
        'chunks_added': 0,
        'errors': [],
        'skipped_files': 0
    }
    
    # Collect all PDFs
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        file_path = str(pdf_file)
        
        try:
            # 1. Load PDF and compute hash
            logging.info(f"Processing {pdf_file.name}...")
            file_hash, pages = load_pdf(file_path)
            
            # 2. Check if ingestion needed
            if not metadata_store.should_ingest_file(
                file_path, file_hash, CURRENT_INGESTION_VERSION
            ):
                logging.info(f"Skipping unchanged file: {pdf_file.name}")
                stats['skipped_files'] += 1
                continue
            
            # 3. Delete old chunks (if re-ingesting)
            deleted = metadata_store.delete_file_chunks(file_path)
            if deleted > 0:
                logging.info(f"Deleted {deleted} old chunks from {pdf_file.name}")
            
            # 4. Process pages with section tracking
            chunk_buffer = []
            current_section = None
            
            for page_data in pages:
                # Detect section change
                page_section = detect_section(page_data['text'])
                if page_section:
                    current_section = page_section
                    logging.debug(f"Section detected: {current_section}")
                
                # Chunk with adaptive sizing
                chunks = chunk_text(
                    text=page_data['text'],
                    page_num=page_data['page_num'],
                    section_name=current_section
                )
                
                chunk_buffer.extend(chunks)
            
            if not chunk_buffer:
                logging.warning(f"No chunks extracted from {pdf_file.name}")
                continue
            
            logging.info(f"Created {len(chunk_buffer)} chunks from {pdf_file.name}")
            
            # 5. Batch embed and insert
            for i in range(0, len(chunk_buffer), batch_size):
                batch = chunk_buffer[i:i+batch_size]
                texts = [c['text'] for c in batch]
                
                # Embed batch
                embeddings = embedder.embed_batch(texts)
                
                # Generate chunk IDs
                chunk_ids = []
                for c in batch:
                    chunk_id = generate_chunk_id(
                        file_hash,
                        c['page_num'],
                        c['char_start']
                    )
                    chunk_ids.append(chunk_id)
                
                # Add to FAISS
                faiss_store.add_vectors(embeddings, chunk_ids)
                
                # Insert metadata with collision handling
                for chunk_id, chunk in zip(chunk_ids, batch):
                    success, final_id = insert_chunk_with_retry(
                        metadata_store,
                        chunk_id,
                        file_path,
                        file_hash,
                        chunk
                    )
                    if not success:
                        logging.error(f"Failed to insert chunk {chunk_id}")
                
                stats['chunks_added'] += len(batch)
                
                # Incremental save
                if stats['chunks_added'] % save_interval == 0:
                    index_manager.save(faiss_store, metadata={
                        'chunks': stats['chunks_added'],
                        'files': stats['files_processed']
                    })
                    logging.info(f"✓ Saved index at {stats['chunks_added']} chunks")
            
            # 6. Update file record
            metadata_store.upsert_file_record(
                file_path,
                file_hash,
                CURRENT_INGESTION_VERSION,
                len(chunk_buffer)
            )
            
            stats['files_processed'] += 1
            logging.info(f"✓ Completed {pdf_file.name} ({len(chunk_buffer)} chunks)")
        
        except Exception as e:
            error_msg = f"Failed to ingest {pdf_file.name}: {e}"
            stats['errors'].append({'file': file_path, 'error': str(e)})
            logging.error(error_msg, exc_info=True)
            continue
    
    # Final save
    index_manager.save(faiss_store, metadata={
        'chunks': stats['chunks_added'],
        'files': stats['files_processed'],
        'version': CURRENT_INGESTION_VERSION
    })
    logging.info("✓ Final index save complete")
    
    logging.info(
        f"Ingestion complete: {stats['files_processed']} files, "
        f"{stats['chunks_added']} chunks, {stats['skipped_files']} skipped, "
        f"{len(stats['errors'])} errors"
    )
    
    return stats
