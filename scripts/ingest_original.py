"""
Production-Grade Ingestion Pipeline
Features:
- Checkpointing (skip already ingested files)
- Fault tolerance (errors don't crash entire batch)
- Memory management (explicit cleanup)
- Connection resilience (retry logic)
"""
import os
import glob
import logging
import sys
import gc
import traceback
from typing import List, Set
from pathlib import Path

from ingestion.spl_xml_parser import SPLXMLParser
from ingestion.chunking_strategy import Chunker
from normalization.rxnorm_integration import DrugNormalizer
from retrieval.hybrid_retriever import DenseEmbedder, SparseEmbedder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('robust_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = r"C:\G\solomindUS\data\xml"
XSLT_PATH = r"C:\G\solomindUS\data\spl.xsl"
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
FAILED_FILES_LOG = "failed_files.log"


def get_already_ingested_files(qm: HierarchicalQdrantManager) -> Set[str]:
    """
    Query Qdrant to get list of already ingested files.
    Returns set of basenames (e.g., "drug.xml")
    """
    try:
        if QDRANT_URL and QDRANT_API_KEY:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Scroll through all parents to get unique source files
        # We'll use the 'set_id' field as a proxy for unique files
        result = client.scroll(
            collection_name="spl_chunks",
            limit=10000,
            with_payload=True
        )
        
        unique_set_ids = set()
        for point in result[0]:
            if 'set_id' in point.payload:
                unique_set_ids.add(point.payload['set_id'])
        
        logger.info(f"Found {len(unique_set_ids)} unique documents already in Qdrant")
        return unique_set_ids
        
    except Exception as e:
        logger.warning(f"Could not query existing files: {e}")
        logger.warning("Will proceed without checkpointing.")
        return set()


def is_file_ingested(metadata, already_ingested: Set[str]) -> bool:
    """Check if this file's set_id is already in the database"""
    return metadata.set_id in already_ingested


def retry_upsert(func, *args, max_retries=3, **kwargs):
    """Retry wrapper for Qdrant upsert operations"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Upsert attempt {attempt + 1} failed: {e}. Retrying...")
                continue
            else:
                raise


def main():
    logger.info("="*80)
    logger.info("ROBUST INGESTION PIPELINE - STARTING")
    logger.info("="*80)
    
    # 1. Verify Environment
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory not found: {DATA_DIR}")
        return
    
    if not os.path.exists(XSLT_PATH):
        logger.warning(f"FDA XSLT stylesheet not found at {XSLT_PATH}")
        logger.warning("Table structures will NOT be preserved correctly.")
    
    # 2. Initialize Components (ONCE)
    try:
        logger.info("Initializing components...")
        parser = SPLXMLParser(xsl_path=XSLT_PATH if os.path.exists(XSLT_PATH) else None)
        chunker = Chunker(chunk_size=500, overlap=50)
        normalizer = DrugNormalizer()
        
        logger.info("Loading embedding model...")
        embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
        sparse_embedder = SparseEmbedder(corpus=None)
        
        # Original Flat QdrantManager
        from vector_db.qdrant_manager import QdrantManager
        qm = QdrantManager(
            host=QDRANT_HOST, 
            port=QDRANT_PORT,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name="spl_chunks"
        )
        
        # Ensure collections exist (do NOT recreate if they already exist)
        try:
            qm.create_collections(dense_vector_size=768, recreate=False)
            logger.info("Collections verified/created")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("Collections already exist, proceeding...")
            else:
                raise
                
    except Exception as e:
        logger.critical(f"Failed to initialize components: {e}")
        traceback.print_exc()
        return
    
    # 3. Get Checkpoint (already ingested files)
    already_ingested = get_already_ingested_files(qm)
    
    # 4. Process Files
    xml_files = glob.glob(os.path.join(DATA_DIR, "*.xml"))
    logger.info(f"Found {len(xml_files)} XML files in {DATA_DIR}")
    
    stats = {
        'total': len(xml_files),
        'skipped': 0,
        'success': 0,
        'failed': 0
    }
    
    failed_files = []
    
    for idx, xml_file in enumerate(xml_files, 1):
        basename = os.path.basename(xml_file)
        logger.info(f"\n[{idx}/{len(xml_files)}] Processing: {basename}")
        
        try:
            # A. Parse (to get set_id for checkpoint check)
            metadata, sections = parser.parse_document(xml_file)
            
            # B. Checkpoint Check
            if is_file_ingested(metadata, already_ingested):
                logger.info(f"  ✓ SKIPPED: {basename} (already ingested, set_id={metadata.set_id})")
                stats['skipped'] += 1
                continue
            
            # C. Validate basic requirements
            if not metadata.drug_name:
                logger.warning(f"  ⚠ WARNING: No drug name found in {basename}, skipping")
                stats['skipped'] += 1
                continue
                
            if len(sections) == 0:
                logger.warning(f"  ⚠ WARNING: No sections found in {basename}, skipping")
                stats['skipped'] += 1
                continue
            
            logger.info(f"  → Parsed {len(sections)} sections. Drug: {metadata.drug_name}")
            
            # D. Chunk
            chunks = chunker.chunk_document(metadata, sections)
            logger.info(f"  → Generated {len(chunks)} chunks")
            
            if len(chunks) == 0:
                logger.warning(f"  ⚠ WARNING: Zero chunks generated for {basename}")
                stats['skipped'] += 1
                continue
            
            # E. Enrich with RxNorm
            if metadata.drug_name:
                drug_info = normalizer.normalize_drug_name(metadata.drug_name)
                if drug_info and 'rxcui' in drug_info:
                    metadata.rxcui = drug_info['rxcui']
                    for c in chunks: 
                        c.metadata['rxcui'] = metadata.rxcui
                    logger.info(f"  → RxCUI: {metadata.rxcui}")
            
            # F. Embed and Upsert
            texts = [c.text for c in chunks]
            logger.info(f"  → Generating embeddings for {len(texts)} chunks...")
            
            dense_vectors = embedder.embed(texts)
            sparse_vectors = sparse_embedder.embed_documents(texts)
            
            chunks_dicts = [{"text": c.text, "metadata": c.metadata} for c in chunks]
            retry_upsert(
                qm.upsert,
                chunks=chunks_dicts,
                dense_embeddings=dense_vectors,
                sparse_embeddings=sparse_vectors
            )
            logger.info(f"  ✓ Upserted {len(chunks)} chunks")
            
            logger.info(f"  ✓ SUCCESS: {basename}")
            stats['success'] += 1
            
        except Exception as e:
            logger.error(f"  ✗ FAILED: {basename}")
            logger.error(f"    Error: {str(e)}")
            traceback.print_exc()
            stats['failed'] += 1
            failed_files.append(basename)
            
            # Log to failed files
            with open(FAILED_FILES_LOG, 'a') as f:
                f.write(f"{basename}\t{str(e)}\n")
        
        finally:
            # I. Memory Cleanup
            gc.collect()
    
    # 5. Final Report
    logger.info("\n" + "="*80)
    logger.info("INGESTION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total files:    {stats['total']}")
    logger.info(f"Skipped:        {stats['skipped']} (already ingested)")
    logger.info(f"Success:        {stats['success']}")
    logger.info(f"Failed:         {stats['failed']}")
    
    if failed_files:
        logger.warning(f"\nFailed files: {failed_files}")
        logger.warning(f"See {FAILED_FILES_LOG} for details")
    else:
        logger.info("\n✓ All files processed successfully!")




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"\n\nFATAL ERROR: Ingestion pipeline crashed!")
        logger.critical(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

