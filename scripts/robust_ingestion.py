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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.spl_xml_parser import SPLXMLParser
from ingestion.hierarchical_chunking import HierarchicalChunker
from normalization.rxnorm_integration import DrugNormalizer
from retrieval.hybrid_retriever import DenseEmbedder, SparseEmbedder
from vector_db.hierarchical_qdrant import HierarchicalQdrantManager
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
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
FAILED_FILES_LOG = "failed_files.log"


def get_already_ingested_files(qm: HierarchicalQdrantManager) -> Set[str]:
    """
    Query Qdrant to get list of already ingested files.
    Returns set of basenames (e.g., "drug.xml")
    """
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Scroll through all parents to get unique source files
        # We'll use the 'set_id' field as a proxy for unique files
        result = client.scroll(
            collection_name="spl_parents",
            limit=10000,  # Assuming we don't have more than 10k unique files
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
        chunker = HierarchicalChunker()
        normalizer = DrugNormalizer()
        
        logger.info("Loading embedding model...")
        embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
        sparse_embedder = SparseEmbedder(corpus=None)
        
        qm = HierarchicalQdrantManager(host=QDRANT_HOST, port=QDRANT_PORT)
        
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
            parents, children = chunker.chunk_document(metadata, sections)
            logger.info(f"  → Generated {len(parents)} parents, {len(children)} children")
            
            if len(children) == 0:
                logger.warning(f"  ⚠ WARNING: Zero children generated for {basename}")
                # We'll still ingest parents, but log this
            
            # E. Enrich with RxNorm
            if metadata.drug_name:
                drug_info = normalizer.normalize_drug_name(metadata.drug_name)
                if drug_info and 'rxcui' in drug_info:
                    metadata.rxcui = drug_info['rxcui']
                    for p in parents: p.rxcui = metadata.rxcui
                    for c in children: c.rxcui = metadata.rxcui
                    logger.info(f"  → RxCUI: {metadata.rxcui}")
            
            # F. Embed (Children Only)
            if children:
                child_texts = [child.sentence_text for child in children]
                logger.info(f"  → Generating embeddings for {len(child_texts)} chunks...")
                
                child_embeddings = embedder.embed(child_texts)
                sparse_vectors = sparse_embedder.embed_documents(child_texts)
                
                # G. Upsert Children (with retry)
                children_dicts = [c.to_dict() for c in children]
                retry_upsert(
                    qm.upsert_children,
                    children=children_dicts,
                    dense_embeddings=child_embeddings,
                    sparse_embeddings=sparse_vectors
                )
                logger.info(f"  ✓ Upserted {len(children)} children")
            
            # H. Upsert Parents (with retry)
            if parents:
                parents_dicts = [p.to_dict() for p in parents]
                retry_upsert(qm.upsert_parents, parents=parents_dicts)
                logger.info(f"  ✓ Upserted {len(parents)} parents")
            
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

