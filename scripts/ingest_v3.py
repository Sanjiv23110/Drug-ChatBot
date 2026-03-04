import os
import sys
import glob
import logging
import traceback
import gc
import json
from typing import List, Dict, Set, Tuple
from dotenv import load_dotenv

# FORCING UTF-8 OUTPUT
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Add parent directory to path to allow imports
# Resolves to c:\G\solomindUS if script is in c:\G\solomindUS\scripts
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
print(f"DEBUG: Added {ROOT_DIR} to sys.path")

from ingestion.spl_xml_parser import SPLXMLParser, SPLMetadata, SPLSection
from ingestion.hierarchical_chunking import HierarchicalChunker, ParentChunk, ChildChunk
from normalization.rxnorm_integration import DrugNormalizer
from vector_db.hierarchical_qdrant import HierarchicalQdrantManager
from retrieval.hybrid_retriever import DenseEmbedder, SparseEmbedder
from qdrant_client import QdrantClient

# Configure Logger - STREAM ONLY, ASCII SAFE
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "xml")
XSLT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "spl_stylesheet.xsl")
FAILED_FILES_LOG = "ingestion_failures.log"

# Qdrant Config
load_dotenv()

def get_already_ingested_files(qm: HierarchicalQdrantManager) -> Set[str]:
    try:
        client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
        result = client.scroll(
            collection_name="spl_parents",
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
        return set()

def is_file_ingested(metadata, already_ingested: Set[str]) -> bool:
    return metadata.set_id in already_ingested

def retry_upsert(func, *args, max_retries=3, **kwargs):
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
    print("=== ULTRA-SAFE INGESTION V3 START ===")
    
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory not found: {DATA_DIR}")
        return
    
    try:
        logger.info("Initializing components...")
        parser = SPLXMLParser(xsl_path=XSLT_PATH if os.path.exists(XSLT_PATH) else None)
        chunker = HierarchicalChunker()
        normalizer = DrugNormalizer()
        
        logger.info("Loading embedding models...")
        embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
        sparse_embedder = SparseEmbedder(corpus=None)
        
        qm = HierarchicalQdrantManager()
        try:
            qm.create_collections(dense_vector_size=768, recreate=False)
        except Exception as e:
            if "already exists" in str(e).lower():
                pass
            else:
                raise

    except Exception as e:
        logger.critical(f"Failed to initialize components: {e}")
        traceback.print_exc()
        return

    logger.info("Checking checkpoints...")
    already_ingested = get_already_ingested_files(qm)
    
    xml_files = glob.glob(os.path.join(DATA_DIR, "*.xml"))
    logger.info(f"Found {len(xml_files)} files.")
    
    stats = {'total': len(xml_files), 'skipped': 0, 'success': 0, 'failed': 0}
    
    for idx, xml_file in enumerate(xml_files, 1):
        basename = os.path.basename(xml_file)
        print(f"[{idx}/{len(xml_files)}] Processing: {basename}")
        
        try:
            metadata, sections = parser.parse_document(xml_file)
            
            if is_file_ingested(metadata, already_ingested):
                print(f"  [SKIPPED] {basename} (Already Ingested)")
                stats['skipped'] += 1
                continue
                
            if not metadata.drug_name:
                print("  [SKIP] No Drug Name")
                stats['skipped'] += 1
                continue
            
            print(f"  -> Drug: {metadata.drug_name} | Sections: {len(sections)}")
            
            parents, children = chunker.chunk_document(metadata, sections)
            
            # Enrich
            if metadata.drug_name:
                drug_info = normalizer.normalize_drug_name(metadata.drug_name)
                if drug_info and 'rxcui' in drug_info:
                    metadata.rxcui = drug_info['rxcui']
                    for p in parents: p.rxcui = metadata.rxcui
                    for c in children: c.rxcui = metadata.rxcui
            
            # Embed & Upsert
            if children:
                child_texts = [c.sentence_text for c in children]
                child_embeddings = embedder.embed(child_texts)
                sparse_vectors = sparse_embedder.embed_documents(child_texts)
                
                children_dicts = [c.to_dict() for c in children]
                retry_upsert(qm.upsert_children, children=children_dicts, dense_embeddings=child_embeddings, sparse_embeddings=sparse_vectors)
                print(f"  -> Upserted {len(children)} children.")
            
            if parents:
                parents_dicts = [p.to_dict() for p in parents]
                retry_upsert(qm.upsert_parents, parents=parents_dicts)
                print(f"  -> Upserted {len(parents)} parents.")
                
            stats['success'] += 1
            print(f"  [SUCCESS] {basename}")
            
        except Exception as e:
            print(f"  [FAILED] {basename}: {e}")
            stats['failed'] += 1
            with open(FAILED_FILES_LOG, 'a') as f:
                f.write(f"{basename}: {str(e)}\n")
        finally:
            gc.collect()

    print("\n=== INGESTION COMPLETE ===")
    print(f"Stats: {stats}")

if __name__ == "__main__":
    main()
