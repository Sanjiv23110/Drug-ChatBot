import os
import glob
import logging
import sys
from typing import List

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.spl_xml_parser import SPLXMLParser
from ingestion.hierarchical_chunking import HierarchicalChunker
from normalization.rxnorm_integration import DrugNormalizer
from retrieval.hybrid_retriever import DenseEmbedder, SparseEmbedder
from vector_db.hierarchical_qdrant import HierarchicalQdrantManager
from qdrant_client.models import SparseVector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = r"C:\G\solomindUS\data\xml" # User specified subdirectory for XMLs
XSLT_PATH = r"C:\G\solomindUS\data\spl.xsl"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

def main():
    logger.info("Starting ingestion process...")

    # 1. Verify Environment
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory not found: {DATA_DIR}")
        return

    # Check for SPL XSLT
    if not os.path.exists(XSLT_PATH):
        logger.warning(f"FDA XSLT stylesheet not found at {XSLT_PATH}.") 
        logger.warning("Table structures will NOT be preserved correctly.")
        # We'll proceed but this is a critical warning for the user
        # In strict mode we might want to exit, but for setup let's warn.
    
    # 2. Initialize Components
    try:
        # Parser
        parser = SPLXMLParser(xsl_path=XSLT_PATH if os.path.exists(XSLT_PATH) else None)
        
        # Chunker
        chunker = HierarchicalChunker()
        
        # Normalizer
        normalizer = DrugNormalizer()
        
        # Embedder (Dense)
        logger.info("Loading embedding model...")
        embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
        
        # Embedder (Sparse) - for keyword search
        sparse_embedder = SparseEmbedder(corpus=None) # No corpus needed for document embedding (uses hashing)
        
        # Vector DB
        qm = HierarchicalQdrantManager(host=QDRANT_HOST, port=QDRANT_PORT)
        # Recreate collections for a fresh start? Let's make it optional or idempotent.
        # For this script, we'll ensure they exist.
        qm.create_collections(dense_vector_size=768, recreate=True)  # Force recreation for testing
        
    except Exception as e:
        logger.critical(f"Failed to initialize components: {e}")
        return

    # 3. Process Files
    xml_files = glob.glob(os.path.join(DATA_DIR, "*.xml"))
    logger.info(f"Found {len(xml_files)} XML files in {DATA_DIR}")

    for xml_file in xml_files:
        # try:  # TEMPORARILY DISABLED - Let errors bubble up!
        logger.info(f"Processing: {os.path.basename(xml_file)}")
        
        # A. Parse
        metadata, sections = parser.parse_document(xml_file)
        print(f"\n=== DEBUG: PARSING ===")
        print(f"File: {os.path.basename(xml_file)}")
        print(f"Sections parsed: {len(sections)}")
        print(f"Drug name: {metadata.drug_name}")
        print(f"NDC: {metadata.ndc_codes}")
        
        # B. Chunk (Hierarchical)
        parents, children = chunker.chunk_document(metadata, sections)
        print(f"\n=== DEBUG: CHUNKING ===")
        print(f"Parents generated: {len(parents)}")
        print(f"Children generated: {len(children)}")
        if children:
            print(f"Sample child: {children[0].sentence_text[:100]}...")
        logger.info(f"  - Generated {len(parents)} parents, {len(children)} children")
        
        # C. Enrich (RxNorm)
        # Only if drug name exists
        if metadata.drug_name:
            drug_info = normalizer.normalize_drug_name(metadata.drug_name)
            if drug_info and 'rxcui' in drug_info:
                metadata.rxcui = drug_info['rxcui']
                # Propagate to chunks
                for p in parents: p.rxcui = metadata.rxcui
                for c in children: c.rxcui = metadata.rxcui
        
        # D. Embed (Children Only)
        print(f"\n=== DEBUG: CHECKING CHILDREN ===")
        print(f"Children count: {len(children)}")
        print(f"Parents count: {len(parents)}")
        
        if children:
            print(f"[DEBUG] Entering children embedding block...")
            child_texts = [child.sentence_text for child in children]
            print(f"\n=== DEBUG: EMBEDDING ===")
            print(f"Texts to embed: {len(child_texts)}")
            print(f"First child text preview: {child_texts[0][:80] if child_texts else 'NONE'}...")
            logger.info("  - Generating embeddings (Dense + Sparse)...")
            
            # Dense
            print(f"[DEBUG] Calling embedder.embed()...")
            child_embeddings = embedder.embed(child_texts)
            print(f"Dense embeddings shape: {child_embeddings.shape}")
            
            # Sparse (Keyword)
            print(f"[DEBUG] Calling sparse_embedder.embed_documents()...")
            sparse_vectors = sparse_embedder.embed_documents(child_texts)
            print(f"Sparse vectors count: {len(sparse_vectors)}")
            
            # E. Upsert
            print(f"\n=== DEBUG: UPSERT CHILDREN ===")
            print(f"Attempting to upsert {len(children)} children...")
            print(f"[DEBUG] Converting children to dicts...")
            children_dicts = [c.to_dict() for c in children]
            print(f"[DEBUG] First child dict keys: {list(children_dicts[0].keys()) if children_dicts else 'NONE'}")
            print(f"[DEBUG] Calling qm.upsert_children()...")
            
            # NO TRY/EXCEPT - Let errors bubble up!
            qm.upsert_children(
                children=children_dicts,
                dense_embeddings=child_embeddings,
                sparse_embeddings=sparse_vectors
            )
            print(f"[OK] Successfully upserted {len(children)} children")
        
        if parents:
            print(f"\n=== DEBUG: UPSERT PARENTS ===")
            print(f"Attempting to upsert {len(parents)} parents...")
            print(f"[DEBUG] Converting parents to dicts...")
            parents_dicts = [p.to_dict() for p in parents]
            print(f"[DEBUG] First parent dict keys: {list(parents_dicts[0].keys()) if parents_dicts else 'NONE'}")
            print(f"[DEBUG] Calling qm.upsert_parents()...")
            
            # NO TRY/EXCEPT - Let errors bubble up!
            qm.upsert_parents(parents=parents_dicts)
            print(f"[OK] Successfully upserted {len(parents)} parents")
            
        logger.info(f"  - Successfully ingested {os.path.basename(xml_file)}")


        # TEMPORARILY DISABLED - Let errors bubble up to see what's failing!
        # except Exception as e:
        #     logger.error(f"Failed to process {xml_file}: {e}")
        #     continue

    logger.info("Ingestion complete.")

if __name__ == "__main__":
    main()
