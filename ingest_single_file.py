
import os
import logging
from ingestion.spl_xml_parser import SPLXMLParser
from ingestion.hierarchical_chunking import HierarchicalChunker
from normalization.rxnorm_integration import DrugNormalizer
from retrieval.hybrid_retriever import DenseEmbedder, SparseEmbedder
from vector_db.hierarchical_qdrant import HierarchicalQdrantManager

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_file(filename):
    DATA_DIR = os.path.join(os.getcwd(), "data", "xml")
    XSLT_PATH = os.path.join(os.getcwd(), "data", "spl_stylesheet.xsl")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    try:
        # Initialize components
        logger.info("Initializing components...")
        parser = SPLXMLParser(xsl_path=XSLT_PATH if os.path.exists(XSLT_PATH) else None)
        chunker = HierarchicalChunker()
        normalizer = DrugNormalizer()
        
        logger.info("Loading embedding model (this may take a moment)...")
        embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
        sparse_embedder = SparseEmbedder(corpus=None)
        
        qm = HierarchicalQdrantManager(host=QDRANT_HOST, port=QDRANT_PORT)
        # IMPORTANT: recreate=False to append/upsert
        try:
            qm.create_collections(dense_vector_size=768, recreate=False)
        except Exception as e:
            if "already exists" in str(e) or "Conflict" in str(e):
                logger.info("Collections already exist, proceeding with upsert.")
            else:
                logger.warning(f"Error creating collections (might be harmless if existing): {e}")
        
        logger.info(f"Processing: {filename}")
        
        # Parse
        print(f"DEBUG: Starting parsing...")
        metadata, sections = parser.parse_document(file_path)
        logger.info(f"Parsed {len(sections)} sections. Drug: {metadata.drug_name}")
        print(f"DEBUG: Parsing done. Sections: {len(sections)}")
        
        # Chunk
        print(f"DEBUG: Starting chunking...")
        parents, children = chunker.chunk_document(metadata, sections)
        logger.info(f"Generated {len(parents)} parents, {len(children)} children")
        print(f"DEBUG: Chunking done. Children count: {len(children)}")
        
        # Normalize (Enrich with RxCUI)
        print(f"DEBUG: Starting normalization...")
        if metadata.drug_name:
            drug_info = normalizer.normalize_drug_name(metadata.drug_name)
            if drug_info and 'rxcui' in drug_info:
                metadata.rxcui = drug_info['rxcui']
                logger.info(f"Normalized to RxCUI: {metadata.rxcui}")
                for p in parents: p.rxcui = metadata.rxcui
                for c in children: c.rxcui = metadata.rxcui
            else:
                logger.warning(f"Could not normalize drug name: {metadata.drug_name}")
        print(f"DEBUG: Normalization done.")
        
        # Embed
        if children:
            child_texts = [child.sentence_text for child in children]
            logger.info("Generating embeddings...")
            
            # Debugging embedding specifically
            print(f"DEBUG: Starting DENSE embedding for {len(child_texts)} texts...")
            child_embeddings = embedder.embed(child_texts)
            print(f"DEBUG: Dense embedding done. Shape: {child_embeddings.shape}")
            
            print(f"DEBUG: Starting SPARSE embedding...")
            sparse_vectors = sparse_embedder.embed_documents(child_texts)
            print(f"DEBUG: Sparse embedding done.")
            
            # Upsert
            print(f"DEBUG: Starting UPSERT...")
            logger.info("Upserting to Qdrant...")
            children_dicts = [c.to_dict() for c in children]
            qm.upsert_children(
                children=children_dicts,
                dense_embeddings=child_embeddings,
                sparse_embeddings=sparse_vectors
            )
            print(f"DEBUG: Upsert done.")
            
        if parents:
            print(f"DEBUG: Upserting parents...")
            parents_dicts = [p.to_dict() for p in parents]
            qm.upsert_parents(parents=parents_dicts)
            print(f"DEBUG: Parent upsert done.")
            
        logger.info("Ingestion successful!")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # The specific file causing issues
    target_file = "9C4F12E5-5E69-44D4-81A5-BD72C375EEF5 (1).xml"
    ingest_file(target_file)
