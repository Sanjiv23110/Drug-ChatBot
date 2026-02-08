
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

def ingest_specific_files(filenames):
    DATA_DIR = os.path.join(os.getcwd(), "data", "xml")
    XSLT_PATH = os.path.join(os.getcwd(), "data", "spl_stylesheet.xsl")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    
    # Initialize components ONCE
    try:
        logger.info("Initializing components...")
        parser = SPLXMLParser(xsl_path=XSLT_PATH if os.path.exists(XSLT_PATH) else None)
        chunker = HierarchicalChunker()
        normalizer = DrugNormalizer()
        
        logger.info("Loading embedding model (this may take a moment)...")
        embedder = DenseEmbedder("pritamdeka/S-PubMedBert-MS-MARCO")
        sparse_embedder = SparseEmbedder(corpus=None)
        
        qm = HierarchicalQdrantManager(host=QDRANT_HOST, port=QDRANT_PORT)
        # Ensure collections exist (recreate=False)
        try:
            qm.create_collections(dense_vector_size=768, recreate=False)
        except Exception as e:
            pass # Ignore if exists

        for filename in filenames:
            file_path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                continue

            logger.info(f"Processing: {filename}")
            
            try:
                # Parse
                metadata, sections = parser.parse_document(file_path)
                logger.info(f"Parsed {len(sections)} sections. Drug: {metadata.drug_name}")
                
                # Chunk
                parents, children = chunker.chunk_document(metadata, sections)
                logger.info(f"Generated {len(parents)} parents, {len(children)} children")
                
                # Normalize
                if metadata.drug_name:
                    drug_info = normalizer.normalize_drug_name(metadata.drug_name)
                    if drug_info and 'rxcui' in drug_info:
                        metadata.rxcui = drug_info['rxcui']
                        logger.info(f"Normalized to RxCUI: {metadata.rxcui}")
                        for p in parents: p.rxcui = metadata.rxcui
                        for c in children: c.rxcui = metadata.rxcui
                
                # Embed
                if children:
                    child_texts = [child.sentence_text for child in children]
                    logger.info(f"Generating embeddings for {len(child_texts)} chunks...")
                    child_embeddings = embedder.embed(child_texts)
                    sparse_vectors = sparse_embedder.embed_documents(child_texts)
                    
                    # Upsert
                    logger.info("Upserting to Qdrant...")
                    children_dicts = [c.to_dict() for c in children]
                    qm.upsert_children(
                        children=children_dicts,
                        dense_embeddings=child_embeddings,
                        sparse_embeddings=sparse_vectors
                    )
                    
                if parents:
                    parents_dicts = [p.to_dict() for p in parents]
                    qm.upsert_parents(parents=parents_dicts)
                    
                logger.info(f"Successfully ingested {filename}")
                
            except Exception as e:
                logger.error(f"Failed to ingest {filename}: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        logger.error(f"Initialization failed: {e}")

if __name__ == "__main__":
    files_to_ingest = [
        "drug(a) (1) (1).xml",
        "drug(b) (1).xml",
        "drug(c) (1).xml",
        "drug(d) (1).xml"
    ]
    ingest_specific_files(files_to_ingest)
