import sys
import os
import glob
import logging
import traceback
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("CleanIngest")

# Paths and Config
DATA_DIR = r"C:\G\solomindUS\data\xml"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333

# CRITICAL: Define correct collections
COLLECTIONS = ["spl_parents", "spl_children"]

def main():
    logger.info("=== STEP 1: CLEAR DATABASE ===")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        for col in COLLECTIONS:
            try:
                # Check if exists first
                if client.collection_exists(col):
                    info = client.get_collection(col)
                    count_before = info.points_count
                    client.delete_collection(col)
                    logger.info(f"✔ Deleted collection '{col}'. Records before: {count_before}")
                else:
                    logger.info(f"ℹ Collection '{col}' does not exist (skipping delete).")
            except Exception as e:
                logger.warning(f"⚠ Error clearing '{col}': {e}")
                
    except Exception as e:
        logger.critical(f"✖ FABRICATED FAILURE: Could not connect to Qdrant to clear DB: {e}")
        sys.exit(1)

    logger.info("\n=== STEP 2: PRE-INGESTION SANITY CHECK ===")
    xml_files = glob.glob(os.path.join(DATA_DIR, "*.xml"))
    if not xml_files:
        logger.critical(f"✖ FATAL: No XML files found in {DATA_DIR}")
        sys.exit(1)
        
    logger.info(f"✔ Found {len(xml_files)} XML documents to process.")
    for f in xml_files[:5]:
        logger.info(f"  - {os.path.basename(f)}")
        
    logger.info("\n=== STEP 3: RUN ROBUST INGESTION ===")
    logger.info("Importing ingestion module...")
    
    # Add scripts directory ensuring robust_ingestion is importable
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(scripts_dir)
    # Also add root for other imports inside robust_ingestion
    root_dir = os.path.dirname(scripts_dir)
    sys.path.append(root_dir)
    
    try:
        import robust_ingestion as ingestion_script
        
        # MONKEY PATCH: Force get_already_ingested_files to return empty set
        # This guarantees ingestion runs regardless of DB state
        def force_empty_checkpoint(qm):
            logger.info("⚡ FORCED EMPTY CHECKPOINT (Monkey Patch active)")
            return set()
            
        ingestion_script.get_already_ingested_files = force_empty_checkpoint
        
        logger.info("Executing main ingestion logic...")
        ingestion_script.main()
        
    except Exception as e:
        logger.critical(f"✖ Ingestion script crashed: {e}")
        traceback.print_exc()
        sys.exit(1)

    logger.info("\n=== STEP 4: POST-INGESTION VERIFICATION ===")
    try:
        # Verify counts
        for col in COLLECTIONS:
            try:
                if client.collection_exists(col):
                    info = client.get_collection(col)
                    logger.info(f"Collection '{col}': {info.points_count} records.")
                    
                    if info.points_count == 0:
                         logger.error(f"✖ FAILURE: Collection '{col}' is empty after ingestion!")
                    else:
                         logger.info(f"✔ Success: '{col}' has data.")
                else:
                    logger.error(f"✖ FAILURE: Collection '{col}' was NOT created!")

            except Exception as e:
                logger.error(f"Error checking '{col}': {e}")

        # Verify specific drug (Lisinopril example if present, or just first file)
        # Using a simple scroll to get one payload
        try:
            res = client.scroll(collection_name="spl_parents", limit=1, with_payload=True)
            if res and res[0]:
                payload = res[0][0].payload
                logger.info(f"\nSample Record Verification:")
                logger.info(f"  - Drug Name: {payload.get('drug_name')}")
                logger.info(f"  - Set ID: {payload.get('set_id')}")
                logger.info("✔ Data payload structure valid.")
            else:
                logger.warning("⚠ Could not retrieve sample record.")
        except Exception as e:
            logger.warning(f"⚠ Could not verify sample payload: {e}")
                
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
