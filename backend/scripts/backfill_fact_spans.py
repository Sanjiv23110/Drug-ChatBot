
import asyncio
import logging
import sys
import os

# Add backend to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.db.session import init_db, get_session
from app.db.models import MonographSection
from app.db.fact_span_model import FactSpan
from app.ingestion.factspan_extractor import FactSpanExtractor
from sqlmodel import select, func

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def backfill():
    """
    Backfill FactSpans from existing MonographSections.
    """
    logger.info("Starting FactSpan Backfill...")
    
    # 1. Initialize DB (create tables if missing)
    logger.info("Verifying schema...")
    await init_db()
    
    # Initialize extractor
    # Note: spacy load happens inside __init__
    extractor = FactSpanExtractor()
    
    async with get_session() as session:
        # 2. Get all sections
        logger.info("Fetching existing sections...")
        stmt = select(
            MonographSection.id,
            MonographSection.drug_name,
            MonographSection.brand_name,
            MonographSection.generic_name,
            MonographSection.section_name,
            MonographSection.original_header,
            MonographSection.content_text,
            MonographSection.document_hash,
            MonographSection.original_filename
        )
        result = await session.execute(stmt)
        sections = result.all()
        
        logger.info(f"Found {len(sections)} sections to check.")
        
        total_spans = 0
        processed_sections = 0
        skipped_sections = 0
        errors = 0
        
        for i, section in enumerate(sections):
            try:
                # Check if spans already exist for this section
                count_stmt = select(func.count(FactSpan.fact_span_id)).where(FactSpan.section_id == section.id)
                count_res = await session.execute(count_stmt)
                count = count_res.scalar()
                
                if count > 0:
                    skipped_sections += 1
                    if skipped_sections % 50 == 0:
                         logger.info(f"Skipped {skipped_sections} already processed sections...")
                    continue
                
                # Extract spans from verified content_text
                # VERBATIM extraction
                spans = extractor.extract(section.content_text, section_id=section.id)
                
                if not spans:
                    logger.warning(f"No spans extracted for section {section.id} ({section.section_name})")
                    continue
                
                # Convert to DB models
                db_spans = []
                for span in spans:
                    db_span = FactSpan(
                        drug_name=section.drug_name,
                        # brand_name/generic_name removed from required model fields?
                        # Check model definition. I kept ONLY what I defined in step 8130!
                        # In step 8130 I defined 'drug_name' but NOT brand/generic.
                        # So I must NOT pass brand/generic.
                        
                        section_id=section.id,
                        section_enum=section.section_name, # Mapped to section_enum
                        original_header=section.original_header,
                        
                        sentence_text=span.text,       # NEW NAME
                        source_type=span.text_type,    # NEW NAME
                        sentence_index=span.sequence_num, # NEW NAME
                        
                        document_hash=section.document_hash,
                        original_filename=section.original_filename,
                        
                        # Initialize required metadata with defaults
                        attribute_tags=[],
                        page_number=None # Extractor doesn't track page number yet
                    )
                    db_spans.append(db_span)
                
                # Bulk save
                session.add_all(db_spans)
                await session.commit()
                
                total_spans += len(db_spans)
                processed_sections += 1
                
                if processed_sections % 10 == 0:
                    logger.info(f"Processed {processed_sections} sections. Total spans added: {total_spans}")
                    
            except Exception as e:
                logger.error(f"Error processing section {section.id}: {e}")
                await session.rollback()
                errors += 1
        
        logger.info("="*50)
        logger.info("BACKFILL COMPLETE")
        logger.info("="*50)
        logger.info(f"Total Sections Checked: {len(sections)}")
        logger.info(f"Processed (New):      {processed_sections}")
        logger.info(f"Skipped (Existing):   {skipped_sections}")
        logger.info(f"Errors:               {errors}")
        logger.info(f"Total FactSpans:      {total_spans}")
        logger.info("="*50)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(backfill())
