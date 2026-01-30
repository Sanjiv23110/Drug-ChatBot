"""
Minimal diagnostic test for ingestion pipeline.
Tests each step individually with detailed logging.
"""
import asyncio
import logging
from pathlib import Path

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

async def test_ingestion_step_by_step():
    """Test each ingestion step individually."""
    
    # Step 1: Import test
    logger.info("=" * 60)
    logger.info("STEP 1: Testing imports...")
    try:
        from app.db.session import get_session, init_db, check_connection
        from app.ingestion.docling_utils import DoclingParser
        from app.ingestion.ingest import DrugMetadataExtractor, HeaderBasedChunker
        from app.db.models import get_or_create_section_mapping, MonographSection
        logger.info("✅ All imports successful")
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return
    
    # Step 2: Database connection test
    logger.info("=" * 60)
    logger.info("STEP 2: Testing database connection...")
    try:
        is_connected = await check_connection()
        if is_connected:
            logger.info("✅ Database connection OK")
        else:
            logger.error("❌ Database connection failed")
            return
    except Exception as e:
        logger.error(f"❌ Connection check error: {e}")
        return
    
    # Step 3: PDF parsing test
    logger.info("=" * 60)
    logger.info("STEP 3: Testing PDF parsing (this takes ~60s)...")
    try:
        parser = DoclingParser(extract_images=False)
        pdf_path = "data/pdfs/00062205.pdf"
        
        if not Path(pdf_path).exists():
            logger.error(f"❌ PDF not found: {pdf_path}")
            return
        
        logger.info(f"Parsing {pdf_path}...")
        parsed = parser.parse(pdf_path)
        
        if parsed.parse_success:
            logger.info(f"✅ PDF parsed successfully")
            logger.info(f"   - Pages: {parsed.page_count}")
            logger.info(f"   - Content length: {len(parsed.markdown_content)} chars")
            logger.info(f"   - Hash: {parsed.document_hash[:16]}...")
        else:
            logger.error(f"❌ PDF parsing failed: {parsed.error_message}")
            return
    except Exception as e:
        logger.error(f"❌ Parsing error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Metadata extraction test
    logger.info("=" * 60)
    logger.info("STEP 4: Testing metadata extraction...")
    try:
        extractor = DrugMetadataExtractor()
        drug_name, brand_name, generic_name = extractor.extract(parsed.markdown_content)
        logger.info(f"✅ Metadata extracted")
        logger.info(f"   - Drug: {drug_name}")
        logger.info(f"   - Brand: {brand_name}")
        logger.info(f"   - Generic: {generic_name}")
    except Exception as e:
        logger.error(f"❌ Metadata extraction error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Chunking test
    logger.info("=" * 60)
    logger.info("STEP 5: Testing header-based chunking...")
    try:
        chunker = HeaderBasedChunker()
        sections = chunker.chunk(parsed.markdown_content)
        logger.info(f"✅ Content chunked into {len(sections)} sections")
        if sections:
            logger.info(f"   - First section: {sections[0].header}")
            logger.info(f"   - First section length: {len(sections[0].content)} chars")
    except Exception as e:
        logger.error(f"❌ Chunking error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Section mapping test (THIS IS CRITICAL)
    logger.info("=" * 60)
    logger.info("STEP 6: Testing section mapping creation...")
    try:
        async with get_session() as session:
            logger.info("Session created, testing mapping...")
            
            # Test with first section
            if sections:
                test_header = sections[0].header_cleaned
                logger.info(f"Creating mapping for: {test_header}")
                
                mapping = await get_or_create_section_mapping(
                    session, test_header, "test"
                )
                logger.info(f"✅ Mapping created: {mapping.normalized_name}")
                
                # CRITICAL: This commit is normally done in _store_sections
                await session.commit()
                logger.info("✅ Mapping committed to database")
    except Exception as e:
        logger.error(f"❌ Section mapping error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 7: Section storage test (THIS IS WHERE IT MIGHT HANG)
    logger.info("=" * 60)
    logger.info("STEP 7: Testing section storage...")
    try:
        async with get_session() as session:
            logger.info("Creating test section...")
            
            # Create ONE test section
            test_section = MonographSection(
                drug_name=drug_name,
                brand_name=brand_name,
                generic_name=generic_name,
                original_filename=parsed.file_name,
                document_hash=parsed.document_hash,
                section_name=sections[0].header_cleaned,
                original_header=sections[0].header,
                content_text=sections[0].content[:500],  # Truncate for test
                content_markdown=sections[0].content[:500],
                image_paths=[],
                has_chemical_structure=False,
                page_start=None,
                page_end=None,
                char_count=len(sections[0].content)
            )
            
            logger.info("Adding section to session...")
            session.add(test_section)
            
            logger.info("Committing section...")
            await session.commit()
            
            logger.info("✅ Test section stored successfully!")
    except Exception as e:
        logger.error(f"❌ Section storage error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 8: Verification
    logger.info("=" * 60)
    logger.info("STEP 8: Verifying data in database...")
    try:
        from sqlalchemy import select, func
        
        async with get_session() as session:
            # Count sections
            result = await session.execute(
                select(func.count()).select_from(MonographSection)
            )
            section_count = result.scalar()
            logger.info(f"✅ Sections in database: {section_count}")
            
            # Count mappings
            from app.db.models import SectionMapping
            result = await session.execute(
                select(func.count()).select_from(SectionMapping)
            )
            mapping_count = result.scalar()
            logger.info(f"✅ Mappings in database: {mapping_count}")
    except Exception as e:
        logger.error(f"❌ Verification error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("=" * 60)
    logger.info("🎉 ALL TESTS PASSED!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_ingestion_step_by_step())
