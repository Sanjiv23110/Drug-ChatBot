import asyncio
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.db.session import get_global_engine
from app.db.models import * # Load models to metadata
from app.db.fact_span_model import * # Load models
from sqlmodel import SQLModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def reset_db():
    logger.info("Resetting Database (Dropping all tables)...")
    engine = get_global_engine()
    
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        
    logger.info("Database reset complete.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(reset_db())
