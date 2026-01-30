"""
Async PostgreSQL session management with connection pooling.

Uses asyncpg for high-performance async operations.
Configured for production use with proper pooling and error handling.
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy import text
from sqlmodel import SQLModel

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Database configuration from environment
DATABASE_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "database": os.getenv("POSTGRES_DB", "medical_rag"),
}

# Build connection URL
DATABASE_URL = (
    f"postgresql+asyncpg://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
    f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
)

# Sync URL for migrations (if needed)
SYNC_DATABASE_URL = (
    f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
    f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
)


def get_engine(
    pool_size: int = 10,
    max_overflow: int = 20,
    pool_timeout: int = 30,
    pool_recycle: int = 1800,
    echo: bool = False
) -> AsyncEngine:
    """
    Create async database engine with connection pooling.
    
    Args:
        pool_size: Number of connections to keep in pool
        max_overflow: Max additional connections beyond pool_size
        pool_timeout: Seconds to wait for available connection
        pool_recycle: Recycle connections after this many seconds
        echo: Log all SQL statements (for debugging)
        
    Returns:
        AsyncEngine instance configured for production
    """
    return create_async_engine(
        DATABASE_URL,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        echo=echo,
        # Performance settings
        pool_pre_ping=True,  # Verify connections before use
    )


# Global engine instance (lazy initialization)
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None


def get_global_engine() -> AsyncEngine:
    """Get or create the global engine instance."""
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def get_session_factory() -> async_sessionmaker:
    """Get or create the global session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_global_engine()
        _session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.
    
    Usage:
        async with get_session() as session:
            result = await session.execute(query)
            await session.commit()
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    except Exception as e:
        await session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        await session.close()


async def init_db():
    """
    Initialize database schema.
    
    Creates all tables defined in models.py.
    Also initializes pgvector and pg_trgm extensions.
    """
    engine = get_global_engine()
    
    async with engine.begin() as conn:
        # Create extensions (requires superuser or extension already installed)
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            logger.info("✓ pgvector extension enabled")
        except Exception as e:
            logger.warning(f"Could not create pgvector extension: {e}")
            
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
            logger.info("✓ pg_trgm extension enabled")
        except Exception as e:
            logger.warning(f"Could not create pg_trgm extension: {e}")
        
        # Create all tables
        await conn.run_sync(SQLModel.metadata.create_all)
        logger.info("✓ Database tables created")


async def close_db():
    """
    Close database connections gracefully.
    
    Should be called on application shutdown.
    """
    global _engine, _session_factory
    
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connections closed")


async def check_connection() -> bool:
    """
    Verify database connection is working.
    
    Returns:
        True if connection is healthy, False otherwise
    """
    try:
        async with get_session() as session:
            result = await session.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


# Health check info
async def get_db_status() -> dict:
    """
    Get database status for health checks.
    
    Returns:
        Dict with connection status and pool info
    """
    engine = get_global_engine()
    pool = engine.pool
    
    return {
        "connected": await check_connection(),
        "pool_size": pool.size() if hasattr(pool, 'size') else 'N/A',
        "checked_in": pool.checkedin() if hasattr(pool, 'checkedin') else 'N/A',
        "checked_out": pool.checkedout() if hasattr(pool, 'checkedout') else 'N/A',
        "database": DATABASE_CONFIG['database'],
        "host": DATABASE_CONFIG['host'],
    }
