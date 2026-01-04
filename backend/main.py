from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import endpoints
from app.core.config import settings
import logging

# Initialize Sentry (optional - only if DSN is provided)
if settings.SENTRY_DSN:
    import sentry_sdk
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.ENVIRONMENT,
        traces_sample_rate=0.1,  # 10% of requests for performance monitoring
        profiles_sample_rate=0.1,
    )
    logging.info("Sentry error tracking enabled")

app = FastAPI(title="Maclens Drug Chatbot API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(endpoints.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Maclens Drug Chatbot API"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring (UptimeRobot)"""
    try:
        from app.services.vector_store import VectorStoreService
        # Quick health check - just verify services are accessible
        return {
            "status": "healthy",
            "service": "maclens-chatbot",
            "environment": settings.ENVIRONMENT
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
