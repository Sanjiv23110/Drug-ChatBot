from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "text-embedding-ada-002"
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = "gpt-35-turbo"
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    
    # Database and Documents
    CHROMA_DB_DIR: str = "chroma_db"
    DOCUMENTS_DIR: str = r"C:\G\chatbot maclens\data"
    
    # Optional monitoring (Sentry)
    SENTRY_DSN: str | None = None
    ENVIRONMENT: str = "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
