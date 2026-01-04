import os
import sys
sys.path.insert(0, 'C:/G/Maclens chatbot w api/backend')

os.chdir('C:/G/Maclens chatbot w api/backend')

from app.core.config import settings

print("Checking settings...")
print(f"API KEY: {settings.COHERE_API_KEY}")
print(f"CHROMA_DB_DIR: {settings.CHROMA_DB_DIR}")
print(f"DOCUMENTS_DIR: {settings.DOCUMENTS_DIR}")
