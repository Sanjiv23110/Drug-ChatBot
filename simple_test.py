import os
os.chdir('C:/G/Maclens chatbot w api/backend')
import sys
sys.path.insert(0, os.getcwd())

print("Loading...")
from dotenv import load_dotenv
load_dotenv()

import cohere
api_key = os.getenv('COHERE_API_KEY')
print(f"API Key loaded: {api_key[:10]}...")

print("Testing Cohere client...")
client = cohere.Client(api_key)

print("Testing embed with search_query...")
result = client.embed(
    texts=["test query"],
    model="embed-english-v3.0",
    input_type="search_query"
)
print(f"Success! Embedding dimension: {len(result.embeddings[0])}")

print("Testing chat...")
chat_result = client.chat(
    message="Say hello",
    model="command"
)
print(f"Chat Success! Response: {chat_result.text[:50]}...")
