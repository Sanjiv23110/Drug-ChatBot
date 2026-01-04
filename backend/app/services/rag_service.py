import cohere
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings
from app.services.vector_store import VectorStoreService
from openai import AzureOpenAI

class RagService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        
        # Initialize Azure OpenAI client for generation
        self.openai_client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful and expert assistant for drug monographs. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer concise and accurate.

Context:
{context}

Question: 
{question}

Answer:"""
        )

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs) # Chroma returns results differently, need to adjust

    def ask(self, question: str):
        # Query the vector store
        results = self.vector_store.query(question, n_results=3)
        
        # Extract context from results
        if not results or not results.get('documents'):
            context = "No relevant information found."
        else:
            context = "\n\n".join(results['documents'][0])
        
        # Construct messages for chat completion
        system_message = """You are a helpful assistant providing drug information from Health Canada monographs. 
Use only the provided context to answer questions. If the information is not in the context, say so clearly. 
Be accurate, concise, and professional."""
        
        user_message = f"""Context from drug monographs:
{context}

Question: {question}

Please provide an accurate answer based only on the context above."""
        
        # Generate answer using Azure OpenAI
        response = self.openai_client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return {
            "answer": response.choices[0].message.content,
            "context": results['documents'][0] if results.get('documents') else [],
            "metadatas": results['metadatas'][0] if results.get('metadatas') else []
        }
