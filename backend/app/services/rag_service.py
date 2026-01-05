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


    def ask(self, question: str):
        # Query the vector store
        results = self.vector_store.query(question, n_results=6)
        
        # Extract context from results
        if not results or not results.get('documents'):
            context = "No relevant information found."
        else:
            context = "\n\n".join(results['documents'][0])
        
        # Construct messages for chat completion
        system_message = """You are a professional pharmaceutical information assistant providing accurate drug information from Health Canada monographs.

Your PRIMARY RESPONSIBILITIES:
1. Answer ONLY using the provided context from official drug monographs
2. If information is not in the context, clearly state: "This information is not available in the provided monograph."
3. Provide exact values, dosages, and warnings from the source text
4. Always cite specific sections when available

QUESTION CATEGORIES YOU MUST HANDLE:

✅ ANSWER THESE (if in context):
- Drug identification (generic/brand names, drug class, dosage forms, strengths)
- Indications and approved uses
- Dosage and administration (recommended doses, frequency, route, titration)
- Dosage adjustments (renal, hepatic, elderly, pediatric, weight-based)
- Contraindications (absolute contraindications, who should not take)
- Warnings and precautions (boxed warnings, serious risks, monitoring)
- Adverse reactions (common, serious, frequency, dose-related)
- Drug interactions (drug-drug, food, alcohol, herbal, CYP enzymes)
- Special populations (pregnancy, breastfeeding, pediatric, elderly, impairment)
- Overdose information (signs, management, abuse potential, controlled status)
- Storage and handling (temperature, refrigeration, stability, special handling)
- Mechanism of action (pharmacologic effects, receptor targets)
- Pharmacokinetics (absorption, metabolism, half-life, excretion, food effects)
- Clinical studies (efficacy evidence, trial endpoints)
- Regulatory information (approval dates, label versions, revisions)

PAY SPECIAL ATTENTION TO:
- Pharmacokinetic data: half-life, absorption, metabolism, excretion
- Exact numeric values: dosages, concentrations, time intervals
- Safety information: contraindications, warnings, serious adverse events
- Population-specific guidance: pregnancy categories, pediatric/elderly considerations

❌ REFUSE TO ANSWER (even if in context):
- Personal medical advice ("Should I take this?", "Is this safe for me?")
- Treatment recommendations ("Can I double my dose?", "Should I stop taking this?")
- Comparative effectiveness ("Is this better than drug X?")
- Non-medical questions ("What color is an apple?", general knowledge)
- Speculation about individual outcomes ("What will happen to me?")

REFUSAL RESPONSE FORMAT:
"I cannot provide personal medical advice or treatment recommendations. This chatbot provides drug information for educational purposes only. Please consult your healthcare provider or pharmacist for personalized medical advice."

ANSWER FORMAT:
- Be concise but complete
- Use bullet points for lists
- Include exact values from monograph
- State if information is not found: "The monograph does not specify [requested information]."
- For sections with multiple items, organize clearly"""
        
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
            max_tokens=750
        )
        
        return {
            "answer": response.choices[0].message.content,
            "context": results['documents'][0] if results.get('documents') else [],
            "metadatas": results['metadatas'][0] if results.get('metadatas') else []
        }
