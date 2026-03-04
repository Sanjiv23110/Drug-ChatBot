"""
Debug retrieval for "overdosage of Dantrium"
Simulate full retrieval pipeline to see what chunks are returned.
"""
import sys
import logging
sys.path.insert(0, 'c:/G/solomindUS')

from orchestrator.qa_orchestrator import RegulatoryQAOrchestrator, SectionClassifier
from retrieval.hybrid_retriever import HybridRetriever
from vector_db.hierarchical_qdrant import HierarchicalQdrantManager
from normalization.drug_normalizer import DrugNormalizer
from generation.constrained_extractor import RegulatoryQAGenerator
from generation.extractive_system import PostGenerationValidator
import openai
import os
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_retrieval():
    load_dotenv()
    
    # Initialize components (mocking generator/validator where possible to save API calls)
    qdrant = HierarchicalQdrantManager()
    retriever = HybridRetriever(qdrant)
    normalizer = DrugNormalizer()
    
    # We only care about retrieval, so we can mock generator/validator or just initialize them
    # Initialize real generator to see if extraction fails there
    client = openai.AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    # Mock validator since we removed it in Task 32, but Orchestrator might still use it
    validator = PostGenerationValidator(client) 
    generator = RegulatoryQAGenerator(client)

    orchestrator = RegulatoryQAOrchestrator(normalizer, retriever, generator, validator)
    
    query = "how to treat overdosage of dantrium?"
    print(f"\n--- DEBUGGING QUERY: '{query}' ---\n")
    
    # 1. Classification
    classifier = SectionClassifier()
    loinc = classifier.classify(query)
    print(f"1. Classification Result: {loinc}")
    if loinc != "34088-5":
        print("   [ERROR] Wrong classification! Expected 34088-5 (OVERDOSAGE)")
    else:
        print("   [OK] Correctly classified as OVERDOSAGE")

    # 2. Entity Normalization
    drug_name = "dantrium" # Mock extraction for now, assuming validator passes
    print(f"2. Assessing Drug: {drug_name}")
    
    # 3. Retrieval Simulation
    # Apply filter for drug and section
    filters = {
        "drug_name": drug_name,
        "loinc_code": loinc # 34088-5
    }
    
    print(f"3. Searching Qdrant with filters: {filters}")
    chunks = retriever.retrieve(query, filters=filters, limit=10)
    
    print(f"\n4. Retrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"   [{i}] Score: {chunk.get('score', 'N/A')} | Section: {chunk['metadata'].get('loinc_section')} | ParentID: {chunk['metadata'].get('parent_id')}")
        print(f"       Text: {chunk['raw_text'][:100]}...")
        
    if not chunks:
        print("\n[CRITICAL FAILURE] No chunks retrieved from Qdrant for this Section+Drug combination.")
        print("Possible causes:")
        print(" - Data not ingested for Dantrium Overdosage.")
        print(" - 'loinc_code' in Qdrant metadata does not match '34088-5'.")
        print(" - 'drug_name' in Qdrant metadata does not match 'dantrium' (case sensitivity?).")

    # 4. Context Expansion Check
    print("\n5. Checking Context Expansion...")
    expanded_chunks = orchestrator._apply_context_expansion(chunks, drug_name, loinc)
    print(f"   Chunks after expansion: {len(expanded_chunks)}")
    if len(expanded_chunks) > len(chunks):
         print("   [SUCCESS] Context expansion triggered!")
         extra = expanded_chunks[-1]
         print(f"   Expanded Chunk Text: {extra['raw_text'][:100]}...")
    else:
         print("   [WARNING] Context expansion did not add chunks.")

if __name__ == "__main__":
    debug_retrieval()
