"""
RAGas Evaluation Script for Medical RAG System

Evaluates the production RAG system using RAGas metrics:
- Faithfulness (no hallucinations)
- Answer Relevancy (answers the question)
- Context Precision (correct chunks ranked first)
- Context Recall (all relevant info retrieved)
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Import production RAG components
from app.vectorstore.faiss_store import FAISSVectorStore
from app.vectorstore.index_manager import IndexManager
from app.metadata.sqlite_store import SQLiteMetadataStore
from app.ingestion.embedder import AzureEmbedder
from app.retrieval.retriever import retrieve
from app.generation.answer_generator import AnswerGenerator

# Load environment variables
load_dotenv()

# Azure OpenAI configuration for RAGas
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-agent")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

def initialize_rag_system():
    """Initialize production RAG components."""
    print("Initializing RAG system...")
    
    # Load FAISS index
    faiss_store = FAISSVectorStore(dimension=1536)
    index_manager = IndexManager("data/faiss/medical_index")
    
    loaded = index_manager.load(dimension=1536)
    if loaded:
        faiss_store.index, faiss_store.chunk_ids, _ = loaded
        print(f"✓ Loaded FAISS index with {faiss_store.count()} vectors")
    else:
        raise Exception("Failed to load FAISS index")
    
    # Initialize other components
    metadata_store = SQLiteMetadataStore("data/metadata.db")
    embedder = AzureEmbedder()
    generator = AnswerGenerator()
    
    return faiss_store, metadata_store, embedder, generator

def run_rag_query(query, faiss_store, metadata_store, embedder, generator):
    """Run a single query through the production RAG pipeline."""
    # Retrieve chunks
    chunks = retrieve(
        query=query,
        faiss_store=faiss_store,
        metadata_store=metadata_store,
        embedder=embedder
    )
    
    # Generate answer
    result = generator.generate(query=query, context_chunks=chunks)
    
    # Extract contexts as text list
    contexts = [chunk.get('chunk_text', '') for chunk in chunks]
    
    return {
        'answer': result['answer'],
        'contexts': contexts
    }

def main():
    print("="*60)
    print("RAGas Evaluation for Medical RAG System")
    print("="*60)
    
    # Initialize RAG system
    faiss_store, metadata_store, embedder, generator = initialize_rag_system()
    
    # Load test set
    test_set_path = Path("evaluation/medical_test_set.csv")
    print(f"\nLoading test set from: {test_set_path}")
    
    df = pd.read_csv(test_set_path)
    print(f"✓ Loaded {len(df)} test questions")
    
    # Run questions through RAG pipeline
    print("\nRunning questions through RAG pipeline...")
    answers = []
    contexts_list = []
    
    for i, row in df.iterrows():
        question = row['question']
        print(f"  Processing {i+1}/{len(df)}: {question[:60]}...")
        
        result = run_rag_query(
            query=question,
            faiss_store=faiss_store,
            metadata_store=metadata_store,
            embedder=embedder,
            generator=generator
        )
        
        answers.append(result['answer'])
        contexts_list.append(result['contexts'])
    
    print(f"✓ Generated {len(answers)} answers")
    
    # Prepare dataset for RAGas
    evaluation_data = {
        'question': df['question'].tolist(),
        'answer': answers,
        'contexts': contexts_list,
        'ground_truth': df['ground_truth'].tolist()
    }
    
    dataset = Dataset.from_dict(evaluation_data)
    
    # Configure Azure OpenAI for RAGas metrics
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        temperature=0
    )
    
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY
    )
    
    # Run RAGas evaluation
    print("\nRunning RAGas evaluation...")
    print("  Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall")
    
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=llm,
        embeddings=embeddings
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    scores = result.to_pandas()
    
    print("\nMetric Scores:")
    print("-" * 60)
    for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
        if metric in scores.columns:
            mean_score = scores[metric].mean()
            print(f"  {metric.replace('_', ' ').title():.<40} {mean_score:.3f}")
    print("-" * 60)
    
    # Compare to targets
    print("\nTarget Comparison (Medical System):")
    print("-" * 60)
    targets = {
        'faithfulness': 0.90,
        'answer_relevancy': 0.85,
        'context_precision': 0.80,
        'context_recall': 0.75
    }
    
    for metric, target in targets.items():
        if metric in scores.columns:
            score = scores[metric].mean()
            status = "✓ PASS" if score >= target else "✗ FAIL"
            print(f"  {metric.replace('_', ' ').title():.<30} {score:.3f} (target: {target:.2f}) {status}")
    print("-" * 60)
    
    # Save detailed results
    output_path = Path("evaluation/rag_evaluation_results.json")
    result_dict = {
        'summary': {
            metric: float(scores[metric].mean()) 
            for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
            if metric in scores.columns
        },
        'targets': targets,
        'detailed_scores': scores.to_dict('records')
    }
    
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    print(f"\n✓ Saved detailed results to: {output_path}")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
