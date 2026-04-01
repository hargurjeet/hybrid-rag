"""
RAG System Evaluation Module

This module evaluates the performance of the RAG (Retrieval-Augmented Generation) pipeline
using RAGAS metrics including faithfulness and answer relevancy.

The evaluation process:
1. Loads a test dataset of questions with ground truth answers
2. For each question, retrieves documents from ChromaDB
3. Reranks documents using a local reranker model
4. Generates answers using Llama
5. Evaluates answers using RAGAS metrics
6. Saves results and checks against thresholds
"""


import json
import numpy as np
import pandas as pd
import os, sys

# Add parent directory to path for importing src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Disable RAGAS telemetry
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

# Import vector database libraries
import chromadb
from chromadb.config import Settings

# Import ML models
from sentence_transformers import SentenceTransformer, CrossEncoder

# Import utility functions from src
from src.utils import retrieve_chroma, generate_answer_with_llama, rerank_local

# Import RAGAS evaluation libraries
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from datasets import Dataset

# Import LangChain components for LLM integration
from langchain_community.llms import Ollama
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


# ============================================
# CONFIGURATION
# ============================================
# Database configuration
COLLECTION_NAME = "ArxivPapers"

# Evaluation thresholds
FAITHFULNESS_THRESHOLD = 0.3      # Minimum faithfulness score required
RELEVANCY_THRESHOLD = 0.75          # Minimum answer relevancy score required

# LLM selection for evaluation
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"  # Use Groq cloud API if true

# Path configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
persist_dir = os.path.join(BASE_DIR, "chroma_db")

# RAGAS evaluation configuration
# Using sequential processing to avoid rate limits and ensure stability
evaluation_config = RunConfig(
    max_workers=1, 
    timeout=180
)

# ============================================
# DATABASE INITIALIZATION
# ============================================

# Initialize persistent ChromaDB client
client = chromadb.PersistentClient(
    path=persist_dir,
    settings=Settings(
        anonymized_telemetry=False  # Disable telemetry for privacy
    )
)

# Get the collection containing ArXiv papers
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# ============================================
# SMART LLM SELECTION FOR EVALUATION
# Automatically matches your main pipeline configuration
# ============================================
 
if USE_GROQ:
    # Cloud evaluation with Groq (more reliable, better structured output)
    print("🌩️  Using Groq for evaluation (cloud)")
    
    from langchain_groq import ChatGroq
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY required when USE_GROQ=true")
    
    # Use same model as main pipeline or specify evaluation model
    GROQ_EVAL_MODEL = os.getenv("GROQ_EVAL_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    
    judge_llm = ChatGroq(
        model=GROQ_EVAL_MODEL,
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        max_tokens=2048
    )
    
    print(f"   Model: {GROQ_EVAL_MODEL}")
else:
    # Local evaluation with Ollama
    print("🏠 Using Ollama for evaluation (local)")
    
    # Use different models based on what's available
    # Priority: llama3.2 > llama3.1 > phi3:mini
    OLLAMA_EVAL_MODEL = os.getenv("OLLAMA_EVAL_MODEL", "phi3:mini")
    
    judge_llm = Ollama(
        model=OLLAMA_EVAL_MODEL,
        temperature=0,
        timeout=300
    )
    
    print(f"   Model: {OLLAMA_EVAL_MODEL}")
    print(f"   💡 Tip: For better results, use llama3.2 or llama3.1:")
    print(f"      ollama pull llama3.2")
    print(f"      export OLLAMA_EVAL_MODEL=llama3.2")

ragas_llm = LangchainLLMWrapper(judge_llm)

# ============================================
# MODEL INITIALIZATION
# ============================================
# Initialize reranker model for improving retrieval quality
# BAAI/bge-reranker-large is a state-of-the-art reranking model
reranker_model = CrossEncoder("BAAI/bge-reranker-large")

# Initialize embedding model for RAGAS evaluation
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Initialize embedding model for document retrieval (same as main pipeline)
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# ============================================
# LOAD EVALUATION DATASET
# ============================================

# Load test questions and ground truth answers
# Dataset format: list of dictionaries with 'question' and 'ground_truth' fields
with open("evaluation/evaluation_qa_dataset_5.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

contexts_list = []
answers_list = []

# ============================================
# EVALUATION LOOP
# Process each test question through the RAG pipeline
# ============================================

for question in df["question"]:

    print("Evaluating:", question)

    # Step 1: Retrieve relevant documents from ChromaDB
    # Gets top 20 most similar documents based on embedding similarity
    retrieved_docs = retrieve_chroma(
        query=question,
        collection=collection,
        model=model,
        top_k=20
    )

    # Step 2: Rerank documents using local cross-encoder model
    # Improves relevance by using a more sophisticated reranking model
    # Returns top 5 most relevant documents
    reranked_docs = rerank_local(
                                    question,
                                    retrieved_docs,
                                    reranker_model,
                                    top_k=5
                                )
    
    # Step 3: Generate answer using Llama with reranked documents as context
    answer = generate_answer_with_llama(
        query=question,
        reranked_docs=reranked_docs
    )

    # Extract text from reranked documents for evaluation
    contexts = [doc.get("text", "") for doc in reranked_docs]
    
    # Store results
    contexts_list.append(contexts)
    answers_list.append(answer)


# Add results to the dataframe
df["contexts"] = contexts_list
df["answer"] = answers_list

# Convert to HuggingFace Dataset format for RAGAS
dataset = Dataset.from_pandas(df)

# ============================================
# RUN RAGAS EVALUATION
# ============================================

# Run evaluation
print("\n" + "="*60)
print("Starting RAGAS Evaluation...")
print("="*60 + "\n")

# Evaluate faithfulness metric
# Faithfulness measures whether the generated answer is factually consistent with the retrieved context
result = evaluate(
    dataset,
    metrics=[faithfulness],
    llm=ragas_llm,
    embeddings=embedding_model,
    run_config=evaluation_config # Force sequential processing
)

print(f'Result: {result}')

# Save evaluation results to CSV for analysis
result_df = result.to_pandas()
result_df.to_csv("rag_evaluation_results.csv", index=False)

# Extract faithfulness scores
faithfulness_score = result["faithfulness"]

# Calculate average faithfulness score (ignoring NaN values)
avg_score = np.nanmean(faithfulness_score)

print(f"Average Faithfulness Score: {avg_score}")

# ============================================
# EVALUATION THRESHOLD CHECK
# ============================================

if avg_score < FAITHFULNESS_THRESHOLD:
    print("❌ Faithfulness below threshold")
    print("❌ Evaluation failed")
    sys.exit(1)
else:
    print("✅ Faithfulness above threshold")
    print("✅ Evaluation passed")

