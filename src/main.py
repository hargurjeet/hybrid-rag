"""
ArXiv Paper hybrid RAG System

This module implements a Retrieval-Augmented Generation (RAG) system for ArXiv papers.
It uses ChromaDB for vector storage, SentenceTransformers for embeddings, Cohere for reranking,
and Llama for answer generation.

Dependencies:
    - chromadb: Vector database for storing document embeddings
    - sentence-transformers: For generating embeddings
    - cohere: For reranking retrieved documents
    - langfuse: For observability and tracing
"""

import os, sys

# Add parent directory to path to enable imports from src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required libraries
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from src.utils import (
    load_arxiv_documents, 
    chunk_documents, 
    upload_chunks_chroma, 
    rerank_with_cohere, 
    generate_answer_with_llama, 
    retrieve_chroma
)

# Import Langfuse for observability and tracing
from langfuse import observe
from langfuse import get_client

# Initialize Langfuse client for tracing
langfuse_client = get_client()

# Load environment variables from .env file
load_dotenv()

# Configuration
file_path = "dataset/arxiv_10k.json"
COLLECTION_NAME = "ArxivPapers"

# Initialize embedding model for generating vector representations
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Set up base directory for persistent storage of Chroma DB
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Set up base directory for persistent storage
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
persist_dir = os.path.join(BASE_DIR, "chroma_db")

# Initialize persistent ChromaDB client
# This creates or connects to a local database that persists across sessions
client = chromadb.PersistentClient(
    path=persist_dir,
    settings=Settings(
        anonymized_telemetry=False  # Optional
    )
)

# Get or create the collection for storing ArXiv papers
collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

# Print debug information about the collection
print("Persist dir:", persist_dir)
print("Collection count:", collection.count())

# -------------------
# INGESTION
# -------------------
if collection.count() == 0:
    print("Collection empty. Ingesting data...")

    docs = load_arxiv_documents(file_path)
    chunks = chunk_documents(docs)

    upload_chunks_chroma(chunks, collection, model)

    print("Ingestion complete.")
else:
    print("Collection already populated. Skipping ingestion.")


# -------------------
# RETRIEVAL
# -------------------

@observe() # Wraps the entire loop iteration as one trace
def run_query(query, collection, model):

    """
    Execute a complete RAG query: retrieve, rerank, and generate answer.
    
    Steps:
    1. Retrieve top-k relevant documents from ChromaDB
    2. Rerank documents using Cohere for better relevance
    3. Generate answer using Llama with reranked documents as context
    """

    # 1. Retrieval (Will show as a nested span)
    # retrieved_docs = hybrid_retrieve_documents(query, collection, model, top_k=20)
    retrieved_docs = retrieve_chroma(query, collection, model, top_k=20)
    
    # 2. Reranking (Will show as a nested span)
    # Note: If using Cohere, add @observe to rerank_with_cohere in utils.py
    reranked_docs = rerank_with_cohere(query, retrieved_docs, top_k=5)

    # 3. Generation (Will show as a 'Generation' span with token usage)
    answer = generate_answer_with_llama(query, reranked_docs)
    
    return answer, reranked_docs

# Interactive query loop
while True:
    query = input("\nAsk something (or type exit): ")

    if query.lower() == "exit":
        break
    
    answer, docs = run_query(query, collection, model)
    print(answer)

# Ensure all traces are sent to Langfuse before exiting
langfuse_client.flush() 