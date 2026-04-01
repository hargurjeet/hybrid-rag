import os, sys
# Add parent directory to path for importing src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Import utility functions from src
from src.utils import (
    retrieve_chroma,           # Vector retrieval from ChromaDB
    retrieve_documents,        # Legacy retrieval function
    rerank_with_cohere,        # Reranking using Cohere API
    generate_answer_with_llama # Answer generation using Llama
)

from config import COLLECTION_NAME

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# ChromaDB persistence directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

def get_chroma_client():
    """
    Initialize and return a persistent ChromaDB client.
    
    Returns:
        chromadb.PersistentClient: ChromaDB client instance
    """
    return chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(
            anonymized_telemetry=False  # Disable telemetry for privacy
        )
    )



def run_pipeline(query, top_k=5, alpha=0.5, use_hybrid=True):

    # Initialize ChromaDB client and get collection
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    try:
        # Step 1: Retrieve relevant documents from ChromaDB
        # Currently only vector search is supported
        # TODO: Implement hybrid search when ChromaDB supports it or add separate BM25 index
        retrieved_docs = retrieve_chroma(
            query=query,
            collection=collection,
            model=model,
            top_k=top_k * 4  # Retrieve more docs before reranking
        )
        
        # Step 2: Rerank documents using Cohere API
        # This improves relevance by using a more sophisticated model
        reranked_docs = rerank_with_cohere(
            query,
            retrieved_docs,
            top_k=top_k
        )
        
        # Step 3: Generate answer using Llama
        # Uses reranked documents as context for better answers
        answer = generate_answer_with_llama(
            query=query,
            reranked_docs=reranked_docs
        )
        
        return answer, reranked_docs
        
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        raise