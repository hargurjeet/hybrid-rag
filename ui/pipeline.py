import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer

from src.utils import (
    hybrid_retrieve_documents,
    retrieve_documents,
    rerank_with_cohere,
    generate_answer_with_llama
)

from config import COLLECTION_NAME, WEAVIATE_URL, WEAVIATE_API_KEY

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def get_weaviate_client():
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )


def run_pipeline(query, top_k=5, alpha=0.5, use_hybrid=True):

    with get_weaviate_client() as client:

        collection = client.collections.get(COLLECTION_NAME)

        # Retrieval
        if use_hybrid:
            retrieved_docs = hybrid_retrieve_documents(
                query=query,
                collection=collection,
                model=model,
                top_k=top_k * 4,
                alpha=alpha
            )
        else:
            retrieved_docs = retrieve_documents(
                query=query,
                collection=collection,
                model=model,
                top_k=top_k * 4
            )

        # Rerank
        reranked_docs = rerank_with_cohere(
            query,
            retrieved_docs,
            top_k=top_k
        )

        # Answer
        answer = generate_answer_with_llama(
            query=query,
            reranked_docs=reranked_docs
        )

        return answer, reranked_docs