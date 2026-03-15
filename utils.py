import json, os
import cohere
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

co = cohere.Client(COHERE_API_KEY)

def load_arxiv_documents(file_path):
    """Parse JSONL arXiv dataset and convert into LangChain Documents"""

    documents = []

    with open(file_path, "r") as f:
        for line in f:
            record = json.loads(line)

            # Combine title + abstract as content
            content = f"{record['title']}\n\n{record['abstract']}"

            metadata = {
                "paper_id": record["id"],
                "authors": record["authors"],
                "categories": record["categories"],
                "update_date": record["update_date"]
            }

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )

    return documents


def chunk_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    return chunks

def upload_chunks(chunks, collection, model):

    with collection.batch.dynamic() as batch:

        for chunk in tqdm(chunks):

            vector = model.encode(chunk.page_content).tolist()

            batch.add_object(
                properties={
                    "paper_id": chunk.metadata["paper_id"],
                    "categories": chunk.metadata["categories"],
                    "chunk_text": chunk.page_content
                },
                vector=vector
            )

    print("Upload complete")

def embed_query(query: str, model: SentenceTransformer):
    """
    Convert query text into embedding vector
    """
    return model.encode(query).tolist()

def retrieve_documents(query, collection, model, top_k=5):
    """
    Retrieve similar documents from Weaviate using vector search
    """

    query_vector = embed_query(query, model)

    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        return_properties=["paper_id", "title", "categories", "chunk_text"],
        return_metadata=["distance"]
    )

    results = []

    for obj in response.objects:
        distance = obj.metadata.distance
        score = 1 - distance if distance is not None else None

        results.append(
            {
                "paper_id": obj.properties["paper_id"],
                "title": obj.properties["title"],
                "categories": obj.properties["categories"],
                "text": obj.properties["chunk_text"],
                "score": score,
                "distance": distance
            }
        )

    return results

def hybrid_retrieve_documents(query, collection, model, top_k=5, alpha=0.5):
    """
    Hybrid retrieval using BM25 + Vector similarity
    """

    query_vector = model.encode(query).tolist()

    response = collection.query.hybrid(
        query=query,
        vector=query_vector,
        alpha=alpha,
        limit=top_k,
        return_properties=["paper_id", "title", "categories", "chunk_text"],
        return_metadata=["score"]
    )

    results = []

    for obj in response.objects:
        results.append(
            {
                "paper_id": obj.properties["paper_id"],
                "title": obj.properties["title"],
                "categories": obj.properties["categories"],
                "text": obj.properties["chunk_text"],
                "score": obj.metadata.score
            }
        )

    return results


def rerank_with_cohere(query, retrieved_docs, top_k=5):
    """
    Rerank retrieved documents using Cohere's rerank model.

    Parameters
    ----------
    query : str
        User search query
    retrieved_docs : list
        Documents returned from retriever
    top_k : int
        Number of final results to return

    Returns
    -------
    list
        Reranked documents
    """

    if not retrieved_docs:
        return []

    documents = [doc["text"] for doc in retrieved_docs]

    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=top_k
    )

    reranked_results = []

    for result in response.results:
        original_doc = retrieved_docs[result.index]

        reranked_results.append({
            "paper_id": original_doc["paper_id"],
            "title": original_doc["title"],
            "categories": original_doc["categories"],
            "text": original_doc["text"],
            "score": result.relevance_score
        })

    return reranked_results