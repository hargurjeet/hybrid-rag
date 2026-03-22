import json, os
import cohere
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import ollama

load_dotenv()

CO_API_KEY = os.getenv("CO_API_KEY")
if not CO_API_KEY:
    raise ValueError("CO_API_KEY is not set in environment variables")

co = cohere.Client(CO_API_KEY)

# ============================================
# NEW: LLM Configuration
# ============================================
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"
USE_HF_INFERENCE = os.getenv("USE_HF_INFERENCE", "false").lower() == "true"


# Groq Setup (RECOMMENDED for cloud deployment)
if USE_GROQ:
    from groq import Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY required when USE_GROQ=true. Get free key from https://console.groq.com")
    
    groq_client = Groq(api_key=GROQ_API_KEY)
    # Available models: llama-3.1-70b-versatile, mixtral-8x7b-32768, gemma-7b-it
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    print(f"Using Groq API with model: {GROQ_MODEL}")

# Only import HF client if needed
elif USE_HF_INFERENCE:
    from huggingface_hub import InferenceClient
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN required when USE_HF_INFERENCE=true")
    
    # Model options (choose one):
    # Option 1: Mistral (good quality, may have rate limits)
    # Option 2: Zephyr (smaller, faster, more reliable on free tier)
    # Option 3: Llama-2 (good quality, stable)
    HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
    
    hf_client = InferenceClient(
        model=HF_MODEL,
        token=HF_TOKEN
    )
    print(f"Using HuggingFace Inference API with model: {HF_MODEL}")
else:

    print("Using Ollama (local)")

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

# ============================================
# MODIFIED: LLM Answer Generation with Toggle
# ============================================

def generate_answer_with_llama(query, reranked_docs, model_name="llama3.2"):
    """
    Generate final answer using one of three backends:
    1. Groq API (USE_GROQ=true) - RECOMMENDED for cloud, free & fast
    2. HuggingFace Inference API (USE_HF_INFERENCE=true) - limited free tier
    3. Ollama (default) - for local development
    
    Configure via environment variables:
    - USE_GROQ=true + GROQ_API_KEY → Groq
    - USE_HF_INFERENCE=true + HF_TOKEN → HuggingFace
    - Both false → Ollama (local)
    """

    context_blocks = []

    for i, doc in enumerate(reranked_docs):
        context_blocks.append(
            f"[Document {i+1} | arXiv:{doc['paper_id']}]\n{doc['text']}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
    You are a research assistant.

    Answer the question using ONLY the context documents.

    When using information from a document,
    cite it using [Document X].

    Example citation:
    The transformer model uses self-attention [Document 2].

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # Choose LLM backend based on configuration
    if USE_GROQ:
        # Groq API - RECOMMENDED for cloud deployment
        # Free, fast (fastest inference in the world), reliable
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=512,
                top_p=0.95
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {e}")
            return f"Error generating answer with Groq: {str(e)}"
        
    # Choose LLM based on configuration
    elif USE_HF_INFERENCE:
        # HuggingFace Inference API
        try:
            # Use chat completion for instruction-tuned models
            response = hf_client.chat_completion(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.7,
                top_p=0.95
            )
            # Extract the message content
            answer = response.choices[0].message.content
            return answer
        except Exception as e:
            print(f"HF Inference error: {e}")
            return f"Error generating answer: {str(e)}"
    else:
        # Ollama (local)
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

    answer = response["message"]["content"]
    return answer

def rerank_local(query, retrieved_docs, reranker_model, top_k=5):

    pairs = [(query, doc["text"]) for doc in retrieved_docs]

    scores = reranker_model.predict(pairs)

    for doc, score in zip(retrieved_docs, scores):
        doc["score"] = float(score)

    ranked_docs = sorted(
        retrieved_docs,
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked_docs[:top_k]