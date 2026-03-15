import os
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from utils import load_arxiv_documents, chunk_documents, upload_chunks, retrieve_documents

load_dotenv()

file_path = "dataset/arxiv_10k.json"
COLLECTION_NAME = "ArxivPapers"

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Best practice: store your credentials in environment variables
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

with weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
) as client:

    print("Connection status:", client.is_ready())

    collections = client.collections.list_all()
    # -------------------
    # INGESTION
    # -------------------

    if COLLECTION_NAME not in collections:

        print("Collection not found. Creating collection and ingesting data...")

        client.collections.create(
            name=COLLECTION_NAME,
            properties=[
                Property(name="paper_id", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="categories", data_type=DataType.TEXT),
                Property(name="chunk_text", data_type=DataType.TEXT),
            ],
        )

        collection = client.collections.get(COLLECTION_NAME)

        docs = load_arxiv_documents(file_path)
        print(f"Loaded documents: {len(docs)}")

        chunks = chunk_documents(docs)
        print(f"Chunked documents: {len(chunks)}")

        upload_chunks(chunks, collection, model)

        print("Ingestion complete.")

    else:
        print(f"Collection '{COLLECTION_NAME}' already exists. Skipping ingestion.")


    # Always get collection
    collection = client.collections.get(COLLECTION_NAME)

    # -------------------
    # RETRIEVAL
    # -------------------
    while True:
        query = input("\nAsk something (or type exit): ")

        if query == "exit":
            break

        results = retrieve_documents(query, collection, model, top_k=10)

        print("\nTop retrieved documents:\n")

        for i, doc in enumerate(results):

            print(f"Result {i+1}")
            print("Paper ID:", doc["paper_id"])
            print("Category:", doc["categories"])
            print("Similarity Score:", doc["score"])
            print("Text snippet:", doc["text"][:300])
            print("-" * 50)