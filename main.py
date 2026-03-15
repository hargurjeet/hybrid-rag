import os
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from utils import load_arxiv_documents, chunk_documents, upload_chunks

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

    if COLLECTION_NAME in collections:
        print(f"Collection '{COLLECTION_NAME}' already exists. Skipping ingestion.")
        exit()

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

    # Only process dataset if collection was newly created
    docs = load_arxiv_documents(file_path)
    print(f"Loaded documents: {len(docs)}")

    chunks = chunk_documents(docs)
    print(f"Chunked documents: {len(chunks)}")

    upload_chunks(chunks, collection, model)

    print("Ingestion complete.")