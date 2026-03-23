import os

# Default config (can be overridden via sidebar)
DEFAULT_TOP_K = 5
DEFAULT_ALPHA = 0.5

COLLECTION_NAME = "ArxivPapers"

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")