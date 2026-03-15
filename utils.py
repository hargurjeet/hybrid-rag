import json
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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

