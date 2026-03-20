
import json
import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

from src.utils import hybrid_retrieve_documents, generate_answer_with_llama, rerank_local

from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from datasets import Dataset

from langchain_community.llms import Ollama
from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()


weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
COLLECTION_NAME = "ArxivPapers"
FAITHFULNESS_THRESHOLD = 0.3
RELEVANCY_THRESHOLD = 0.75


# Setup local Mixtral judge model via Ollama
judge_llm = Ollama(
    model="phi3:mini",
    temperature=0,
    timeout=300 # Increase to 5 minutes
)

ragas_llm = LangchainLLMWrapper(judge_llm)
reranker_model = CrossEncoder("BAAI/bge-reranker-large")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Load evaluation dataset
with open("evaluation/evaluation_qa_dataset_5.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

contexts_list = []
answers_list = []

# Connect to Weaviate
with weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
) as client:

    collection = client.collections.get(COLLECTION_NAME)

    for question in df["question"]:

        print("Evaluating:", question)

        retrieved_docs = hybrid_retrieve_documents(
            query=question,
            collection=collection,
            model=model,
            top_k=20
        )

        # reranked_docs = rerank_with_cohere(question, retrieved_docs, top_k=5)
        reranked_docs = rerank_local(
                                        question,
                                        retrieved_docs,
                                        reranker_model,
                                        top_k=5
                                    )

        answer = generate_answer_with_llama(
            query=question,
            reranked_docs=reranked_docs
        )

        contexts = [doc.get("text", "") for doc in reranked_docs]

        contexts_list.append(contexts)
        answers_list.append(answer)

# Add results to dataset
df["contexts"] = contexts_list
df["answer"] = answers_list

dataset = Dataset.from_pandas(df)

# Run evaluation
result = evaluate(
    dataset,
    metrics=[
        faithfulness
    ],
    llm=ragas_llm,
    embeddings=embedding_model,
    run_config={"max_workers": 1, "timeout": 180} # Force sequential processing
)

print(f'result: {result}')

result_df = result.to_pandas()
result_df.to_csv("rag_evaluation_results.csv", index=False)

faithfulness_score = result["faithfulness"]
# relevancy_score = result["answer_relevancy"]
print(f"Value: {faithfulness_score} | Type: {type(faithfulness_score)}")

avg_score = np.nanmean(faithfulness_score)

print(f"Average Faithfulness Score: {avg_score}")

if avg_score < FAITHFULNESS_THRESHOLD:
    print("❌ Faithfulness below threshold")
    print("❌ Evaluation failed")
    sys.exit(1)
else:
    print("✅ Faithfulness above threshold")
    print("✅ Evaluation passed")

# if relevancy_score < RELEVANCY_THRESHOLD:
#     print("❌ Answer relevancy below threshold")
#     sys.exit(1)

