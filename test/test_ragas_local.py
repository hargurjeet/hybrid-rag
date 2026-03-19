from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

from langchain_community.llms import Ollama
from ragas.llms import LangchainLLMWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------
# Local judge model
# -------------------

judge_llm = Ollama(
    model="mistral:7b",
    temperature=0
)

ragas_llm = LangchainLLMWrapper(judge_llm)

# -------------------
# Local embeddings
# -------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# -------------------
# Fake dataset
# -------------------

data = {
    "question": [
        "What is retrieval augmented generation?"
    ],
    "answer": [
        "Retrieval augmented generation is a technique that retrieves documents and uses them to generate answers."
    ],
    "contexts": [[
        "Retrieval augmented generation (RAG) combines document retrieval with language generation."
    ]],
    "ground_truth": [
        "RAG is a system that retrieves documents and generates answers using them."
    ]
}

dataset = Dataset.from_dict(data)

# -------------------
# Run evaluation
# -------------------

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=ragas_llm,
    embeddings=embedding_model
)

print(result)