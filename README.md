---
title: Hybrid Rag Hf
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
---

# 🧠 Production-Grade Hybrid RAG System

> 🚀 A production-ready Retrieval-Augmented Generation (RAG) system with hybrid search, reranking, evaluation gating, and real-time observability.

---

## 🔥 Live Demo

👉 **Streamlit App (Hugging Face Spaces)**  
https://huggingface.co/spaces/Hargurjeet/hybrid_rag_hf

---

## 📌 Overview

This project implements a **production-grade RAG pipeline** that goes beyond a basic chatbot by incorporating:

- 🔍 **Hybrid Retrieval** (BM25 + Vector Search)
- 🎯 **Reranking** for improved relevance
- 🧠 **Flexible LLM Backend** (Groq / Hugging Face / Ollama)
- 📊 **Evaluation with RAGAS**
- 🚦 **CI/CD Quality Gating**
- 🖥️ **Interactive UI with Observability**

---

## 🏗️ System Architecture
```
User Query
    ↓
Hybrid Retrieval (BM25 + Vector Search)
    ↓
Top-K Documents
    ↓
Reranker (Cohere / Cross-Encoder)
    ↓
Refined Context
    ↓
LLM (Groq / HF / Ollama)
    ↓
Final Answer + Citations
```

---

## ⚙️ Key Features

### 🔍 Hybrid Retrieval
- Combines:
  - **BM25 (keyword search)**
  - **Dense embeddings (semantic search)**
- Improves recall and robustness

---

### 🎯 Reranking
- Uses **Cohere reranker**
- Reorders retrieved chunks for:
  - Higher relevance
  - Better grounding

---

### 🧠 Multi-LLM Support

**Local (cost-efficient):**
- `Ollama` → Llama 3.2, Phi-3 mini

**Cloud (production-ready):**
- `Groq` → ultra-fast inference
- `Hugging Face Inference API`

---

### 📊 Evaluation (RAGAS)
- Metric used:
  - ✅ **Faithfulness**
- Evaluates:
  - Hallucination risk
  - Grounding quality

---

### 🚦 CI/CD Quality Gating

Automated evaluation pipeline ensures quality before deployment:
```python
if faithfulness < threshold:
    fail_pipeline()
else:
    deploy()
```

👉 Prevents low-quality RAG deployments

---

### 🖥️ Streamlit UI

**Includes:**

- 🔍 Query interface
- 📄 Retrieved document inspection
- 🧠 Answer with citations
- ⏱️ Latency tracking
- 🕘 Query history
- 📊 Evaluation tab with:
  - Live logs
  - Metrics
  - Pass/fail status

---

## 📚 Dataset

- **Source:** Kaggle – Cornell University arXiv dataset
- Due to size constraints:
  - Subset of ~50 research papers used
- Documents:
  - Chunked
  - Embedded
  - Stored in Weaviate

---

## 🧰 Tech Stack

| Category    | Tools                              |
|-------------|------------------------------------|
| Language    | Python                             |
| UI          | Streamlit                          |
| Vector DB   | Weaviate                           |
| Embeddings  | Sentence Transformers              |
| Retrieval   | Hybrid (BM25 + Vector)             |
| Reranking   | Cohere                             |
| LLM         | Groq, HuggingFace, Ollama          |
| Evaluation  | RAGAS                              |
| Deployment  | Hugging Face Spaces                |
| Packaging   | uv (pyproject.toml)                |

---

## 📁 Project Structure
```
HYBRID_RAG/
├── ui/              # Streamlit UI
├── src/             # Core RAG logic
├── evaluation/      # RAGAS evaluation pipeline
├── dataset/         # Raw data (not included in Docker)
├── results/         # Evaluation outputs
├── pyproject.toml   # Dependencies (uv)
├── uv.lock          # Reproducible lockfile
```

---

## 🚀 Running Locally

### 1. Clone repo
```bash
git clone https://github.com/hargurjeet/hybrid-rag
cd hybrid-rag
```

### 2. Install dependencies (uv)
```bash
uv sync
```

### 3. Set environment variables
```bash
export WEAVIATE_URL=...
export WEAVIATE_API_KEY=...
export CO_API_KEY=...
export GROQ_API_KEY=...
```

### 4. Run Streamlit
```bash
uv run streamlit run ui/app.py
```

---

## 🧪 Run Evaluation
```bash
uv run python evaluation/evaluate_rag.py
```

---

## 🐳 Docker Deployment
```bash
docker build -t hybrid-rag .
docker run -p 8501:8501 hybrid-rag
```

---

## ⚠️ Limitations

- Evaluation uses a small subset (~5 questions) due to compute constraints
- Full-scale evaluation would include:
  - Larger dataset
  - Multiple metrics (relevancy, precision, etc.)

---

## 🎯 Future Improvements

- 📈 Multi-metric evaluation dashboard
- ⚡ Fast vs full evaluation modes
- 🧠 Multi-agent RAG (LangGraph)
- 📊 Experiment tracking (LangSmith-style)
- 💾 Persistent memory + chat history

---

## 💡 Key Learnings

- Hybrid retrieval significantly improves recall vs vector-only
- Reranking is critical for answer quality
- Evaluation (RAGAS) is essential for production readiness
- CI gating prevents silent degradation in RAG systems

---

## 🤝 Connect

If you found this useful or want to collaborate:

- **LinkedIn:** https://www.linkedin.com/in/hargurjeet/

---

## ⭐ Star This Repo

If you like this project, consider giving it a ⭐ — it helps a lot! 


