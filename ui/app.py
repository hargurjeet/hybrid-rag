import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
from pipeline import run_pipeline
from components import render_sidebar, render_documents, render_answer
from evaluation import render_evaluation_tab
import time

if "history" not in st.session_state:
    st.session_state.history = []
# -------------------------------
# Session State Init
# -------------------------------
if "query" not in st.session_state:
    st.session_state.query = ""

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="RAG Research Assistant",
    layout="wide"
)

st.title("🧠 Research Assistant (Production HYBRID RAG)")

with st.expander("ℹ️ About this Project"):

    st.markdown("""
### 📚 Data Source
- Dataset sourced from **Kaggle – Cornell University arXiv dataset**
- Due to large dataset size, a **subset of ~50 research papers** was selected

---

### ⚙️ Processing & Retrieval
- Papers are **chunked using recursive text splitting**
- Stored in a **vector database (Weaviate)**
- Retrieval uses:
  - 🔹 Hybrid Search (BM25 + Vector)
  - 🔹 Reranking (Cohere)

---

### 🧠 LLM Strategy (Cost-Optimized)
- Local inference using:
  - **Ollama**
  - Models: `llama3.2`, `phi3:mini`
- Used for:
  - Development
  - Offline experimentation

---

### ⚡ Production Inference
- Integrated **Groq API** for fast inference
- Reason:
  - Local models not viable on free-tier hosting (e.g., Hugging Face Spaces)
  - Groq provides **low-latency, high-performance inference**

---

### 🎯 Goal of This Project
To build a **production-grade RAG system** with:
- Hybrid retrieval
- Reranking
- Evaluation (RAGAS)
- Observability via UI
""")

tab1, tab2 = st.tabs(["🔍 Ask Questions", "📊 Evaluation"])

with tab1:

    st.subheader("🔥 Try Sample Questions")

    sample_questions = [
        "What are multidimension recurrent neural networks?",
        "What bottom friction model is used in the shallow-water flows study?",
        "Explain the Chezy law"
    ]

    cols = st.columns(3, gap="medium")

    for i, question in enumerate(sample_questions):
        with cols[i]:
            if st.button(question, use_container_width=True):
                st.session_state.query = question

    config = render_sidebar()

    st.sidebar.subheader("🕘 Query History")

    for i, item in enumerate(reversed(st.session_state.history[-5:])):
        if st.sidebar.button(item["query"], key=f"history_{i}"):
            st.session_state.query = item["query"]

    query = st.text_input(
        "🔍 Ask a research question:",
        value=st.session_state.query
    )

    st.session_state.query = query

    if st.button("Submit") and query:

        with st.spinner("Running RAG pipeline..."):

            start_time = time.time()

            answer, docs = run_pipeline(
                query=query,
                top_k=config["top_k"],
                alpha=config["alpha"],
                use_hybrid=config["use_hybrid"]
            )

            latency = time.time() - start_time

        render_answer(answer)
        st.caption(f"⏱️ Latency: {latency:.2f} sec")
        st.caption(f"📄 Retrieved: {config['top_k']} docs")
        render_documents(docs)

        if config["debug"]:
            st.subheader("🐞 Debug Info")
            st.json(docs)

        st.session_state.history.append({
        "query": query,
        "answer": answer
                        })

with tab2:
    render_evaluation_tab()