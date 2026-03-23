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

st.title("🧠 Research Assistant (Production RAG)")

tab1, tab2 = st.tabs(["🔍 Ask Questions", "📊 Evaluation"])

with tab1:

    st.subheader("🔥 Try Sample Questions")

    sample_questions = [
        "What are multidimension recurrent neural networks?",
        "What bottom friction model is used in the shallow-water flows study?",
        "Explain the Chezy law"
    ]

    cols = st.columns(3)

    for i, question in enumerate(sample_questions):
        if cols[i].button(question):
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