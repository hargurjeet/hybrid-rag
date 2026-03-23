import streamlit as st
from pipeline import run_pipeline
from components import render_sidebar, render_documents, render_answer

st.set_page_config(
    page_title="RAG Research Assistant",
    layout="wide"
)

st.title("🧠 Research Assistant (Production RAG)")

# Sidebar controls
config = render_sidebar()

# Input
query = st.text_input("🔍 Ask a research question:")

if st.button("Submit") and query:

    with st.spinner("Running RAG pipeline..."):

        answer, docs = run_pipeline(
            query=query,
            top_k=config["top_k"],
            alpha=config["alpha"],
            use_hybrid=config["use_hybrid"]
        )

    render_answer(answer)
    render_documents(docs)

    if config["debug"]:
        st.subheader("🐞 Debug Info")
        st.json(docs)