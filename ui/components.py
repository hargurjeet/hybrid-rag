import streamlit as st

def render_sidebar():
    st.sidebar.title("⚙️ Controls")

    top_k = st.sidebar.slider("Top-K Results", 1, 10, 5)
    alpha = st.sidebar.slider("Hybrid Alpha", 0.0, 1.0, 0.5)

    retrieval_mode = st.sidebar.radio(
        "Retrieval Mode",
        ["Hybrid", "Vector"]
    )

    debug = st.sidebar.checkbox("Show Debug Info")

    return {
        "top_k": top_k,
        "alpha": alpha,
        "use_hybrid": retrieval_mode == "Hybrid",
        "debug": debug
    }


def render_documents(docs):
    st.subheader("📄 Retrieved Documents")

    for i, doc in enumerate(docs):
        with st.expander(f"Rank {i+1} | Score: {round(doc['score'], 3)}"):
            st.write(f"**Paper ID:** {doc['paper_id']}")
            st.write(f"**Category:** {doc['categories']}")
            st.write(doc["text"][:500])


def render_answer(answer):
    st.subheader("🧠 Final Answer")
    st.write(answer)