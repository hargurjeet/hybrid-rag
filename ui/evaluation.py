import streamlit as st
import subprocess
import pandas as pd
import os

RESULT_FILE = "rag_evaluation_results.csv"


def run_evaluation_stream():
    """
    Run evaluation script and stream logs in real-time
    """
    process = subprocess.Popen(
        ["python", "evaluation/evaluate_rag.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in iter(process.stdout.readline, ''):
        yield line

    process.stdout.close()
    process.wait()


def render_evaluation_tab():

    import os
    import pandas as pd
    import streamlit as st

    RESULT_FILE = "rag_evaluation_results.csv"

    st.title("📊 RAG Evaluation")

    if st.button("🚀 Run Evaluation"):

        st.info("Running evaluation...")

        log_container = st.empty()   # 👈 dynamic UI
        log_text = ""

        for line in run_evaluation_stream():
            log_text += line
            log_container.text(log_text)  # 👈 live update

        st.success("✅ Evaluation Completed")

        # Load results
        if os.path.exists(RESULT_FILE):
            df = pd.read_csv(RESULT_FILE)

            st.subheader("📈 Results")
            st.dataframe(df)

            avg_score = df["faithfulness"].mean()

            st.metric(
                label="Average Faithfulness",
                value=round(avg_score, 3)
            )

            if avg_score < 0.3:
                st.error("❌ Evaluation Failed")
            else:
                st.success("✅ Evaluation Passed")