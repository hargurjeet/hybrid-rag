import streamlit as st
import subprocess
import pandas as pd
import os

RESULT_FILE = "rag_evaluation_results.csv"


def run_evaluation_stream():
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

    st.title("📊 RAG Evaluation")

    # -------------------------------
    # About Section
    # -------------------------------
    with st.expander("ℹ️ About Evaluation"):

        st.markdown("""
### ⚠️ Evaluation Scope
- Due to **computational constraints**, evaluation is performed on a **small subset of Q&A pairs**

**Currently:**
- ✅ ~5 evaluation questions  
- ✅ Metric: **Faithfulness (RAGAS)**  

**Production Setup:**
- Evaluation would run on a **much larger dataset**
- Multiple metrics (faithfulness, relevancy, etc.) would be tracked  

---

### 🔄 CI/CD Integration (Quality Gating)
- Automated evaluation pipeline ensures quality before deployment

**Workflow:**
1. Run RAG evaluation  
2. Compute average faithfulness score  
3. Compare against threshold  

**Outcome:**
- ❌ Below threshold → Deployment fails  
- ✅ Above threshold → Deployment succeeds  

---

### 🎯 Why This Matters
- Prevents silent quality degradation  
- Ensures reliable RAG performance  
- Brings ML systems closer to production-grade engineering  
""")

    # -------------------------------
    # Run Evaluation
    # -------------------------------
    if st.button("🚀 Run Evaluation"):

        st.info("Running evaluation...")

        log_container = st.empty()
        log_text = ""

        for line in run_evaluation_stream():
            log_text += line

            # Prevent UI slowdown
            if len(log_text) > 5000:
                log_text = log_text[-5000:]

            log_container.code(log_text)

        st.success("✅ Evaluation Completed")

        # -------------------------------
        # Results
        # -------------------------------
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

            # -------------------------------
            # Deployment Logic
            # -------------------------------
            st.markdown("### 🧪 Deployment Gate Logic")

            st.code("""
if avg_faithfulness < threshold:
    fail_pipeline()
else:
    deploy()
""")

            st.caption("Threshold: 0.3 (configurable in evaluation pipeline)")

        else:
            st.warning("⚠️ Results file not found. Evaluation may have failed.")