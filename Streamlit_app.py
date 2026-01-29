# Streamlit_app.py

import streamlit as st
import pandas as pd
import os

from preprocessing import run_preprocessing
from causal_discovery import run_causal_discovery

st.set_page_config(page_title="Causal Discovery App", layout="wide")

st.title("🧠 Causal Discovery in Manufacturing Data")
st.write("Upload your dataset, run causal discovery algorithms, and ask questions.")

# Step 1 – Upload
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Step 2 – Clean & Preprocess
    if st.button("🧹 Run Preprocessing"):
        df_clean, metadata = run_preprocessing(df)
        st.session_state["df_clean"] = df_clean
        st.session_state["metadata"] = metadata
        st.success("Preprocessing complete!")
        st.dataframe(df_clean.head())

    # Step 3 – Causal Discovery
    if "df_clean" in st.session_state:
        if st.button("🔍 Run Causal Discovery"):
            outputs = run_causal_discovery(st.session_state["df_clean"])
            st.session_state["outputs"] = outputs
            st.success("Causal discovery completed!")

            for alg, path in outputs.items():
                st.write(f"**{alg} result:**")
                if path.endswith(".graphml"):
                    st.download_button(
                        label=f"Download {alg} GraphML",
                        data=open(path, "rb").read(),
                        file_name=f"{alg}.graphml"
                    )

    # Step 4 – Ask LLM (coming next)
    if "outputs" in st.session_state:
        st.subheader("🤖 Ask LLM about the causal graph")

        question = st.text_area("Your question (e.g., What are the strongest causes of delay?)")

        if st.button("Ask"):
            st.warning("LLM integration not yet implemented.")
