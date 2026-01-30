# Streamlit_app.py

import os
import streamlit as st
import pandas as pd
import sys

from preprocessing import run_preprocessing
from causal_discovery import run_causal_discovery

# === STREAMLIT CONFIG ===
st.set_page_config(page_title="Causal Discovery App", layout="wide")
st.title("Causal Discovery App")

# === FILE UPLOAD ===
file = st.file_uploader("Upload your CSV file", type=["csv"])

if file is not None:
    try:
        df_raw = pd.read_csv(file)
        st.subheader("Raw Data")
        st.dataframe(df_raw.head())

        # === PREPROCESSING ===
        df_clean, metadata = run_preprocessing(df_raw)

        st.subheader("Preprocessed Data")
        st.dataframe(df_clean.head())

        st.subheader("Variable Types")
        st.json(metadata["var_types_clean"])

        # === BUTTON TO RUN CAUSAL DISCOVERY ===
        if st.button("Run Causal Discovery"):
            with st.spinner("Running algorithms... please wait..."):
                try:
                    outputs = run_causal_discovery(df_clean)
                    st.success("Causal discovery completed!")

                    st.subheader("📁 Output Files")
                    st.json(outputs)

                    for alg, path in outputs.items():
                        if path.endswith(".graphml") and os.path.exists(path):
                            st.markdown(f"**{alg}**")
                            with open(path, "r") as f:
                                content = f.read()
                            st.code(content, language="xml")

                except Exception as e:
                    st.error(f"Error during causal discovery: {str(e)}")

    except Exception as e:
        st.error(f"Could not process file: {str(e)}")
