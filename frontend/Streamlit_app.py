import os
import streamlit as st
import pandas as pd
import requests
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from preprocessing import run_preprocessing

# === BACKEND CONFIG ===
BACKEND_URL = "http://localhost:8000/discover"

# === STREAMLIT APP ===
st.set_page_config(page_title="Causal Discovery App", layout="wide")
st.title("🔍 Causal Discovery App")

# === FILE UPLOAD ===
file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if file is not None:
    try:
        df_raw = pd.read_csv(file)
        st.subheader("📄 Raw Data")
        st.dataframe(df_raw.head())

        # === PREPROCESSING ===
        df_clean, metadata = run_preprocessing(df_raw)

        st.subheader("🧼 Preprocessed Data")
        st.dataframe(df_clean.head())

        st.subheader("ℹ️ Variable Types")
        st.json(metadata["var_types_clean"])

        # === BUTTON TO RUN CAUSAL DISCOVERY ===
        if st.button("🚀 Run Causal Discovery"):
            with st.spinner("Running algorithms... please wait"):
                try:
                    response = requests.post(
                        BACKEND_URL,
                        json=df_clean.to_dict(orient="records"),
                        timeout=120
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Causal discovery completed!")
                        st.subheader("📁 Output Files")
                        st.json(result["outputs"])
                    else:
                        st.error(f"❌ Failed: {response.status_code} - {response.text}")

                except Exception as e:
                    st.error(f"⚠️ Error connecting to backend: {str(e)}")

    except Exception as e:
        st.error(f"Could not process file: {str(e)}")
