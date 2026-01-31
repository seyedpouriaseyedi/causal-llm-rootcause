# Streamlit_app.py
import os
import json
from datetime import datetime

import streamlit as st
import pandas as pd

from preprocessing import run_preprocessing
from causal_discovery import run_causal_discovery, DiscoveryConfig

from llm_phase.pipeline import (
    generate_root_cause_report_inputs,
    build_prompt,
    run_llm_and_validate,
)

from llm_phase.openai_client import generate_root_cause_report_json, chat_grounded


# =========================================
# Session defaults (critical for tabs)
# =========================================
if "cd_outputs" not in st.session_state:
    st.session_state["cd_outputs"] = None
if "df_clean" not in st.session_state:
    st.session_state["df_clean"] = None
if "var_types_clean" not in st.session_state:
    st.session_state["var_types_clean"] = None
if "llm_payload" not in st.session_state:
    st.session_state["llm_payload"] = None
if "llm_work_dir" not in st.session_state:
    st.session_state["llm_work_dir"] = None
if "llm_payload_sig" not in st.session_state:
    st.session_state["llm_payload_sig"] = None
if "run_id" not in st.session_state:
    st.session_state["run_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
if "qa_messages" not in st.session_state:
    st.session_state["qa_messages"] = None  # will be initialized in Tab 3


# =========================================
# Streamlit config
# =========================================
st.set_page_config(page_title="Causal Discovery App", layout="wide")
st.title("Causal Discovery App")


# =========================================
# Helpers
# =========================================
def _safe_read_csv(path: str) -> pd.DataFrame | None:
    try:
        if path and os.path.exists(path) and path.lower().endswith(".csv"):
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def _show_file_links(paths: dict):
    """Render download buttons for any existing files in a dict."""
    for k, p in paths.items():
        if isinstance(p, str) and os.path.exists(p):
            fname = os.path.basename(p)
            with open(p, "rb") as f:
                st.download_button(
                    label=f"Download {fname}",
                    data=f,
                    file_name=fname,
                    mime="application/octet-stream",
                    key=f"dl_{k}_{fname}",
                )


def _maybe_show_png(alg_dir: str, png_name_hint: str):
    png_path = os.path.join(alg_dir, png_name_hint)
    if os.path.exists(png_path):
        st.image(png_path, caption=png_name_hint, use_column_width=True)


def _df_to_compact_text(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df is None or df.empty:
        return "(empty)"
    df2 = df.head(max_rows).copy()
    return df2.to_csv(index=False)


def _read_text_file(path: str) -> str:
    try:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        return ""
    return ""


def build_llm_context_pack(outputs: dict, var_types_clean: dict | None, llm_payload: dict | None) -> str:
    """
    Builds a grounded context pack from your causal discovery outputs + (optional) consensus/ranking artifacts.
    Keep it short to avoid token blowup.
    """
    lines = []
    lines.append("SYSTEM RULES:")
    lines.append("- You are a grounded assistant. Use ONLY the data in CONTEXT below.")
    lines.append("- If the answer is not supported by the context, say: 'Not enough information in the provided outputs.'")
    lines.append("- Do NOT invent edges, variables, or numbers.")
    lines.append("- When you claim something, cite which table/file section it came from (e.g., NOTEARS edges, LiNGAM edges, consensus_edges).")
    lines.append("")

    # Variable types
    lines.append("CONTEXT: VARIABLE TYPES")
    if var_types_clean:
        for k, v in list(var_types_clean.items())[:200]:
            lines.append(f"- {k}: {v}")
    else:
        lines.append("(var_types_clean not provided)")
    lines.append("")

    runs = outputs.get("runs", {}) if outputs else {}

    def add_edges_section(title: str, edges_path: str, max_rows: int = 30):
        lines.append(f"CONTEXT: {title}")
        df = _safe_read_csv(edges_path) if edges_path else None
        if df is None or df.empty:
            lines.append("(missing/empty)")
        else:
            cols = [c for c in ["source", "target", "freq", "weight_median", "corr", "abs_weight_median"] if c in df.columns]
            if cols:
                df = df[cols]
            lines.append(_df_to_compact_text(df, max_rows=max_rows))
        lines.append("")

    if "NOTEARS" in runs:
        add_edges_section("NOTEARS edges (FINAL_edges_...csv)", runs["NOTEARS"].get("edges_path"))
    if "LiNGAM" in runs:
        add_edges_section("LiNGAM edges (FINAL_edges_...csv)", runs["LiNGAM"].get("edges_path"))
    if "DAGGNN" in runs and not runs["DAGGNN"].get("skipped"):
        add_edges_section("DAG-GNN edges (FINAL_edges_...csv)", runs["DAGGNN"].get("edges_path"))
    if "PC" in runs and not runs["PC"].get("skipped"):
        add_edges_section("PC edges (FINAL_edges_...csv)", runs["PC"].get("edges_path"))
    if "GES" in runs:
        add_edges_section("GES stable edges (STABLE_edges_GES.csv)", runs["GES"].get("stable_edges_path"))

    # Optional: consensus/ranking if you built llm_payload
    if llm_payload:
        cons = llm_payload.get("consensus", {})
        rank = llm_payload.get("ranking_
