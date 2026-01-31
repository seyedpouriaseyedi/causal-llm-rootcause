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
                    key=f"dl_{k}_{fname}"
                )


def _maybe_show_png(alg_dir: str, png_name_hint: str):
    png_path = os.path.join(alg_dir, png_name_hint)
    if os.path.exists(png_path):
        st.image(png_path, caption=png_name_hint, use_column_width=True)


def parse_exclude_vars(txt: str):
    items = [x.strip() for x in txt.split(",") if x.strip()]
    return set(items) if items else None


def parse_invalid_edges(txt: str):
    edges = set()
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] and parts[1]:
            edges.add((parts[0], parts[1]))
    return edges if edges else None


# =========================================
# Upload + preprocessing (shared by both tabs)
# =========================================
file = st.file_uploader("Upload your CSV file", type=["csv"])
if file is None:
    st.stop()

try:
    df_raw = pd.read_csv(file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.subheader("Raw Data (preview)")
st.dataframe(df_raw.head(50), use_container_width=True)

st.markdown("---")
st.subheader("Preprocessing")

try:
    df_clean, metadata = run_preprocessing(df_raw)
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

st.success("Preprocessing completed.")
st.subheader("Preprocessed Data (preview)")
st.dataframe(df_clean.head(50), use_container_width=True)

var_types_clean = metadata.get("var_types_clean", None)
st.subheader("Variable Types (from preprocessing)")
if var_types_clean is None:
    st.warning("metadata['var_types_clean'] not found. Causal discovery will auto-detect types.")
else:
    st.json(var_types_clean)


# =========================================
# Tabs
# =========================================
tab1, tab2 = st.tabs(["Causal Discovery", "LLM Root Cause Report"])


# =========================================
# TAB 1 — Causal discovery
# =========================================
with tab1:
    st.markdown("---")
    st.subheader("Causal Discovery Settings (Notebook Defaults)")

    colA, colB, colC = st.columns(3)

    with colA:
        out_dir = st.text_input("Output directory", value="outputs_step2")
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    with colB:
        st.markdown("**NOTEARS**")
        notears_k_runs = st.number_input("NOTEARS bootstraps (K)", min_value=1, value=30, step=1)
        notears_sample_frac = st.slider("NOTEARS sample fraction", 0.10, 1.00, 0.65, 0.01)
        notears_edge_q = st.slider("NOTEARS edge quantile (Q)", 0.50, 0.99, 0.90, 0.01)
        notears_freq_thr = st.slider("NOTEARS freq threshold", 0.00, 1.00, 0.60, 0.01)

    with colC:
        st.markdown("**GES / LiNGAM / PC**")
        ges_boot_runs = st.number_input("GES bootstraps", min_value=1, value=30, step=1)
        ges_boot_frac = st.slider("GES sample fraction", 0.10, 1.00, 0.70, 0.01)
        ges_stable_thr = st.slider("GES stable edge freq", 0.00, 1.00, 0.40, 0.01)

        lingam_boot_runs = st.number_input("LiNGAM bootstraps", min_value=1, value=30, step=1)
        lingam_freq_thr = st.slider("LiNGAM freq threshold", 0.00, 1.00, 0.10, 0.01)

        pc_boot_runs = st.number_input("PC bootstraps", min_value=1, value=30, step=1)
        pc_alpha = st.slider("PC alpha", 0.00, 1.00, 0.30, 0.01)
        pc_freq_thr = st.slider("PC freq threshold", 0.00, 1.00, 0.30, 0.01)

    st.markdown("### DAG-GNN (Notebook Defaults)")
    dag_col1, dag_col2, dag_col3 = st.columns(3)

    with dag_col1:
        daggnn_enabled = st.checkbox("Enable DAG-GNN", value=True)
        daggnn_boot_runs = st.number_input("DAG-GNN bootstraps", min_value=1, value=30, step=1)

    with dag_col2:
        daggnn_edge_abs_thr = st.slider("DAG-GNN edge abs threshold", 0.0, 0.2, 0.01, 0.005)
        daggnn_freq_thr = st.slider("DAG-GNN freq threshold", 0.0, 1.0, 0.20, 0.01)

    with dag_col3:
        daggnn_iters = st.number_input("DAG-GNN iters", min_value=10, value=500, step=10)
        daggnn_hidden_dim = st.number_input("DAG-GNN hidden dim", min_value=8, value=128, step=8)

    st.markdown("---")
    st.subheader("Display Options")
    plot = st.checkbox("Show interactive matplotlib windows (NOT recommended in Streamlit)", value=False)
    save_plots = st.checkbox("Save PNGs and show in Streamlit", value=True)

    st.markdown("---")
    st.subheader("PC Domain Filters (optional)")
    exclude_vars_text = st.text_area(
        "Exclude variables (comma-separated)",
        value="",
        help="Example: colA,colB,colC"
    )
    invalid_edges_text = st.text_area(
        "Invalid directed edges (one per line as: source,target)",
        value="",
        help="Example line: ProductionVolume,DefectRate"
    )
    domain_exclude_vars = parse_exclude_vars(exclude_vars_text)
    domain_invalid_edges = parse_invalid_edges(invalid_edges_text)

    st.markdown("---")
    if st.button("Run Causal Discovery", type="primary"):
        cfg = DiscoveryConfig(
            out_dir=out_dir,
            seed=int(seed),

            # NOTEARS
            notears_k_runs=int(notears_k_runs),
            notears_sample_frac=float(notears_sample_frac),
            notears_edge_q=float(notears_edge_q),
            notears_freq_thr=float(notears_freq_thr),

            # GES
            ges_boot_runs=int(ges_boot_runs),
            ges_boot_frac=float(ges_boot_frac),
            ges_stable_thr=float(ges_stable_thr),
            ges_keep_continuous_only=True,

            # LiNGAM
            lingam_boot_runs=int(lingam_boot_runs),
            lingam_freq_thr=float(lingam_freq_thr),

            # DAG-GNN
            daggnn_enabled=bool(daggnn_enabled),
            daggnn_boot_runs=int(daggnn_boot_runs),
            daggnn_edge_abs_thr=float(daggnn_edge_abs_thr),
            daggnn_freq_thr=float(daggnn_freq_thr),
            daggnn_iters=int(daggnn_iters),
            daggnn_hidden_dim=int(daggnn_hidden_dim),

            # PC
            pc_boot_runs=int(pc_boot_runs),
            pc_alpha=float(pc_alpha),
            pc_freq_thr=float(pc_freq_thr),

            # Plotting
            plot=bool(plot),
            save_plots=bool(save_plots),
        )

        with st.spinner("Running algorithms... this can take a while (especially DAG-GNN)."):
            try:
                outputs = run_causal_discovery(
                    df_clean,
                    var_types=var_types_clean,
                    config=cfg,
                    domain_exclude_vars=domain_exclude_vars,
                    domain_invalid_edges=domain_invalid_edges,
                )
            except Exception as e:
                st.error(f"Error during causal discovery: {e}")
                st.stop()

        st.success("Causal discovery completed!")

        # ✅ Persist for tab2 and reruns
        st.session_state["cd_outputs"] = outputs
        st.session_state["df_clean"] = df_clean
        st.session_state["var_types_clean"] = var_types_clean
        st.session_state["llm_payload"] = None  # reset payload for new run

    # ---------------------------------
    # Show results from last run (if exists)
    # ---------------------------------
    outputs = st.session_state.get("cd_outputs")

    if outputs is None:
        st.info("No causal discovery results yet. Click **Run Causal Discovery** above.")
    else:
        st.markdown("---")
        st.subheader("Results Summary")

        st.write("**Output directory:**", outputs.get("out_dir"))
        st.write("**Numeric cols used:**", outputs.get("numeric_cols"))

        runs = outputs.get("runs", {})

        # NOTEARS
        if "NOTEARS" in runs:
            st.markdown("## NOTEARS (Bootstrapped)")
            info = runs["NOTEARS"]
            st.write("Algorithm directory:", info.get("alg_dir"))
            _show_file_links({
                "adj": info.get("adj_path"),
                "edges": info.get("edges_path"),
                "graph": info.get("graph_path"),
                "summary": info.get("summary_path"),
            })
            edges_df = _safe_read_csv(info.get("edges_path", ""))
            if edges_df is not None:
                st.dataframe(edges_df.head(50), use_container_width=True)
            if save_plots and info.get("alg_dir"):
                _maybe_show_png(info["alg_dir"], "FINAL_graph_NOTEARS.png")

        # GES
        if "GES" in runs:
            st.markdown("## GES (Bootstrap Stability)")
            info = runs["GES"]
            st.write("Algorithm directory:", info.get("alg_dir"))
            stable_path = info.get("stable_edges_path")
            if stable_path and os.path.exists(stable_path):
                _show_file_links({"stable_edges": stable_path})
                stable_df = _safe_read_csv(stable_path)
                if stable_df is not None:
                    st.dataframe(stable_df.head(50), use_container_width=True)
            if save_plots and info.get("alg_dir"):
                _maybe_show_png(info["alg_dir"], "STABLE_graph_GES.png")

        # LiNGAM
        if "LiNGAM" in runs:
            st.markdown("## LiNGAM (Bootstrapped)")
            info = runs["LiNGAM"]
            st.write("Algorithm directory:", info.get("alg_dir"))
            _show_file_links({
                "adj": info.get("adj_path"),
                "edges": info.get("edges_path"),
                "graph": info.get("graph_path"),
                "summary": info.get("summary_path"),
            })
            edges_df = _safe_read_csv(info.get("edges_path", ""))
            if edges_df is not None:
                st.dataframe(edges_df.head(50), use_container_width=True)
            if save_plots and info.get("alg_dir"):
                _maybe_show_png(info["alg_dir"], "FINAL_graph_LiNGAM_Bootstrap.png")

        # DAG-GNN
        if "DAGGNN" in runs:
            st.markdown("## DAG-GNN (Bootstrapped)")
            info = runs["DAGGNN"]
            if info.get("skipped"):
                st.warning(f"DAG-GNN skipped: {info.get('reason', 'unknown')}")
                if "selected_vars" in info:
                    st.write("Selected vars:", info["selected_vars"])
                if "roles" in info:
                    st.json(info["roles"])
            else:
                st.write("Algorithm directory:", info.get("alg_dir"))
                _show_file_links({
                    "adj": info.get("adj_path"),
                    "edges": info.get("edges_path"),
                    "graph": info.get("graph_path"),
                    "summary": info.get("summary_path"),
                })
                edges_df = _safe_read_csv(info.get("edges_path", ""))
                if edges_df is not None:
                    st.dataframe(edges_df.head(50), use_container_width=True)
                if save_plots and info.get("alg_dir"):
                    _maybe_show_png(info["alg_dir"], "FINAL_graph_DAG-GNN_Bootstrap.png")

        # PC
        if "PC" in runs:
            st.markdown("## PC (Bootstrapped)")
            info = runs["PC"]
            st.write("Algorithm directory:", info.get("alg_dir"))
            if info.get("skipped"):
                st.warning("PC skipped (not enough variables after filtering).")
            else:
                _show_file_links({
                    "adj": info.get("adj_path"),
                    "edges": info.get("edges_path"),
                    "graph": info.get("graph_path"),
                    "summary": info.get("summary_path"),
                })
                edges_df = _safe_read_csv(info.get("edges_path", ""))
                if edges_df is not None:
                    st.dataframe(edges_df.head(50), use_container_width=True)
                if save_plots and info.get("alg_dir"):
                    _maybe_show_png(info["alg_dir"], "FINAL_graph_PC_Bootstrap.png")


# =========================================
# TAB 2 — LLM Root Cause Report integration
# =========================================
with tab2:
    st.markdown("---")
    st.subheader("LLM Root Cause Report (Fixed Function)")

    if st.session_state.get("cd_outputs") is None:
        st.info("Run **Causal Discovery** first (Tab 1).")
        st.stop()

    outputs = st.session_state["cd_outputs"]
    df_clean_ss = st.session_state["df_clean"]

    # ---- Choose target + incident ----
    default_target = "Machine failure" if "Machine failure" in df_clean_ss.columns else df_clean_ss.columns[-1]
    target = st.selectbox("Target variable", options=list(df_clean_ss.columns),
                          index=list(df_clean_ss.columns).index(default_target))

    st.markdown("### Select an incident")
    if target in df_clean_ss.columns and df_clean_ss[target].nunique(dropna=True) <= 2:
        fail_rows = df_clean_ss.index[df_clean_ss[target] == 1].tolist()
        mode = st.radio("Incident pool", ["Failures only (target=1)", "All rows"], horizontal=True)
        if mode.startswith("Failures") and len(fail_rows) > 0:
            incident_index = st.selectbox("Incident index", options=fail_rows)
        else:
            incident_index = st.number_input("Incident index (row number)", min_value=0, max_value=len(df_clean_ss)-1, value=0, step=1)
    else:
        incident_index = st.number_input("Incident index (row number)", min_value=0, max_value=len(df_clean_ss)-1, value=0, step=1)

    # ---- Consensus parameters ----
    st.markdown("### Consensus settings (deterministic)")
    c1, c2, c3 = st.columns(3)
    with c1:
        min_alg_count = st.number_input("Min algorithm agreement (edges)", min_value=1, max_value=5, value=2, step=1)
    with c2:
        min_support_mean = st.slider("Min mean support", 0.0, 1.0, 0.30, 0.01)
    with c3:
        max_path_len = st.number_input("Max path length", min_value=2, max_value=8, value=4, step=1)

    # ---- Work dir ----
    default_work_dir = os.path.join(
        outputs.get("out_dir", "outputs_step2"),
        f"llm_phase_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    work_dir = st.text_input("Work directory for LLM phase outputs", value=default_work_dir)

    # ---- Build deterministic payload ----
    if st.button("Build consensus + ranking + evidence", type="primary"):
        with st.spinner("Building consensus graph and ranking candidates..."):
            payload = generate_root_cause_report_inputs(
                outputs=outputs,
                df_clean=df_clean_ss,
                target=target,
                incident_index=int(incident_index),
                work_dir=work_dir,
                min_alg_count=int(min_alg_count),
                min_support_mean=float(min_support_mean),
                max_path_len=int(max_path_len),
            )
        st.session_state["llm_payload"] = payload
        st.success("LLM-prep completed. Candidate ranking + evidence ready.")

    if st.session_state.get("llm_payload") is None:
        st.info("Click **Build consensus + ranking + evidence** to prepare LLM inputs.")
        st.stop()

    payload = st.session_state["llm_payload"]

    # ---- Show ranking + evidence ----
    st.markdown("## Ranked candidates")
    cand_path = payload["ranking"]["candidates_path"]
    if os.path.exists(cand_path):
        cand_df = pd.read_csv(cand_path)
        st.dataframe(cand_df.head(15), use_container_width=True)

    st.markdown("## Incident evidence (top candidates)")
    st.json(payload["incident_evidence"])

    st.markdown("## Consensus outputs")
    st.write("Consensus edges:", payload["consensus"]["consensus_edges_path"])
    st.write("Consensus graph:", payload["consensus"]["consensus_graph_path"])
    _show_file_links({
        "consensus_edges": payload["consensus"]["consensus_edges_path"],
        "consensus_graph": payload["consensus"]["consensus_graph_path"],
        "consensus_summary": payload["consensus"]["consensus_summary_path"],
        "candidates": payload["ranking"]["candidates_path"],
        "top_paths": payload["ranking"]["top_paths_path"],
    })

    # ---- Build prompt ----
    st.markdown("## Prompt (fixed function report)")
    prompt_template_path = os.path.join("llm_phase", "prompt_template.txt")
    prompt = build_prompt(
        prompt_template_path=prompt_template_path,
        payload=payload,
        target=target,
        incident_index=int(incident_index),
    )
    st.text_area("Prompt to send to LLM", value=prompt, height=320)

    # ---- Manual JSON paste mode (no API yet) ----
    st.markdown("## LLM response (paste JSON here)")
    st.caption("Paste the model's JSON output. Validator will enforce allowed vars/edges.")
    llm_json_text = st.text_area("LLM JSON response", value="", height=250)

    if st.button("Validate LLM JSON"):
        allowed_vars = set(payload["allowed_vars"])
        allowed_edges = set((a, b) for a, b in payload["allowed_edges"])

        try:
            # run_llm_and_validate expects a function that returns JSON string
            parsed = json.loads(llm_json_text)
            parsed_valid = run_llm_and_validate(
                call_llm_fn=lambda _prompt: json.dumps(parsed),
                prompt="(manual paste mode)",
                allowed_vars=allowed_vars,
                allowed_edges=allowed_edges,
            )
            st.success("LLM output is valid and compliant.")
            st.json(parsed_valid)
        except Exception as e:
            st.error(f"Validation failed: {e}")
