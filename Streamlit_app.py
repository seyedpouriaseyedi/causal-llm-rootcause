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
# Session defaults
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
    st.session_state["qa_messages"] = None


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
    return df.head(max_rows).to_csv(index=False)


def _read_text_file(path: str) -> str:
    try:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        return ""
    return ""


def build_llm_context_pack(outputs: dict, var_types_clean: dict | None, llm_payload: dict | None) -> str:
    lines = []
    lines.append("SYSTEM RULES:")
    lines.append("- You are a grounded assistant. Use ONLY the data in CONTEXT below.")
    lines.append("- If the answer is not supported by the context, say: 'Not enough information in the provided outputs.'")
    lines.append("- Do NOT invent edges, variables, or numbers.")
    lines.append("- When you claim something, cite which table/file section it came from (e.g., NOTEARS edges, LiNGAM edges, consensus_edges).")
    lines.append("")

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
        add_edges_section("NOTEARS edges", runs["NOTEARS"].get("edges_path"))
    if "LiNGAM" in runs:
        add_edges_section("LiNGAM edges", runs["LiNGAM"].get("edges_path"))
    if "DAGGNN" in runs and not runs["DAGGNN"].get("skipped"):
        add_edges_section("DAG-GNN edges", runs["DAGGNN"].get("edges_path"))
    if "PC" in runs and not runs["PC"].get("skipped"):
        add_edges_section("PC edges", runs["PC"].get("edges_path"))
    if "GES" in runs:
        add_edges_section("GES stable edges", runs["GES"].get("stable_edges_path"))

    if llm_payload:
        cons = llm_payload.get("consensus", {})
        rank = llm_payload.get("ranking", {})

        lines.append("CONTEXT: CONSENSUS EDGES (if available)")
        cons_edges = cons.get("consensus_edges_path")
        df = _safe_read_csv(cons_edges) if cons_edges else None
        if df is None or df.empty:
            lines.append("(missing/empty)")
        else:
            cols = [c for c in ["source", "target", "algo_count", "support_mean", "edge_score", "conflict", "algs"] if c in df.columns]
            if cols:
                df = df[cols]
            lines.append(_df_to_compact_text(df, max_rows=40))
        lines.append("")

        lines.append("CONTEXT: RANKED CANDIDATES (if available)")
        cand_path = rank.get("candidates_path")
        df = _safe_read_csv(cand_path) if cand_path else None
        if df is None or df.empty:
            lines.append("(missing/empty)")
        else:
            lines.append(_df_to_compact_text(df, max_rows=30))
        lines.append("")

        lines.append("CONTEXT: TOP PATHS JSON (if available)")
        top_paths_path = rank.get("top_paths_path")
        top_paths_text = _read_text_file(top_paths_path) if top_paths_path else ""
        if top_paths_text.strip():
            lines.append(top_paths_text[:4000] + ("\n...(truncated)" if len(top_paths_text) > 4000 else ""))
        else:
            lines.append("(missing/empty)")
        lines.append("")

    return "\n".join(lines)


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
# Upload + preprocessing
# =========================================
file = st.file_uploader("Upload your CSV file", type=["csv"], key="uploader_csv")
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
tab1, tab2, tab3 = st.tabs(["Causal Discovery", "LLM Root Cause Report", "LLM Q&A (Grounded)"])


# =========================================
# TAB 1 — Causal discovery
# =========================================
with tab1:
    st.markdown("---")
    st.subheader("Causal Discovery Settings (Notebook Defaults)")

    colA, colB, colC = st.columns(3)

    with colA:
        out_dir = st.text_input("Output directory", value="outputs_step2", key="cd_out_dir")
        seed = st.number_input("Random seed", min_value=0, value=42, step=1, key="cd_seed")

    with colB:
        st.markdown("**NOTEARS**")
        notears_k_runs = st.number_input("NOTEARS bootstraps (K)", min_value=1, value=30, step=1, key="notears_k")
        notears_sample_frac = st.slider("NOTEARS sample fraction", 0.10, 1.00, 0.65, 0.01, key="notears_frac")
        notears_edge_q = st.slider("NOTEARS edge quantile (Q)", 0.50, 0.99, 0.90, 0.01, key="notears_q")
        notears_freq_thr = st.slider("NOTEARS freq threshold", 0.00, 1.00, 0.60, 0.01, key="notears_thr")

    with colC:
        st.markdown("**GES / LiNGAM / PC**")
        ges_boot_runs = st.number_input("GES bootstraps", min_value=1, value=30, step=1, key="ges_k")
        ges_boot_frac = st.slider("GES sample fraction", 0.10, 1.00, 0.70, 0.01, key="ges_frac")
        ges_stable_thr = st.slider("GES stable edge freq", 0.00, 1.00, 0.40, 0.01, key="ges_thr")

        lingam_boot_runs = st.number_input("LiNGAM bootstraps", min_value=1, value=30, step=1, key="lingam_k")
        lingam_freq_thr = st.slider("LiNGAM freq threshold", 0.00, 1.00, 0.10, 0.01, key="lingam_thr")

        pc_boot_runs = st.number_input("PC bootstraps", min_value=1, value=30, step=1, key="pc_k")
        pc_alpha = st.slider("PC alpha", 0.00, 1.00, 0.30, 0.01, key="pc_alpha")
        pc_freq_thr = st.slider("PC freq threshold", 0.00, 1.00, 0.30, 0.01, key="pc_thr")

    st.markdown("### DAG-GNN (Notebook Defaults)")
    dag_col1, dag_col2, dag_col3 = st.columns(3)

    with dag_col1:
        daggnn_enabled = st.checkbox("Enable DAG-GNN", value=True, key="dag_enabled")
        daggnn_boot_runs = st.number_input("DAG-GNN bootstraps", min_value=1, value=30, step=1, key="dag_k")

    with dag_col2:
        daggnn_edge_abs_thr = st.slider("DAG-GNN edge abs threshold", 0.0, 0.2, 0.01, 0.005, key="dag_abs")
        daggnn_freq_thr = st.slider("DAG-GNN freq threshold", 0.0, 1.0, 0.20, 0.01, key="dag_thr")

    with dag_col3:
        daggnn_iters = st.number_input("DAG-GNN iters", min_value=10, value=500, step=10, key="dag_iters")
        daggnn_hidden_dim = st.number_input("DAG-GNN hidden dim", min_value=8, value=128, step=8, key="dag_hdim")

    st.markdown("---")
    st.subheader("Display Options")
    plot = st.checkbox("Show interactive matplotlib windows (NOT recommended in Streamlit)", value=False, key="plot_ui")
    save_plots = st.checkbox("Save PNGs and show in Streamlit", value=True, key="save_plots")

    st.markdown("---")
    st.subheader("PC Domain Filters (optional)")
    exclude_vars_text = st.text_area("Exclude variables (comma-separated)", value="", key="exclude_vars_text")
    invalid_edges_text = st.text_area("Invalid directed edges (one per line as: source,target)", value="", key="invalid_edges_text")
    domain_exclude_vars = parse_exclude_vars(exclude_vars_text)
    domain_invalid_edges = parse_invalid_edges(invalid_edges_text)

    st.markdown("---")
    if st.button("Run Causal Discovery", type="primary", key="btn_run_cd"):
        cfg = DiscoveryConfig(
            out_dir=out_dir,
            seed=int(seed),

            notears_k_runs=int(notears_k_runs),
            notears_sample_frac=float(notears_sample_frac),
            notears_edge_q=float(notears_edge_q),
            notears_freq_thr=float(notears_freq_thr),

            ges_boot_runs=int(ges_boot_runs),
            ges_boot_frac=float(ges_boot_frac),
            ges_stable_thr=float(ges_stable_thr),
            ges_keep_continuous_only=True,

            lingam_boot_runs=int(lingam_boot_runs),
            lingam_freq_thr=float(lingam_freq_thr),

            daggnn_enabled=bool(daggnn_enabled),
            daggnn_boot_runs=int(daggnn_boot_runs),
            daggnn_edge_abs_thr=float(daggnn_edge_abs_thr),
            daggnn_freq_thr=float(daggnn_freq_thr),
            daggnn_iters=int(daggnn_iters),
            daggnn_hidden_dim=int(daggnn_hidden_dim),

            pc_boot_runs=int(pc_boot_runs),
            pc_alpha=float(pc_alpha),
            pc_freq_thr=float(pc_freq_thr),

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

        st.session_state["cd_outputs"] = outputs
        st.session_state["df_clean"] = df_clean
        st.session_state["var_types_clean"] = var_types_clean

        st.session_state["llm_payload"] = None
        st.session_state["llm_payload_sig"] = None
        st.session_state["llm_work_dir"] = None
        st.session_state["run_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state["qa_messages"] = None

    outputs = st.session_state.get("cd_outputs")
    if outputs is None:
        st.info("No causal discovery results yet. Click **Run Causal Discovery** above.")
    else:
        st.markdown("---")
        st.subheader("Results Summary")

        st.write("**Output directory:**", outputs.get("out_dir"))
        st.write("**Numeric cols used:**", outputs.get("numeric_cols"))

        runs = outputs.get("runs", {})

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
# TAB 2 — LLM Root Cause Report
# =========================================
with tab2:
    st.markdown("---")
    st.subheader("LLM Root Cause Report (API + Validator)")

    if st.session_state.get("cd_outputs") is None:
        st.info("Run **Causal Discovery** first (Tab 1).")
        st.stop()

    outputs = st.session_state["cd_outputs"]
    df_clean_ss = st.session_state["df_clean"]

    preferred_targets = ["Tool wear", "Torque", "Rotational speed", "Process temperature", "Air temperature"]
    default_target = next((t for t in preferred_targets if t in df_clean_ss.columns), df_clean_ss.columns[-1])

    target = st.selectbox(
        "Target variable (KPI to explain)",
        options=list(df_clean_ss.columns),
        index=list(df_clean_ss.columns).index(default_target),
        key="tab2_target",
    )

    st.markdown("### Select an incident")
    incident_mode = st.radio(
        "Choose incident by:",
        ["Row index", "Top anomalies in target"],
        horizontal=True,
        key="tab2_incident_mode",
    )

    if incident_mode == "Row index":
        incident_index = st.number_input(
            "Incident index (row number)",
            min_value=0,
            max_value=len(df_clean_ss) - 1,
            value=0,
            step=1,
            key="tab2_incident_row",
        )
    else:
        x = df_clean_ss[target].astype(float)
        mu = float(x.mean(skipna=True))
        sd = float(x.std(skipna=True))
        if not (sd > 1e-12):
            sd = 1.0
        z_abs = ((x - mu) / sd).abs()
        top_idx = z_abs.sort_values(ascending=False).head(50).index.tolist()
        incident_index = st.selectbox("Select from top 50 anomalies", options=top_idx, key="tab2_incident_anom")
        st.write("Anomaly z-score:", float((x.loc[incident_index] - mu) / sd))

    st.markdown("### Consensus settings (deterministic)")
    c1, c2, c3 = st.columns(3)
    with c1:
        min_alg_count = st.number_input("Min algorithm agreement (edges)", min_value=1, max_value=5, value=2, step=1, key="tab2_min_alg")
    with c2:
        min_support_mean = st.slider("Min mean support", 0.0, 1.0, 0.30, 0.01, key="tab2_min_support")
    with c3:
        max_path_len = st.number_input("Max path length", min_value=2, max_value=8, value=4, step=1, key="tab2_max_path")

    # Stable work dir per CD run
    if st.session_state.get("llm_work_dir") is None:
        st.session_state["llm_work_dir"] = os.path.join(
            outputs.get("out_dir", "outputs_step2"),
            f"llm_phase_{st.session_state['run_id']}",
        )

    work_dir = st.text_input(
        "Work directory for LLM phase outputs",
        value=st.session_state["llm_work_dir"],
        key="tab2_work_dir",
    )
    st.session_state["llm_work_dir"] = work_dir

    # Payload signature guard
    current_sig = {
        "target": target,
        "incident_index": int(incident_index),
        "min_alg_count": int(min_alg_count),
        "min_support_mean": float(min_support_mean),
        "max_path_len": int(max_path_len),
        "work_dir": work_dir,
    }
    prev_sig = st.session_state.get("llm_payload_sig")
    if st.session_state.get("llm_payload") is not None and prev_sig != current_sig:
        st.session_state["llm_payload"] = None
        st.session_state["llm_payload_sig"] = None
        st.warning("Selections changed since last payload build. Please rebuild consensus + ranking.")
        st.stop()

    if st.button("Build consensus + ranking + evidence", type="primary", key="btn_build_payload"):
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
        st.session_state["llm_payload_sig"] = current_sig
        st.success("LLM-prep completed. Candidate ranking + evidence ready.")

    if st.session_state.get("llm_payload") is None:
        st.info("Click **Build consensus + ranking + evidence** to prepare LLM inputs.")
        st.stop()

    payload = st.session_state["llm_payload"]

    # Load candidates
    cand_df = None
    cand_path = payload.get("ranking", {}).get("candidates_path")
    if cand_path and os.path.exists(cand_path):
        cand_df = pd.read_csv(cand_path)

    st.markdown("## Ranked candidates")
    if cand_df is None:
        st.error("Candidates file missing. Rebuild payload.")
        st.stop()
    if cand_df.empty:
        st.error("No candidates found under current thresholds. Loosen thresholds or change target.")
        st.stop()

    st.dataframe(cand_df.head(15), use_container_width=True)

    st.markdown("## Incident evidence (top candidates)")
    st.json(payload.get("incident_evidence", {}))

    st.markdown("## Consensus outputs")
    _show_file_links({
        "consensus_edges": payload["consensus"]["consensus_edges_path"],
        "consensus_graph": payload["consensus"]["consensus_graph_path"],
        "consensus_summary": payload["consensus"]["consensus_summary_path"],
        "candidates": payload["ranking"]["candidates_path"],
        "top_paths": payload["ranking"]["top_paths_path"],
    })

    # Build prompt
    st.markdown("## Prompt (used for API call)")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    prompt_template_path = os.path.join(BASE_DIR, "llm_phase", "prompt_template.txt")

    if not os.path.exists(prompt_template_path):
        st.error(f"Prompt template not found at: {prompt_template_path}")
        st.stop()

    prompt = build_prompt(
        prompt_template_path=prompt_template_path,
        payload=payload,
        target=target,
        incident_index=int(incident_index),
    )
    st.text_area("Prompt (read-only)", value=prompt, height=320, key="tab2_prompt_view")

    # LLM API mode
    st.markdown("## Generate report via API (no manual copy/paste)")

    model_choice = st.selectbox(
        "LLM model",
        options=["gpt-5-mini", "gpt-5.2"],
        index=0,
        key="tab2_model_choice",
    )

    if st.button("Run LLM via API + Validate", type="primary", key="btn_run_llm_api"):
        allowed_vars = set(payload.get("allowed_vars", []))
        allowed_edges = set((a, b) for a, b in payload.get("allowed_edges", []))

        try:
            with st.spinner("Calling LLM API..."):
                report_json = generate_root_cause_report_json(prompt=prompt, model=model_choice)

            parsed_valid = run_llm_and_validate(
                call_llm_fn=lambda _prompt: json.dumps(report_json),
                prompt=prompt,
                allowed_vars=allowed_vars,
                allowed_edges=allowed_edges,
            )

            st.success("LLM output is valid and compliant.")
            st.json(parsed_valid)

        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"LLM API / validation failed: {e}")


# =========================================
# TAB 3 — LLM Q&A (Grounded) — Live Chat (API)
# =========================================
with tab3:
    st.markdown("---")
    st.subheader("LLM Q&A (Grounded) — Live Chat (API)")

    if st.session_state.get("cd_outputs") is None:
        st.info("Run **Causal Discovery** first (Tab 1).")
        st.stop()

    outputs = st.session_state["cd_outputs"]
    var_types_ss = st.session_state.get("var_types_clean")
    llm_payload = st.session_state.get("llm_payload")

    model_choice = st.selectbox(
        "Chat model",
        options=["gpt-5-mini", "gpt-5.2"],
        index=0,
        key="tab3_model_choice",
    )

    context_pack = build_llm_context_pack(outputs, var_types_ss, llm_payload)

    if st.session_state["qa_messages"] is None:
        st.session_state["qa_messages"] = [{"role": "system", "content": context_pack}]
    else:
        st.session_state["qa_messages"][0] = {"role": "system", "content": context_pack}

    if st.button("Reset chat", key="tab3_reset_chat"):
        st.session_state["qa_messages"] = [{"role": "system", "content": context_pack}]
        st.rerun()

    for msg in st.session_state["qa_messages"][1:]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_text = st.chat_input("Ask about the graphs / candidates / evidence...", key="tab3_chat_input")

    if user_text:
        st.session_state["qa_messages"].append({"role": "user", "content": user_text})

        with st.chat_message("user"):
            st.write(user_text)

        with st.chat_message("assistant"):
            with st.spinner("Calling LLM API..."):
                try:
                    answer = chat_grounded(
                        messages=st.session_state["qa_messages"],
                        model=model_choice,
                        verbosity="medium",
                    )
                except RuntimeError as e:
                    answer = str(e)
                except Exception as e:
                    answer = f"API error: {e}"

            st.write(answer)

        st.session_state["qa_messages"].append({"role": "assistant", "content": answer})
