# Streamlit_app.py
import os
import streamlit as st
import pandas as pd

from preprocessing import run_preprocessing
from causal_discovery import run_causal_discovery, DiscoveryConfig

st.set_page_config(page_title="Causal Discovery App", layout="wide")
st.title("Causal Discovery App")

# -----------------------------
# Helpers
# -----------------------------
def _safe_read_csv(path: str) -> pd.DataFrame | None:
    try:
        if path and os.path.exists(path) and path.lower().endswith(".csv"):
            return pd.read_csv(path)
    except Exception:
        return None
    return None

def _show_file_links(paths: dict):
    # Render download buttons for any existing files in a dict.
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


# -----------------------------
# Upload
# -----------------------------
file = st.file_uploader("Upload your CSV file", type=["csv"])

if file is None:
    st.stop()

# -----------------------------
# Read + show raw
# -----------------------------
try:
    df_raw = pd.read_csv(file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.subheader("Raw Data (preview)")
st.dataframe(df_raw.head(50), use_container_width=True)

# -----------------------------
# Preprocessing
# -----------------------------
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

# -----------------------------
# Causal Discovery Config UI
# -----------------------------
st.markdown("---")
st.subheader("Causal Discovery Settings (Notebook Defaults)")

colA, colB, colC = st.columns(3)

with colA:
    out_dir = st.text_input("Output directory", value="outputs_step2")
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

with colB:
    # NOTEARS
    st.markdown("**NOTEARS**")
    notears_k_runs = st.number_input("NOTEARS bootstraps (K)", min_value=1, value=30, step=1)
    notears_sample_frac = st.slider("NOTEARS sample fraction", 0.10, 1.00, 0.65, 0.01)
    notears_edge_q = st.slider("NOTEARS edge quantile (Q)", 0.50, 0.99, 0.90, 0.01)
    notears_freq_thr = st.slider("NOTEARS freq threshold", 0.00, 1.00, 0.60, 0.01)

with colC:
    # GES / LiNGAM / PC
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

# Streamlit-friendly plotting choices:
# - plot=False to avoid plt.show()
# - save_plots=True to save PNGs and display them with st.image
st.markdown("---")
st.subheader("Display Options")
plot = st.checkbox("Show interactive matplotlib windows (NOT recommended in Streamlit)", value=False)
save_plots = st.checkbox("Save PNGs and show in Streamlit", value=True)

# PC domain filters
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

domain_exclude_vars = parse_exclude_vars(exclude_vars_text)
domain_invalid_edges = parse_invalid_edges(invalid_edges_text)

# -----------------------------
# Run button
# -----------------------------
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

    # -----------------------------
    # Show outputs (cleanly)
    # -----------------------------
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
                # In your DAG-GNN code, PNG is only saved in that plotting block.
                # If save_plots is enabled, this should exist.
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
