# causal_discovery.py
from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Iterable

import numpy as np
import pandas as pd
import networkx as nx

from joblib import Parallel, delayed

# Algorithms
from causalnex.structure.notears import from_pandas
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from lingam import DirectLiNGAM

# Optional: torch only if DAG-GNN is enabled
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


@dataclass
class DiscoveryConfig:
    out_dir: str = "outputs_step2"
    seed: int = 42

    # NOTEARS
    notears_k_runs: int = 30
    notears_sample_frac: float = 0.65
    notears_edge_q: float = 0.9
    notears_freq_thr: float = 0.6
    notears_cap_factor: float = 2.0
    notears_min_factor: float = 0.5
    notears_bootstrap: bool = True
    notears_n_jobs: int = 4

    # GES
    ges_boot_runs: int = 30
    ges_boot_frac: float = 0.70
    ges_stable_thr: float = 0.40  # appeared in >= 40% of runs
    ges_keep_continuous_only: bool = True

    # LiNGAM
    lingam_boot_runs: int = 30
    lingam_freq_thr: float = 0.10  # keep edges with freq >= 0.10

    # DAG-GNN
    daggnn_enabled: bool = True
    daggnn_boot_runs: int = 30
    daggnn_edge_abs_thr: float = 0.01
    daggnn_freq_thr: float = 0.20
    daggnn_lr: float = 0.01
    daggnn_iters: int = 500
    daggnn_lam: float = 1e-4
    daggnn_hidden_dim: int = 128

    # PC
    pc_boot_runs: int = 30
    pc_sample_frac: float = 1.0  # notebook samples full-size with replacement
    pc_alpha: float = 0.30
    pc_freq_thr: float = 0.30

    # Plotting
    plot: bool = True          # notebook shows plots
    save_plots: bool = False   # set True for headless environments


def _ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def detect_type_series(s: pd.Series) -> str:
    """Mimics Step1-style coarse typing used in the notebook."""
    if pd.api.types.is_numeric_dtype(s):
        n_unique = s.nunique(dropna=True)
        if n_unique <= 2:
            return "binary"
        elif n_unique <= 15:
            return "discrete_numeric"
        else:
            return "continuous"
    return "categorical"


def build_numeric_df(
    df_clean: pd.DataFrame,
    var_types: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    Replicates notebook cleaning for causal discovery:

    - Choose numeric columns by var_types if provided; otherwise auto-detect.
    - Replace inf with NaN
    - Drop all-NaN columns
    - Drop zero-variance columns
    """
    if var_types is None:
        var_types = {c: detect_type_series(df_clean[c]) for c in df_clean.columns}

    numeric_cols = [
        c for c, t in var_types.items()
        if c in df_clean.columns and t in ("continuous", "discrete_numeric", "binary")
    ]

    df_num = df_clean[numeric_cols].copy()
    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    df_num = df_num.dropna(axis=1, how="all")

    zero_var_cols = [c for c in df_num.columns if df_num[c].nunique(dropna=True) <= 1]
    if zero_var_cols:
        df_num = df_num.drop(columns=zero_var_cols)

    if df_num.shape[1] < 2:
        raise ValueError("Not enough usable numeric variables for causal discovery.")

    return df_num, list(df_num.columns), var_types


def classify_variable_roles_basic(df: pd.DataFrame) -> Dict[str, str]:
    """
    Notebook utility version (used by NOTEARS / DAG-GNN / PC):
    - binary if nunique <= 2
    - categorical if 2 < nunique <= 10
    - continuous if numeric dtype
    - other otherwise
    """
    roles: Dict[str, str] = {}
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        dtype = df[col].dtype

        if nunique <= 2:
            roles[col] = "binary"
        elif 2 < nunique <= 10:
            roles[col] = "categorical"
        elif np.issubdtype(dtype, np.number):
            roles[col] = "continuous"
        else:
            roles[col] = "other"
    return roles


def classify_variable_roles_ratio(df: pd.DataFrame) -> Dict[str, str]:
    """
    GES-cell version (ratio-based).
    """
    roles: Dict[str, str] = {}
    n = len(df)
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        ratio = (nunique / n) if n > 0 else 0.0

        if nunique <= 2 or ratio < 0.02:
            roles[col] = "binary_outcome"
        elif nunique <= 10 or ratio < 0.05:
            roles[col] = "categorical"
        elif pd.api.types.is_numeric_dtype(df[col]):
            roles[col] = "continuous"
        else:
            roles[col] = "unknown"
    return roles


def choose_discovery_vars(roles: Dict[str, str], keep_continuous_only: bool = True) -> List[str]:
    if keep_continuous_only:
        return [c for c, r in roles.items() if r == "continuous"]
    return list(roles.keys())


def adj_from_nx(nodes: Iterable[str], g: nx.DiGraph) -> pd.DataFrame:
    nodes = list(nodes)
    node_set = set(nodes)
    A = pd.DataFrame(0, index=nodes, columns=nodes, dtype=np.int8)
    for u, v in g.edges():
        if u in node_set and v in node_set:
            A.loc[u, v] = 1
    return A


def make_dag_by_score(g_in: nx.DiGraph, score_attr: str = "score", default_score: float = 0.0) -> nx.DiGraph:
    """
    Enforce DAG by repeatedly removing the weakest edge (by score_attr) from detected cycles.
    Matches NOTEARS helper in notebook.
    """
    g = nx.DiGraph()
    g.add_nodes_from(g_in.nodes())
    for u, v, d in g_in.edges(data=True):
        s = float(d.get(score_attr, default_score))
        g.add_edge(u, v, **{**d, score_attr: s})

    while not nx.is_directed_acyclic_graph(g):
        try:
            cycle = nx.find_cycle(g, orientation="original")
            weakest = min(
                cycle,
                key=lambda x: float(g.edges[x[0], x[1]].get(score_attr, default_score)),
            )
            g.remove_edge(*weakest[:2])
        except Exception:
            break
    return g


def _save_graph_plot(
    G: nx.DiGraph,
    title: str,
    out_png: Optional[str],
    node_color: str = "skyblue",
    edge_label_attr: Optional[str] = None,
    edge_width_attr: Optional[str] = None,
    edge_color_fn=None,
    show: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 7))
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=10)

    edges = list(G.edges())

    if edge_color_fn is None:
        edge_colors = ["green" for _ in edges]
    else:
        edge_colors = [edge_color_fn(u, v, G.edges[u, v]) for (u, v) in edges]

    if edge_width_attr is None:
        edge_widths = [2.0 for _ in edges]
    else:
        edge_widths = [2.5 * float(G.edges[u, v].get(edge_width_attr, 1.0)) for (u, v) in edges]

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrowsize=20)

    if edge_label_attr is not None:
        edge_labels = {(u, v): f"{float(G.edges[u, v].get(edge_label_attr, 0.0)):.2f}" for (u, v) in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if out_png is not None:
        plt.savefig(out_png, dpi=300)
    if show:
        plt.show()
    plt.close()


def run_notears_bootstrapped(
    df_num: pd.DataFrame,
    cfg: DiscoveryConfig,
    results_graph: Dict[str, nx.DiGraph],
    results_adj: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    ALG_NAME = "NOTEARS_BOOTSTRAPPED"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    alg_dir = _ensure_out_dir(os.path.join(cfg.out_dir, f"{ALG_NAME}_{run_tag}"))

    # parameters (match notebook)
    K_RUNS = int(cfg.notears_k_runs)
    SAMPLE_FRAC = float(cfg.notears_sample_frac)
    EDGE_Q = float(cfg.notears_edge_q)
    FREQ_THR = float(cfg.notears_freq_thr)
    CAP_FACTOR = float(cfg.notears_cap_factor)
    MIN_FACTOR = float(cfg.notears_min_factor)
    BOOTSTRAP = bool(cfg.notears_bootstrap)
    SEED = int(cfg.seed)
    N_JOBS = int(cfg.notears_n_jobs)

    print(f"\nRunning {ALG_NAME}")
    print("ALG_DIR:", alg_dir)
    print(f"{ALG_NAME} | K_RUNS={K_RUNS} | SAMPLE_FRAC={SAMPLE_FRAC} | FREQ_THR={FREQ_THR}")

    # role detection like notebook
    roles = {}
    for col in df_num.columns:
        nunique = df_num[col].nunique(dropna=True)
        dtype = df_num[col].dtype
        if nunique <= 2:
            roles[col] = "binary"
        elif 2 < nunique <= 10:
            roles[col] = "categorical"
        elif np.issubdtype(dtype, np.number):
            roles[col] = "continuous"
        else:
            roles[col] = "other"
    subset_cols = [c for c, r in roles.items() if r == "continuous"]

    if len(subset_cols) < 2:
        raise ValueError("NOTEARS skipped — not enough continuous variables (need ≥2).")

    df_notears = df_num[subset_cols].copy()
    X = df_notears.values
    n, p = X.shape

    def one_run(k_idx: int):
        rng = np.random.default_rng(SEED + k_idx)
        idx = rng.choice(n, int(n * SAMPLE_FRAC), replace=BOOTSTRAP)
        df_k = pd.DataFrame(X[idx], columns=subset_cols)

        sm = from_pandas(df_k)
        edges = list(sm.edges(data=True))
        weights = np.array([abs(float(e[2].get("weight", 0.0))) for e in edges])

        if len(weights) == 0:
            return np.zeros((p, p), dtype=np.int8), {}, 0

        thr = np.quantile(weights, EDGE_Q)
        max_edges = int(CAP_FACTOR * p)
        min_edges = int(MIN_FACTOR * p)

        top = [e for e in edges if abs(float(e[2].get("weight", 0.0))) >= thr]
        top = sorted(top, key=lambda e: abs(float(e[2].get("weight", 0.0))), reverse=True)
        top = top[: max(min_edges, min(len(top), max_edges))]

        g = nx.DiGraph()
        g.add_nodes_from(subset_cols)
        for u, v, d in top:
            w = float(d.get("weight", 0.0))
            g.add_edge(u, v, weight=w)

        A = adj_from_nx(subset_cols, g).values.astype(np.int8)
        wmap = {(u, v): float(g.edges[u, v].get("weight", 0.0)) for u, v in g.edges()}
        return A, wmap, int(A.sum())

    t0 = time.time()
    outs = Parallel(n_jobs=N_JOBS)(delayed(one_run)(k) for k in range(K_RUNS))
    print(f"Completed {K_RUNS} NOTEARS runs in {time.time() - t0:.1f} sec")

    # aggregate
    A_list = [o[0] for o in outs]
    w_maps = [o[1] for o in outs]

    freq = np.mean(np.stack(A_list), axis=0)
    edge_freq = pd.DataFrame(freq, index=subset_cols, columns=subset_cols)
    edge_freq_path = os.path.join(alg_dir, "edge_freq_after_prune.csv")
    edge_freq.to_csv(edge_freq_path)

    weights_by_edge: Dict[Tuple[str, str], List[float]] = {}
    for wm in w_maps:
        for e, w in wm.items():
            weights_by_edge.setdefault(e, []).append(w)

    rows = []
    for u in subset_cols:
        for v in subset_cols:
            if u == v:
                continue
            f = float(edge_freq.loc[u, v])
            if f > 0:
                ws = weights_by_edge.get((u, v), [])
                rows.append({
                    "source": u,
                    "target": v,
                    "freq": f,
                    "weight_median": float(np.median(ws)) if ws else 0.0,
                    "abs_weight_median": float(np.median(np.abs(ws))) if ws else 0.0,
                })

    edge_stats = pd.DataFrame(rows).sort_values(["freq", "abs_weight_median"], ascending=[False, False])
    edge_stats_path = os.path.join(alg_dir, "edge_stats_freq_weight_NOTEARS.csv")
    edge_stats.to_csv(edge_stats_path, index=False)

    print("\n================ NOTEARS Summary ================")
    print(f"Total candidate edges: {len(edge_stats)}")
    print(f"K_RUNS={K_RUNS} | FREQ_THR={FREQ_THR} | EDGE_Q={EDGE_Q}\n")
    print("Top 10 edges by freq and |median|:\n")
    if not edge_stats.empty:
        print(edge_stats.head(10).to_string(index=False))
    print("\n=================================================\n")

    keep = edge_stats[edge_stats["freq"] >= FREQ_THR]
    print(f"Edges kept by freq >= {FREQ_THR}: {len(keep)}")

    g_cons = nx.DiGraph()
    g_cons.add_nodes_from(subset_cols)
    for _, r in keep.iterrows():
        score = float(r["freq"]) + 1e-6 * float(r["abs_weight_median"])
        g_cons.add_edge(
            r["source"], r["target"],
            freq=float(r["freq"]),
            weight_median=float(r["weight_median"]),
            abs_weight_median=float(r["abs_weight_median"]),
            score=float(score),
        )

    g_final = make_dag_by_score(g_cons, score_attr="score")
    A_final = adj_from_nx(subset_cols, g_final)

    out_adj = os.path.join(alg_dir, "FINAL_adj_NOTEARS.csv")
    out_edges = os.path.join(alg_dir, "FINAL_edges_NOTEARS.csv")
    out_graph = os.path.join(alg_dir, "FINAL_graph_NOTEARS.graphml")
    out_json = os.path.join(alg_dir, "FINAL_summary_NOTEARS.json")

    A_final.to_csv(out_adj)
    nx.write_graphml(g_final, out_graph)

    pd.DataFrame([{
        "source": u, "target": v,
        "freq": float(d.get("freq", 0.0)),
        "weight_median": float(d.get("weight_median", 0.0)),
    } for u, v, d in g_final.edges(data=True)]).to_csv(out_edges, index=False)

    with open(out_json, "w") as f:
        json.dump({
            "algorithm": ALG_NAME,
            "K_RUNS": K_RUNS,
            "SAMPLE_FRAC": SAMPLE_FRAC,
            "EDGE_Q": EDGE_Q,
            "FREQ_THR": FREQ_THR,
            "BOOTSTRAP": BOOTSTRAP,
            "final_edge_count": int(g_final.number_of_edges()),
        }, f, indent=2)

    if cfg.plot or cfg.save_plots:
        out_png = os.path.join(alg_dir, "FINAL_graph_NOTEARS.png") if cfg.save_plots else None

        def edge_color_fn(u, v, d):
            return "green" if float(d.get("weight_median", 0.0)) > 0 else "red"

        _save_graph_plot(
            g_final,
            title="Final NOTEARS DAG (Bootstrapped)",
            out_png=out_png,
            node_color="skyblue",
            edge_label_attr="weight_median",
            edge_width_attr="freq",
            edge_color_fn=edge_color_fn,
            show=cfg.plot,
        )

    results_graph[ALG_NAME] = g_final
    results_adj[ALG_NAME] = A_final

    print("Saved outputs:\n", out_adj, "\n", out_edges, "\n", out_graph, "\n", out_json)

    return {
        "alg_dir": alg_dir,
        "adj_path": out_adj,
        "edges_path": out_edges,
        "graph_path": out_graph,
        "summary_path": out_json,
        "subset_cols": subset_cols,
    }


def run_ges_bootstrap(df_num: pd.DataFrame, cfg: DiscoveryConfig) -> Dict[str, Any]:
    """
    Matches notebook behavior:
      - uses continuous variables only
      - bootstraps ges() and collects edge frequencies + median scores
      - saves STABLE_edges_GES.csv
      - plots stable graph
    Note: notebook does not register GES into results_graph/results_adj.
    """
    ALG_NAME = "GES_Bootstrap"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    alg_dir = _ensure_out_dir(os.path.join(cfg.out_dir, f"{ALG_NAME}_{run_tag}"))

    roles = classify_variable_roles_ratio(df_num)
    subset_cols = choose_discovery_vars(roles, keep_continuous_only=cfg.ges_keep_continuous_only)

    if len(subset_cols) < 2:
        print("GES Bootstrap skipped — not enough continuous variables (need ≥2).")
        return {"alg_dir": alg_dir, "skipped": True}

    df_ges = df_num[subset_cols].copy()

    BOOT_RUNS = int(cfg.ges_boot_runs)
    BOOT_FRAC = float(cfg.ges_boot_frac)
    STABLE_THR = float(cfg.ges_stable_thr)

    print(f"\nRunning {ALG_NAME}")
    print(f"Bootstrap Config: RUNS={BOOT_RUNS}, FRAC={BOOT_FRAC}, SEED={cfg.seed}")

    edge_counter: Dict[Tuple[str, str], int] = {}
    edge_scores: Dict[Tuple[str, str], List[float]] = {}

    X_data = df_ges.to_numpy()
    n_samples = len(X_data)
    var_names = list(df_ges.columns)

    for i in range(BOOT_RUNS):
        sample_idx = np.random.choice(n_samples, int(BOOT_FRAC * n_samples), replace=True)
        X_boot = X_data[sample_idx]
        try:
            result = ges(X_boot)
            G = result["G"]
            edges = list(G.graph.keys())
            for u_idx, v_idx in edges:
                src, tgt = var_names[u_idx], var_names[v_idx]
                score = float(G.graph[(u_idx, v_idx)])
                edge_counter[(src, tgt)] = edge_counter.get((src, tgt), 0) + 1
                edge_scores.setdefault((src, tgt), []).append(score)
        except Exception as e:
            print(f"Bootstrap {i+1} failed:", str(e))

    stable_edges = []
    for (src, tgt), cnt in edge_counter.items():
        rel_freq = cnt / BOOT_RUNS
        if rel_freq >= STABLE_THR:
            scores = edge_scores.get((src, tgt), [])
            median_score = float(np.median(scores)) if scores else 0.0
            stable_edges.append({
                "source": src,
                "target": tgt,
                "freq": round(rel_freq, 2),
                "median_score": round(median_score, 5),
            })

    stable_df = pd.DataFrame(stable_edges).sort_values("freq", ascending=False)
    edges_out = os.path.join(alg_dir, "STABLE_edges_GES.csv")
    stable_df.to_csv(edges_out, index=False)

    print(f"GES Bootstrapping complete. Stable edges saved to: {edges_out}")
    if not stable_df.empty:
        print("Top edges:\n", stable_df.head(10).to_string(index=False))
    else:
        print("No stable edges found (threshold too high or data too noisy).")

    # Plot stable graph (notebook)
    if (cfg.plot or cfg.save_plots) and (not stable_df.empty):
        G_stable = nx.DiGraph()
        for _, row in stable_df.iterrows():
            G_stable.add_edge(row["source"], row["target"], score=float(row["median_score"]), freq=float(row["freq"]))

        out_png = os.path.join(alg_dir, "STABLE_graph_GES.png") if cfg.save_plots else None

        _save_graph_plot(
            G_stable,
            title="Stable Causal DAG (GES + Bootstrapping)",
            out_png=out_png,
            node_color="lightblue",
            edge_label_attr="score",
            edge_width_attr="freq",
            edge_color_fn=lambda u, v, d: "slateblue",
            show=cfg.plot,
        )

    return {
        "alg_dir": alg_dir,
        "stable_edges_path": edges_out,
        "subset_cols": subset_cols,
        "stable_edges": stable_df,
    }


def run_lingam_bootstrap(
    df_num: pd.DataFrame,
    cfg: DiscoveryConfig,
    results_graph: Dict[str, nx.DiGraph],
    results_adj: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    ALG_NAME = "DirectLiNGAM_Bootstrap"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    alg_dir = _ensure_out_dir(os.path.join(cfg.out_dir, f"{ALG_NAME}_{run_tag}"))

    print(f"\nRunning {ALG_NAME}...\nOutput directory: {alg_dir}")

    roles = {}
    for col in df_num.columns:
        nunique = df_num[col].nunique(dropna=True)
        dtype = df_num[col].dtype
        if nunique <= 2:
            roles[col] = "binary"
        elif 2 < nunique <= 10:
            roles[col] = "categorical"
        elif np.issubdtype(dtype, np.number):
            roles[col] = "continuous"
        else:
            roles[col] = "other"

    continuous_cols = [col for col, role in roles.items() if role == "continuous"]

    print(" Variable Roles:")
    for col, role in roles.items():
        print(f"  {col}: {role}")

    if len(continuous_cols) < 2:
        print(" Not enough continuous variables (≥2 required). Skipping LiNGAM.")
        return {"alg_dir": alg_dir, "skipped": True}

    df_lingam = df_num[continuous_cols].copy()
    cols = df_lingam.columns.tolist()
    X_full = df_lingam.values

    N_BOOT = int(cfg.lingam_boot_runs)
    FREQ_THRESHOLD = float(cfg.lingam_freq_thr)

    edge_records: Dict[Tuple[str, str], List[float]] = {}

    for i in range(N_BOOT):
        idx = np.random.choice(len(X_full), size=len(X_full), replace=True)
        model = DirectLiNGAM()
        model.fit(X_full[idx])
        B = model.adjacency_matrix_
        for ii, src in enumerate(cols):
            for jj, tgt in enumerate(cols):
                if abs(B[ii, jj]) > 1e-6:
                    edge_records.setdefault((src, tgt), []).append(float(B[ii, jj]))

    edge_list = []
    for (src, tgt), weights in edge_records.items():
        med = float(np.median(weights))
        edge_list.append({
            "source": src,
            "target": tgt,
            "freq": round(len(weights) / N_BOOT, 2),
            "weight_median": med,
            "abs_weight_median": abs(med),
        })

    edges_df = pd.DataFrame(edge_list)
    edges_df = edges_df[edges_df["freq"] >= FREQ_THRESHOLD]

    # Resolve bidirectional conflicts (notebook)
    edges_df = edges_df.sort_values("abs_weight_median", ascending=False)
    unique_edges: Dict[Tuple[str, str], Any] = {}
    for _, r in edges_df.iterrows():
        s = str(r["source"])
        t = str(r["target"])
        if (t, s) in unique_edges:
            existing = unique_edges[(t, s)]
            if float(r["freq"]) > float(existing["freq"]):
                del unique_edges[(t, s)]
                unique_edges[(s, t)] = r
        else:
            unique_edges[(s, t)] = r
    edges_df = pd.DataFrame(list(unique_edges.values()))

    # Build graph
    G = nx.DiGraph()
    G.add_nodes_from(cols)
    for _, row in edges_df.iterrows():
        G.add_edge(
            row["source"], row["target"],
            weight_median=float(row["weight_median"]),
            freq=float(row["freq"]),
        )

    A_df = adj_from_nx(cols, G)

    out_adj = os.path.join(alg_dir, "FINAL_adj_LiNGAM_Bootstrap.csv")
    out_edges = os.path.join(alg_dir, "FINAL_edges_LiNGAM_Bootstrap.csv")
    out_graph = os.path.join(alg_dir, "FINAL_graph_LiNGAM_Bootstrap.graphml")
    out_json = os.path.join(alg_dir, "FINAL_summary_LiNGAM_Bootstrap.json")

    A_df.to_csv(out_adj)
    edges_df.to_csv(out_edges, index=False)
    nx.write_graphml(G, out_graph)

    with open(out_json, "w") as f:
        json.dump({
            "algorithm": ALG_NAME,
            "bootstraps": N_BOOT,
            "n_edges": int(len(edges_df)),
            "n_vars": int(len(cols)),
            "vars_used": cols,
            "freq_threshold": FREQ_THRESHOLD,
        }, f, indent=2)

    results_graph[ALG_NAME] = G
    results_adj[ALG_NAME] = A_df

    print(f"\n{ALG_NAME} finished. Edges: {len(edges_df)} | Variables: {len(cols)}")
    if not edges_df.empty:
        print("Top edges by |median weight|:")
        for _, row in edges_df.sort_values("abs_weight_median", ascending=False).head(10).iterrows():
            print(f"  {row['source']} → {row['target']} | freq = {row['freq']} | median = {row['weight_median']:.4f}")

    if cfg.plot or cfg.save_plots:
        out_png = os.path.join(alg_dir, "FINAL_graph_LiNGAM_Bootstrap.png") if cfg.save_plots else None

        def edge_color_fn(u, v, d):
            return "green" if float(d.get("weight_median", 0.0)) > 0 else "red"

        _save_graph_plot(
            G,
            title="Final Causal DAG – LiNGAM (Bootstrapped)",
            out_png=out_png,
            node_color="skyblue",
            edge_label_attr="weight_median",
            edge_width_attr="freq",
            edge_color_fn=edge_color_fn,
            show=cfg.plot,
        )

    return {
        "alg_dir": alg_dir,
        "adj_path": out_adj,
        "edges_path": out_edges,
        "graph_path": out_graph,
        "summary_path": out_json,
        "subset_cols": cols,
    }

def run_daggnn_bootstrap(
    df_num: pd.DataFrame,
    cfg: DiscoveryConfig,
    results_graph: Dict[str, nx.DiGraph],
    results_adj: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    if not cfg.daggnn_enabled:
        return {"skipped": True, "reason": "disabled"}

    if torch is None or nn is None:
        print("DAG-GNN skipped: torch is not available in this environment.")
        return {"skipped": True, "reason": "torch_missing"}

    # NOTEBOOK NAME (Cell 8A/8C)
    ALG_NAME = "DAG_GNN_Bootstrap"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    alg_dir = _ensure_out_dir(os.path.join(cfg.out_dir, f"{ALG_NAME}_{run_tag}"))

    print(f"\nRunning {ALG_NAME}...\nOutput directory: {alg_dir}")

    # Notebook variable selection: classify -> choose_discovery_vars(roles)
    roles = classify_variable_roles_basic(df_num)
    subset_cols = choose_discovery_vars(roles, keep_continuous_only=True)

    if len(subset_cols) < 2:
        print(f"{ALG_NAME} skipped: Not enough continuous variables (need ≥2).")
    return {
        "alg_dir": alg_dir,
        "skipped": True,
        "reason": "not_enough_continuous_vars",
        "n_selected": len(subset_cols),
        "selected_vars": subset_cols,
        "roles": roles,
    }

    print(f"Bootstrapping DAG-GNN with stronger model (hidden_dim=128, iters=500, lam=0.0001)")

    from torch import optim

    N_BOOT = int(cfg.daggnn_boot_runs)          # notebook: 30
    X_full = df_num[subset_cols].copy().values.astype(np.float32)
    n, p = X_full.shape

    lr = float(cfg.daggnn_lr)                  # notebook: 0.01
    n_iters = int(cfg.daggnn_iters)            # notebook: 500
    lam = float(cfg.daggnn_lam)                # notebook: 0.0001
    EDGE_ABS_THR = float(cfg.daggnn_edge_abs_thr)  # notebook: 0.01
    FREQ_THR = float(cfg.daggnn_freq_thr)      # notebook: 0.2
    hidden_dim = int(cfg.daggnn_hidden_dim)    # notebook: 128

    edge_records: Dict[Tuple[str, str], List[float]] = {}

    def train_daggnn_boosted(X: np.ndarray, lr=0.01, n_iters=500, lam=0.0001) -> np.ndarray:
        # Notebook model: fc1/fc2 MLP only
        class DAGGNNModel(nn.Module):
            def __init__(self, d_in: int, hidden_dim: int = 128):
                super().__init__()
                self.fc1 = nn.Linear(d_in, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, d_in)

            def forward(self, X):
                h = torch.relu(self.fc1(X))
                return self.fc2(h)

        d = X.shape[1]
        model = DAGGNNModel(d, hidden_dim=hidden_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        X_tensor = torch.from_numpy(X).float()

        for _ in range(n_iters):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = torch.mean((X_tensor - pred) ** 2)
            loss += lam * sum(torch.abs(p).sum() for p in model.parameters())  # notebook L1 reg
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            # Notebook adjacency proxy
            W_est = model.fc2.weight.detach().cpu().numpy().T
        return W_est

    # Bootstrapping (notebook loop)
    for _ in range(N_BOOT):
        sample_idx = np.random.choice(n, size=n, replace=True)
        X_boot = X_full[sample_idx]
        W_est = train_daggnn_boosted(X_boot, lr=lr, n_iters=n_iters, lam=lam)

        for i, u in enumerate(subset_cols):
            for j, v in enumerate(subset_cols):
                if i == j:
                    continue
                w = float(W_est[i, j])
                if abs(w) >= EDGE_ABS_THR:
                    edge_records.setdefault((u, v), []).append(w)

    # Build edges_df
    edge_list = []
    for (u, v), weights in edge_records.items():
        w_med = float(np.median(weights))
        edge_list.append({
            "source": u,
            "target": v,
            "freq": round(len(weights) / N_BOOT, 2),
            "weight_mean": float(np.mean(weights)),
            "weight_median": w_med,
            "abs_weight_median": abs(w_med),
        })

    edges_df = pd.DataFrame(edge_list)
    if not edges_df.empty:
        edges_df = edges_df[edges_df["freq"] >= FREQ_THR]
        edges_df = edges_df.sort_values("abs_weight_median", ascending=False)

    # Graph
    G = nx.DiGraph()
    G.add_nodes_from(subset_cols)
    for _, row in edges_df.iterrows():
        G.add_edge(
            row["source"], row["target"],
            weight_median=float(row["weight_median"]),
            freq=float(row["freq"]),
        )

    A_df = adj_from_nx(subset_cols, G)

    # Save (notebook filenames)
    out_adj = os.path.join(alg_dir, "FINAL_adj_DAG-GNN_Bootstrap.csv")
    out_edges = os.path.join(alg_dir, "FINAL_edges_DAG-GNN_Bootstrap.csv")
    out_graph = os.path.join(alg_dir, "FINAL_graph_DAG-GNN_Bootstrap.graphml")
    out_json = os.path.join(alg_dir, "FINAL_summary_DAG-GNN_Bootstrap.json")

    A_df.to_csv(out_adj)
    edges_df.to_csv(out_edges, index=False)
    nx.write_graphml(G, out_graph)

    with open(out_json, "w") as f:
        json.dump({
            "algorithm": ALG_NAME,
            "bootstraps": N_BOOT,
            "n_edges": int(len(edges_df)),
            "n_vars": int(len(subset_cols)),
            "vars_used": subset_cols,
        }, f, indent=2)

    results_graph[ALG_NAME] = G
    results_adj[ALG_NAME] = A_df

    print(f" Bootstrapped DAG-GNN finished. Edges: {len(edges_df)} | Variables: {len(subset_cols)}")

    # Plot — notebook has a separate Cell 8C with min_weight_display=0.02
    if (cfg.plot or cfg.save_plots) and (len(edges_df) > 0):
        import matplotlib.pyplot as plt

        min_weight_display = 0.02
        edges_to_draw = [(u, v) for u, v, d in G.edges(data=True)
                         if abs(float(d.get("weight_median", 0.0))) >= min_weight_display]

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(12, 7))
        nx.draw_networkx_nodes(G, pos, node_color="lightgreen", node_size=800)
        nx.draw_networkx_labels(G, pos, font_size=10)

        edge_colors = ["green" if float(G[u][v]["weight_median"]) > 0 else "red" for u, v in edges_to_draw]
        edge_widths = [3 * abs(float(G[u][v]["weight_median"])) for u, v in edges_to_draw]
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, edge_color=edge_colors,
                               width=edge_widths, arrowsize=20)

        edge_labels = {(u, v): f'{float(G[u][v]["weight_median"]):.3f}' for u, v in edges_to_draw}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

        plt.title(f"Filtered Bootstrapped DAG ({ALG_NAME}) — Showing edges |median weight| ≥ {min_weight_display}")
        plt.axis("off")
        plt.tight_layout()

        if cfg.save_plots:
            plt.savefig(os.path.join(alg_dir, "FINAL_graph_DAG-GNN_Bootstrap.png"), dpi=300)

        if cfg.plot:
            plt.show()
        plt.close()

    return {
        "alg_dir": alg_dir,
        "adj_path": out_adj,
        "edges_path": out_edges,
        "graph_path": out_graph,
        "summary_path": out_json,
        "subset_cols": subset_cols,
    }




def run_pc_bootstrap(
    df_num: pd.DataFrame,
    cfg: DiscoveryConfig,
    results_graph: Dict[str, nx.DiGraph],
    results_adj: Dict[str, pd.DataFrame],
    domain_exclude_vars: Optional[set] = None,
    domain_invalid_edges: Optional[set] = None,
) -> Dict[str, Any]:
    ALG_NAME = "PC_Bootstrap"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    alg_dir = _ensure_out_dir(os.path.join(cfg.out_dir, f"{ALG_NAME}_{run_tag}"))

    roles = classify_variable_roles_basic(df_num)
    subset_cols = [col for col, role in roles.items() if role == "continuous"]

    if len(subset_cols) < 2:
        print(f"{ALG_NAME} skipped: Not enough continuous variables (need ≥2).")
        return {"alg_dir": alg_dir, "skipped": True}

    df_pc = df_num[subset_cols].copy()

    N_BOOT = int(cfg.pc_boot_runs)
    FREQ_THRESHOLD = float(cfg.pc_freq_thr)
    alpha = float(cfg.pc_alpha)

    DOMAIN_EXCLUDE_VARS = set() if domain_exclude_vars is None else set(domain_exclude_vars)
    DOMAIN_INVALID_EDGES = set() if domain_invalid_edges is None else set(domain_invalid_edges)

    cols = [c for c in subset_cols if c not in DOMAIN_EXCLUDE_VARS]
    if len(cols) < 2:
        print(f"{ALG_NAME} skipped after exclusions: <2 variables.")
        return {"alg_dir": alg_dir, "skipped": True}

    X_full = df_pc[cols].values
    edge_records: Dict[Tuple[str, str], int] = {}

    print(f"Bootstrapping {N_BOOT} runs for PC...")

    for b in range(N_BOOT):
        idx = np.random.choice(len(X_full), len(X_full), replace=True)
        X_boot = X_full[idx]

        cg = pc(data=X_boot, alpha=alpha, indep_test=fisherz, uc_rule=0, verbose=False)
        adj = cg.G.graph
        m = adj.shape[0]

        for i in range(m):
            for j in range(m):
                if adj[i, j] == 1:
                    u, v = cols[i], cols[j]
                    if (u, v) in DOMAIN_INVALID_EDGES:
                        continue
                    edge_records[(u, v)] = edge_records.get((u, v), 0) + 1

    edge_freq = {k: v / N_BOOT for k, v in edge_records.items()}

    # edge strength = Pearson correlation (notebook); avoid SciPy dependency
    edge_strength = {}
    for (u, v) in edge_freq.keys():
        try:
            corr = float(np.corrcoef(df_pc[u].values, df_pc[v].values)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            corr = 0.0
        edge_strength[(u, v)] = corr

    edges_df = pd.DataFrame([{
        "source": u,
        "target": v,
        "freq": float(edge_freq[(u, v)]),
        "corr": float(edge_strength[(u, v)]),
        "abs_corr": abs(float(edge_strength[(u, v)])),
    } for (u, v) in edge_freq.keys() if edge_freq[(u, v)] >= FREQ_THRESHOLD])

    if edges_df.empty:
        print("⚠ No stable edges above frequency threshold.")
    else:
        edges_df = edges_df.sort_values(["freq", "abs_corr"], ascending=[False, False])
        print(f"\nStable PC edges (freq ≥ {FREQ_THRESHOLD}): {len(edges_df)}")
        print(edges_df.head(10).to_string(index=False))

    G = nx.DiGraph()
    G.add_nodes_from(cols)
    for _, row in edges_df.iterrows():
        G.add_edge(row["source"], row["target"], freq=float(row["freq"]), corr=float(row["corr"]))

    A_df = adj_from_nx(cols, G)

    out_adj = os.path.join(alg_dir, "FINAL_adj_PC_Bootstrap.csv")
    out_edges = os.path.join(alg_dir, "FINAL_edges_PC_Bootstrap.csv")
    out_graph = os.path.join(alg_dir, "FINAL_graph_PC_Bootstrap.graphml")
    out_json = os.path.join(alg_dir, "FINAL_summary_PC_Bootstrap.json")

    A_df.to_csv(out_adj)
    edges_df.to_csv(out_edges, index=False)
    nx.write_graphml(G, out_graph)

    with open(out_json, "w") as f:
        json.dump({
            "algorithm": ALG_NAME,
            "bootstraps": N_BOOT,
            "n_edges": int(len(edges_df)),
            "n_vars": int(len(cols)),
            "vars_used": cols,
            "freq_threshold": FREQ_THRESHOLD,
            "alpha": alpha,
        }, f, indent=2)

    results_graph[ALG_NAME] = G
    results_adj[ALG_NAME] = A_df

    print(f"\n{ALG_NAME} finished. Saved {len(edges_df)} stable edges.")

    if (cfg.plot or cfg.save_plots) and (not edges_df.empty):
        out_png = os.path.join(alg_dir, "FINAL_graph_PC_Bootstrap.png") if cfg.save_plots else None
        _save_graph_plot(
            G,
            title="Final Causal DAG – PC (Bootstrapped)",
            out_png=out_png,
            node_color="lightcoral",
            edge_label_attr="freq",
            edge_width_attr="freq",
            edge_color_fn=lambda u, v, d: "green",
            show=cfg.plot,
        )

    return {
        "alg_dir": alg_dir,
        "adj_path": out_adj,
        "edges_path": out_edges,
        "graph_path": out_graph,
        "summary_path": out_json,
        "subset_cols": cols,
    }


def run_causal_discovery(
    df_clean: pd.DataFrame,
    var_types: Optional[Dict[str, str]] = None,
    config: Optional[DiscoveryConfig] = None,
    domain_exclude_vars: Optional[set] = None,
    domain_invalid_edges: Optional[set] = None,
) -> Dict[str, Any]:
    """
    One-call entry point that mirrors the notebook pipeline.

    Parameters
    ----------
    df_clean:
        Cleaned dataset (DataFrame).
    var_types:
        Optional mapping column -> {"continuous","discrete_numeric","binary","categorical",...}
        If omitted, types are auto-detected like in your notebook's Step1 fallback.
    config:
        DiscoveryConfig for algorithm parameters. If None, defaults match the notebook.
    domain_exclude_vars / domain_invalid_edges:
        Optional PC domain filters (used in notebook cell 9B).
    """
    cfg = config or DiscoveryConfig()
    _ensure_out_dir(cfg.out_dir)
    _set_global_seed(cfg.seed)

    results_graph: Dict[str, nx.DiGraph] = {}
    results_adj: Dict[str, pd.DataFrame] = {}

    df_num, numeric_cols, used_var_types = build_numeric_df(df_clean, var_types=var_types)
    print("Numeric columns used for causal discovery:", numeric_cols)
    print(f"Data matrix shape for causal discovery: {df_num.shape}")

    outputs: Dict[str, Any] = {
        "out_dir": cfg.out_dir,
        "numeric_cols": numeric_cols,
        "var_types": used_var_types,
        "results_graph": results_graph,
        "results_adj": results_adj,
        "runs": {},
    }

    # NOTEARS
    start = time.time()
    outputs["runs"]["NOTEARS"] = run_notears_bootstrapped(df_num, cfg, results_graph, results_adj)
    print("NOTEARS:", round(time.time() - start, 2), "sec")

    # GES 
    start = time.time()
    outputs["runs"]["GES"] = run_ges_bootstrap(df_num, cfg)
    print("GES:", round(time.time() - start, 2), "sec")

    # LiNGAM
    start = time.time()
    outputs["runs"]["LiNGAM"] = run_lingam_bootstrap(df_num, cfg, results_graph, results_adj)
    print("LiNGAM:", round(time.time() - start, 2), "sec")

    # DAG-GNN
    start = time.time()
    outputs["runs"]["DAGGNN"] = run_daggnn_bootstrap(df_num, cfg, results_graph, results_adj)
    print("DAGGNN:", round(time.time() - start, 2), "sec")

    # PC
    start = time.time()
    outputs["runs"]["PC"] = run_pc_bootstrap(
        df_num, cfg, results_graph, results_adj,
        domain_exclude_vars=domain_exclude_vars,
        domain_invalid_edges=domain_invalid_edges,
    )
    print("PC:", round(time.time() - start, 2), "sec")

    return outputs

