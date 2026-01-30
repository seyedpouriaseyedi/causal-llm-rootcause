# causal_discovery.py

import os
import time
import numpy as np
import pandas as pd
import networkx as nx

from joblib import Parallel, delayed
from causalnex.structure.notears import from_pandas
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from lingam import DirectLiNGAM


def run_causal_discovery(df_clean: pd.DataFrame, freq_threshold=0.3, k_runs=20):
    OUT_DIR = "outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    results_graph = {}
    results_adj = {}

    def detect_type(col):
        if pd.api.types.is_numeric_dtype(col):
            n_unique = col.nunique()
            if n_unique <= 2:
                return "binary"
            elif n_unique <= 15:
                return "discrete_numeric"
            else:
                return "continuous"
        return "categorical"

    var_types = {c: detect_type(df_clean[c]) for c in df_clean.columns}
    numeric_cols = [c for c, t in var_types.items() if t in ("continuous", "discrete_numeric", "binary")]

    df_num = df_clean[numeric_cols].copy()
    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    df_num = df_num.dropna(axis=1, how="all")

    zero_var_cols = [c for c in df_num.columns if df_num[c].nunique() <= 1]
    df_num.drop(columns=zero_var_cols, inplace=True)

    if df_num.shape[1] < 2:
        raise ValueError("Not enough usable numeric variables for causal discovery.")

    cols = df_num.columns


    # ===============================
    # NOTEARS BOOTSTRAPPED
    # ===============================

    from datetime import datetime
    import json
    import matplotlib.pyplot as plt

    ALG_NAME = "NOTEARS_BOOTSTRAPPED"
    RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
    ALG_DIR = os.path.join(OUT_DIR, f"{ALG_NAME}_{RUN_TAG}")
    os.makedirs(ALG_DIR, exist_ok=True)

    # ---- Parameters (match notebook) ----
    K_RUNS = k_runs
    SAMPLE_FRAC = 0.65
    EDGE_Q = 0.9
    CAP_FACTOR = 2.0
    MIN_FACTOR = 0.5
    BOOTSTRAP = True
    SEED = 42
    N_JOBS = -1

    print(f"\nRunning {ALG_NAME}")

    # Variable Role Detection
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
        raise ValueError("NOTEARS skipped — not enough continuous variables")
        
    # DAG Enforcement Helper
    def make_dag_by_score(g_in, score_attr="score", default_score=0.0):
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
                    key=lambda x: float(
                        g.edges[x[0], x[1]].get(score_attr, default_score)
                    ),
                )
                g.remove_edge(*weakest[:2])
            except:
                break
        return g

    # Bootstrapped NOTEARS Runs
    
    df_notears = df_num[subset_cols].copy()
    X = df_notears.values
    n, p = X.shape

    def adj_from_nx(columns, g):
        A = pd.DataFrame(0, index=columns, columns=columns, dtype=int)
        for u, v in g.edges():
            A.loc[u, v] = 1
        return A

    def one_run(k_idx):

        rng = np.random.default_rng(SEED + k_idx)
        idx = rng.choice(n, int(n * SAMPLE_FRAC), replace=BOOTSTRAP)

        df_k = pd.DataFrame(X[idx], columns=subset_cols)
        sm = from_pandas(df_k)

        edges = list(sm.edges(data=True))
        weights = np.array(
            [abs(float(e[2].get("weight", 0.0))) for e in edges]
        )

        if len(weights) == 0:
            return np.zeros((p, p)), {}, 0

        thr = np.quantile(weights, EDGE_Q)

        max_edges = int(CAP_FACTOR * p)
        min_edges = int(MIN_FACTOR * p)

        top = [
            e
            for e in edges
            if abs(float(e[2].get("weight", 0.0))) >= thr
        ]

        top = sorted(
            top,
            key=lambda e: abs(float(e[2].get("weight", 0.0))),
            reverse=True,
        )

        top = top[: max(min_edges, min(len(top), max_edges))]

        g = nx.DiGraph()
        g.add_nodes_from(subset_cols)

        for u, v, d in top:
            w = float(d.get("weight", 0.0))
            g.add_edge(u, v, weight=w)

        A = adj_from_nx(subset_cols, g).values.astype(np.int8)

        wmap = {
            (u, v): float(g.edges[u, v].get("weight", 0.0))
            for u, v in g.edges()
        }

        return A, wmap, int(A.sum())
        
    # Run Bootstrapping 
    
    t0 = time.time()

    outs = Parallel(n_jobs=N_JOBS)(
        delayed(one_run)(k) for k in range(K_RUNS)
    )

    print(f"Completed {K_RUNS} NOTEARS runs in {time.time()-t0:.1f} sec")

    # Edge Frequency Matrix + Stats Table

    A_list = [o[0] for o in outs]
    w_maps = [o[1] for o in outs]

    freq = np.mean(np.stack(A_list), axis=0)
    edge_freq = pd.DataFrame(freq, index=subset_cols, columns=subset_cols)

    edge_freq_path = os.path.join(ALG_DIR, "edge_freq_after_prune.csv")
    edge_freq.to_csv(edge_freq_path)

    weights_by_edge = {}
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

                rows.append(
                    {
                        "source": u,
                        "target": v,
                        "freq": f,
                        "weight_median": float(np.median(ws)) if ws else 0,
                        "abs_weight_median": float(np.median(np.abs(ws)))
                        if ws
                        else 0,
                    }
                )

    edge_stats = pd.DataFrame(rows).sort_values(
        ["freq", "abs_weight_median"], ascending=[False, False]
    )

    edge_stats_path = os.path.join(
        ALG_DIR, "edge_stats_freq_weight_NOTEARS.csv"
    )
    edge_stats.to_csv(edge_stats_path, index=False)

    # Final Graph + DAG
    
    keep = edge_stats[edge_stats["freq"] >= freq_threshold]

    g_cons = nx.DiGraph()
    g_cons.add_nodes_from(subset_cols)

    for _, r in keep.iterrows():
        score = float(r["freq"]) + 1e-6 * float(r["abs_weight_median"])

        g_cons.add_edge(
            r["source"],
            r["target"],
            freq=r["freq"],
            weight_median=r["weight_median"],
            abs_weight_median=r["abs_weight_median"],
            score=score,
        )

    g_final = make_dag_by_score(g_cons, score_attr="score")
    A_final = adj_from_nx(subset_cols, g_final)

    # Save Outputs 
    
    out_adj = os.path.join(ALG_DIR, "FINAL_adj_NOTEARS.csv")
    out_edges = os.path.join(ALG_DIR, "FINAL_edges_NOTEARS.csv")
    out_graph = os.path.join(ALG_DIR, "FINAL_graph_NOTEARS.graphml")
    out_json = os.path.join(ALG_DIR, "FINAL_summary_NOTEARS.json")

    A_final.to_csv(out_adj)
    nx.write_graphml(g_final, out_graph)

    pd.DataFrame(
        [
            {
                "source": u,
                "target": v,
                "freq": d.get("freq", 0),
                "weight_median": d.get("weight_median", 0),
            }
            for u, v, d in g_final.edges(data=True)
        ]
    ).to_csv(out_edges, index=False)

    with open(out_json, "w") as f:
        json.dump(
            {
                "algorithm": ALG_NAME,
                "K_RUNS": K_RUNS,
                "SAMPLE_FRAC": SAMPLE_FRAC,
                "EDGE_Q": EDGE_Q,
                "FREQ_THR": freq_threshold,
                "final_edges": int(g_final.number_of_edges()),
            },
            f,
            indent=2,
        )
        
    # Graph Visualization

    pos = nx.spring_layout(g_final, seed=42)

    plt.figure(figsize=(12, 7))

    nx.draw_networkx_nodes(g_final, pos, node_size=800)
    nx.draw_networkx_labels(g_final, pos)

    edge_colors = [
        "green" if g_final[u][v].get("weight_median", 0) > 0 else "red"
        for u, v in g_final.edges()
    ]

    edge_widths = [
        2.5 * g_final[u][v].get("freq", 1) for u, v in g_final.edges()
    ]

    nx.draw_networkx_edges(
        g_final,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        arrowsize=20,
    )

    plt.title("Final NOTEARS DAG (Bootstrapped)")
    plt.axis("off")
    plt.show()

    #Register Results
    results_graph[ALG_NAME] = g_final
    results_adj[ALG_NAME] = A_final


    # ===============================
    # BOOTSTRAPPED GES
    # ===============================

    ALG_NAME = "GES_BOOTSTRAPPED"
    RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
    ALG_DIR = os.path.join(OUT_DIR, f"{ALG_NAME}_{RUN_TAG}")
    os.makedirs(ALG_DIR, exist_ok=True)

    print(f"\nRunning {ALG_NAME}")

    X = df_num.values
    cols = df_num.columns
    n, p = X.shape


        path = os.path.join(ALG_DIR, "graph.graphml")
        nx.write_graphml(G, path)

        return path

    # GES helper : EDGE Adjacency + Run Function
    def one_ges_run(k):
        rng = np.random.default_rng(SEED + 10 * k)
        idx = rng.choice(n, int(n * SAMPLE_FRAC), replace=True)
        x_sub = X[idx]

        try:
            result = ges(x_sub)
        except Exception as e:
            return pd.DataFrame(0, index=cols, columns=cols), []

        G_boot = nx.DiGraph()
        G_boot.add_nodes_from(cols)

        try:
            for (i, j) in result["G"].graph.keys():
                G_boot.add_edge(cols[i], cols[j])
        except:
            pass

        A = pd.DataFrame(0, index=cols, columns=cols)
        for u, v in G_boot.edges():
            A.loc[u, v] = 1

        edge_list = list(G_boot.edges())
        return A, edge_list
        
    #GES Bootstrapping Execution
    outs = Parallel(n_jobs=N_JOBS)(
        delayed(one_ges_run)(k) for k in range(K_RUNS)
    )

    A_list = [o[0] for o in outs]
    all_edges = [e for o in outs for e in o[1]]

    edge_freq_mat = np.mean(np.stack([a.values for a in A_list]), axis=0)
    edge_freq_df = pd.DataFrame(edge_freq_mat, index=cols, columns=cols)

    edge_freq_path = os.path.join(ALG_DIR, "edge_freq_GES.csv")
    edge_freq_df.to_csv(edge_freq_path)

    edge_counts = pd.Series(all_edges).value_counts().reset_index()
    edge_counts.columns = ["edge", "count"]
    edge_counts["source"] = edge_counts["edge"].apply(lambda x: x[0])
    edge_counts["target"] = edge_counts["edge"].apply(lambda x: x[1])
    edge_counts["freq"] = edge_counts["count"] / K_RUNS

    # Final Edge Filtering + DAG
    keep = edge_counts[edge_counts["freq"] >= freq_threshold]

    G_cons = nx.DiGraph()
    G_cons.add_nodes_from(cols)

    for _, row in keep.iterrows():
        u = row["source"]
        v = row["target"]
        f = float(row["freq"])
        G_cons.add_edge(u, v, freq=f, score=f)

    G_final = make_dag_by_score(G_cons, score_attr="score")
    A_final = adj_from_nx(cols, G_final)
    
    # Save Outputs
    
    A_final.to_csv(os.path.join(ALG_DIR, "FINAL_adj_GES.csv"))

    pd.DataFrame(
        [
            {
                "source": u,
                "target": v,
                "freq": d.get("freq", 0),
            }
            for u, v, d in G_final.edges(data=True)
        ]
    ).to_csv(os.path.join(ALG_DIR, "FINAL_edges_GES.csv"), index=False)

    nx.write_graphml(G_final, os.path.join(ALG_DIR, "FINAL_graph_GES.graphml"))

    with open(os.path.join(ALG_DIR, "FINAL_summary_GES.json"), "w") as f:
        json.dump(
            {
                "algorithm": ALG_NAME,
                "K_RUNS": K_RUNS,
                "SAMPLE_FRAC": SAMPLE_FRAC,
                "FREQ_THR": freq_threshold,
                "final_edges": int(G_final.number_of_edges()),
            },
            f,
            indent=2,
        )

    # Visualization
    pos = nx.spring_layout(G_final, seed=42)

    plt.figure(figsize=(12, 7))
    nx.draw_networkx_nodes(G_final, pos, node_size=800)
    nx.draw_networkx_labels(G_final, pos)

    edge_colors = ["blue" for _ in G_final.edges()]
    edge_widths = [
        2.5 * G_final[u][v].get("freq", 1) for u, v in G_final.edges()
    ]

    nx.draw_networkx_edges(
        G_final,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        arrowsize=20,
    )

    plt.title("Final GES DAG (Bootstrapped)")
    plt.axis("off")
    plt.show()

    # Register Results
    results_graph[ALG_NAME] = G_final
    results_adj[ALG_NAME] = A_final

    # ===============================
    # BOOTSTRAPPED LiNGAM
    # ===============================

    ALG_NAME = "LiNGAM_BOOTSTRAPPED"
    RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
    ALG_DIR = os.path.join(OUT_DIR, f"{ALG_NAME}_{RUN_TAG}")
    os.makedirs(ALG_DIR, exist_ok=True)

    print(f"\nRunning {ALG_NAME}")

    X = df_num.values
    cols = df_num.columns
    n, p = X.shape

    # LiNGAM Bootsrapping Function
    def one_lingam_run(k):
        rng = np.random.default_rng(SEED + 10 * k)
        idx = rng.choice(n, int(n * SAMPLE_FRAC), replace=True)
        x_sub = X[idx]

        try:
            model = DirectLiNGAM()
            model.fit(x_sub)
            B = model.adjacency_matrix_
        except Exception as e:
            B = np.zeros((p, p))

        return B

    # Run K Bootstraps + Calculate Frequencies 
    
    B_stack = Parallel(n_jobs=N_JOBS)(
        delayed(one_lingam_run)(k) for k in range(K_RUNS)
    )

    B_stack = np.stack(B_stack)
    B_mean = np.mean(np.abs(B_stack), axis=0)

    edge_freq_df = pd.DataFrame(B_mean, index=cols, columns=cols)
    edge_freq_path = os.path.join(ALG_DIR, "edge_strength_LiNGAM.csv")
    edge_freq_df.to_csv(edge_freq_path)

    # Build Graph from Thresholded Matrix
    
    G = nx.DiGraph()
    G.add_nodes_from(cols)

    for i, u in enumerate(cols):
        for j, v in enumerate(cols):
            if B_mean[i, j] >= freq_threshold:
                G.add_edge(u, v, weight=B_mean[i, j], score=B_mean[i, j])

    G_final = make_dag_by_score(G, score_attr="score")
    A_final = adj_from_nx(cols, G_final)

    # Save Outputs
    
    A_final.to_csv(os.path.join(ALG_DIR, "FINAL_adj_LiNGAM.csv"))

    pd.DataFrame(
        [
            {
                "source": u,
                "target": v,
                "weight": d.get("weight", 0),
            }
            for u, v, d in G_final.edges(data=True)
        ]
    ).to_csv(os.path.join(ALG_DIR, "FINAL_edges_LiNGAM.csv"), index=False)

    nx.write_graphml(G_final, os.path.join(ALG_DIR, "FINAL_graph_LiNGAM.graphml"))

    with open(os.path.join(ALG_DIR, "FINAL_summary_LiNGAM.json"), "w") as f:
        json.dump(
            {
                "algorithm": ALG_NAME,
                "K_RUNS": K_RUNS,
                "SAMPLE_FRAC": SAMPLE_FRAC,
                "FREQ_THR": freq_threshold,
                "final_edges": int(G_final.number_of_edges()),
            },
            f,
            indent=2,
        )

    # Register Results
    
    results_graph[ALG_NAME] = G_final
    results_adj[ALG_NAME] = A_final

    # Visualization
    import matplotlib.pyplot as plt

    GRAPH_PATH = os.path.join(ALG_DIR, "FINAL_graph_LiNGAM_Bootstrap.graphml")

    if not os.path.exists(GRAPH_PATH):
        print("Graph file not found.")
    else:
        G = nx.read_graphml(GRAPH_PATH)

        for u, v, d in G.edges(data=True):
            d["freq"] = float(d.get("freq", 0.0))
            d["weight_median"] = float(d.get("weight_median", 0.0))

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(12, 7))
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800)
        nx.draw_networkx_labels(G, pos, font_size=10)

        edge_colors = ['green' if d["weight_median"] > 0 else 'red' for _, _, d in G.edges(data=True)]
        edge_widths = [2.5 * d["freq"] for _, _, d in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrowsize=20)

        edge_labels = {(u, v): f"{d['weight_median']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.title("Final Causal DAG – LiNGAM (Bootstrapped)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(ALG_DIR, "FINAL_graph_LiNGAM_Bootstrap.png"), dpi=300)
        plt.show()



    # ---------------- PC ----------------
    def run_pc():
        ALG_DIR = os.path.join(OUT_DIR, "PC")
        os.makedirs(ALG_DIR, exist_ok=True)

        cg = pc(df_num.values, alpha=0.05, indep_test=fisherz)
        adj = cg.G.graph

        G = nx.DiGraph()
        cols = df_num.columns

        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] == 1:
                    G.add_edge(cols[i], cols[j])

        path = os.path.join(ALG_DIR, "graph.graphml")
        nx.write_graphml(G, path)

        return path

    # ---------------- LiNGAM ----------------
    def run_lingam():
        ALG_DIR = os.path.join(OUT_DIR, "LiNGAM")
        os.makedirs(ALG_DIR, exist_ok=True)

        model = DirectLiNGAM()
        model.fit(df_num.values)

        B = model.adjacency_matrix_

        G = nx.DiGraph()
        cols = df_num.columns

        for i, u in enumerate(cols):
            for j, v in enumerate(cols):
                if abs(B[i, j]) > 1e-4:
                    G.add_edge(u, v)

        path = os.path.join(ALG_DIR, "graph.graphml")
        nx.write_graphml(G, path)

        return path

    # ---------------- RUN ALL ----------------
    outputs = {}

    start = time.time()
    outputs["NOTEARS"] = run_notears()
    print("NOTEARS:", time.time() - start)

    try:
        start = time.time()
        outputs["GES"] = run_ges()
        print("GES:", time.time() - start)
    except Exception as e:
        outputs["GES"] = "FAILED"
        print("GES FAILED:", e)

    try:
        start = time.time()
        outputs["PC"] = run_pc()
        print("PC:", time.time() - start)
    except Exception as e:
        outputs["PC"] = "FAILED"
        print("PC FAILED:", e)

    try:
        start = time.time()
        outputs["LiNGAM"] = run_lingam()
        print("LiNGAM:", time.time() - start)
    except Exception as e:
        outputs["LiNGAM"] = "FAILED"
        print("LiNGAM FAILED:", e)

    return outputs
