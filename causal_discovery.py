# causal_discovery.py

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
import time

from joblib import Parallel, delayed
from causalnex.structure.notears import from_pandas
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from lingam import DirectLiNGAM


# =========================
# MAIN ENTRY FUNCTION
# =========================

def run_causal_discovery(df_clean: pd.DataFrame):

    OUT_DIR = "outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    results_graph = {}
    results_adj = {}

    # =========================
    # Prepare numeric dataset
    # =========================

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

    numeric_cols = [
        c for c, t in var_types.items()
        if t in ("continuous", "discrete_numeric", "binary")
    ]

    df_num = df_clean[numeric_cols].copy()

    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    df_num = df_num.dropna(axis=1, how="all")

    zero_var = [c for c in df_num.columns if df_num[c].nunique() <= 1]
    df_num.drop(columns=zero_var, inplace=True)

    subset_cols = list(df_num.columns)

    if len(subset_cols) < 2:
        raise ValueError("Not enough usable variables for causal discovery.")

    # =========================
    # NOTEARS BOOTSTRAP
    # =========================

    def run_notears():

        ALG_NAME = "NOTEARS_BOOTSTRAP"
        ALG_DIR = os.path.join(OUT_DIR, ALG_NAME)
        os.makedirs(ALG_DIR, exist_ok=True)

        df_notears = df_num.copy()
        X = df_notears.values
        cols = df_notears.columns

        n, p = X.shape

        K_RUNS = 20
        SAMPLE_FRAC = 0.7
        EDGE_Q = 0.9

        def one_run(k):

            idx = np.random.choice(n, int(n * SAMPLE_FRAC), replace=True)
            df_k = pd.DataFrame(X[idx], columns=cols)

            sm = from_pandas(df_k)

            g = nx.DiGraph()
            g.add_nodes_from(cols)

            for u, v, d in sm.edges(data=True):
                g.add_edge(u, v, weight=float(d.get("weight", 0)))

            A = pd.DataFrame(0, index=cols, columns=cols)

            for u, v in g.edges():
                A.loc[u, v] = 1

            return A.values

        outs = Parallel(n_jobs=-1)(
            delayed(one_run)(k) for k in range(K_RUNS)
        )

        freq = np.mean(np.stack(outs), axis=0)

        freq_df = pd.DataFrame(freq, index=cols, columns=cols)

        freq_path = os.path.join(ALG_DIR, "edge_frequency.csv")
        freq_df.to_csv(freq_path)

        G_final = nx.DiGraph()
        G_final.add_nodes_from(cols)

        for i, u in enumerate(cols):
            for j, v in enumerate(cols):
                if freq[i, j] > 0.3:
                    G_final.add_edge(u, v, freq=float(freq[i, j]))

        graph_path = os.path.join(ALG_DIR, "final_graph.graphml")
        nx.write_graphml(G_final, graph_path)

        results_graph[ALG_NAME] = G_final
        results_adj[ALG_NAME] = freq_df

        return graph_path

    # =========================
    # GES
    # =========================

    def run_ges():

        ALG_NAME = "GES"
        ALG_DIR = os.path.join(OUT_DIR, ALG_NAME)
        os.makedirs(ALG_DIR, exist_ok=True)

        X = df_num.values
        result = ges(X)

        G = nx.DiGraph()
        G.add_nodes_from(df_num.columns)

        for (i, j) in result["G"].graph.keys():
            G.add_edge(df_num.columns[i], df_num.columns[j])

        graph_path = os.path.join(ALG_DIR, "graph.graphml")
        nx.write_graphml(G, graph_path)

        results_graph[ALG_NAME] = G

        return graph_path

    # =========================
    # PC
    # =========================

    def run_pc():

        ALG_NAME = "PC"
        ALG_DIR = os.path.join(OUT_DIR, ALG_NAME)
        os.makedirs(ALG_DIR, exist_ok=True)

        X = df_num.values

        cg = pc(X, alpha=0.05, indep_test=fisherz)

        adj = cg.G.graph

        G = nx.DiGraph()
        cols = df_num.columns

        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] == 1:
                    G.add_edge(cols[i], cols[j])

        graph_path = os.path.join(ALG_DIR, "graph.graphml")
        nx.write_graphml(G, graph_path)

        results_graph[ALG_NAME] = G

        return graph_path

    # =========================
    # LiNGAM
    # =========================

    def run_lingam():

        ALG_NAME = "LiNGAM"
        ALG_DIR = os.path.join(OUT_DIR, ALG_NAME)
        os.makedirs(ALG_DIR, exist_ok=True)

        model = DirectLiNGAM()
        model.fit(df_num.values)

        B = model.adjacency_matrix_

        G = nx.DiGraph()
        cols = df_num.columns

        for i, u in enumerate(cols):
            for j, v in enumerate(cols):
                if abs(B[i, j]) > 1e-4:
                    G.add_edge(u, v, weight=float(B[i, j]))

        graph_path = os.path.join(ALG_DIR, "graph.graphml")
        nx.write_graphml(G, graph_path)

        results_graph[ALG_NAME] = G

        return graph_path

# =========================
# RUN ALL
# =========================

outputs = {}

# NOTEARS
start = time.time()
outputs["NOTEARS"] = run_notears()
print(f"NOTEARS finished in {time.time() - start:.2f} seconds")

# GES
start = time.time()
try:
    outputs["GES"] = run_ges()
    print(f"GES finished in {time.time() - start:.2f} seconds")
except Exception as e:
    outputs["GES"] = "FAILED"
    print(f"GES failed after {time.time() - start:.2f} seconds: {str(e)}")

# PC
start = time.time()
try:
    outputs["PC"] = run_pc()
    print(f"PC finished in {time.time() - start:.2f} seconds")
except Exception as e:
    outputs["PC"] = "FAILED"
    print(f"PC failed after {time.time() - start:.2f} seconds: {str(e)}")

# LiNGAM
start = time.time()
try:
    outputs["LiNGAM"] = run_lingam()
    print(f"LiNGAM finished in {time.time() - start:.2f} seconds")
except Exception as e:
    outputs["LiNGAM"] = "FAILED"
    print(f"LiNGAM failed after {time.time() - start:.2f} seconds: {str(e)}")

    return outputs
