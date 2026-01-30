# causal_discovery.py

import os
import numpy as np
import pandas as pd
import networkx as nx
import time

from joblib import Parallel, delayed
from causalnex.structure.notears import from_pandas
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from lingam import DirectLiNGAM


def run_causal_discovery(df_clean: pd.DataFrame):

    OUT_DIR = "outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    results_graph = {}
    results_adj = {}

    # ---------------- TYPE DETECTION ----------------
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

    if len(df_num.columns) < 2:
        raise ValueError("Not enough usable variables")

    # ---------------- NOTEARS ----------------
    def run_notears():

        ALG_DIR = os.path.join(OUT_DIR, "NOTEARS")
        os.makedirs(ALG_DIR, exist_ok=True)

        X = df_num.values
        cols = df_num.columns
        n = X.shape[0]

        K_RUNS = 20
        SAMPLE_FRAC = 0.7

        def one_run():
            idx = np.random.choice(n, int(n * SAMPLE_FRAC), replace=True)
            df_k = pd.DataFrame(X[idx], columns=cols)

            sm = from_pandas(df_k)

            A = pd.DataFrame(0, index=cols, columns=cols)
            for u, v, _ in sm.edges(data=True):
                A.loc[u, v] = 1

            return A.values

        outs = Parallel(n_jobs=-1)(
            delayed(one_run)() for _ in range(K_RUNS)
        )

        freq = np.mean(np.stack(outs), axis=0)
        freq_df = pd.DataFrame(freq, index=cols, columns=cols)

        freq_df.to_csv(os.path.join(ALG_DIR, "edge_frequency.csv"))

        G = nx.DiGraph()
        G.add_nodes_from(cols)

        for i, u in enumerate(cols):
            for j, v in enumerate(cols):
                if freq[i, j] > 0.3:
                    G.add_edge(u, v)

        path = os.path.join(ALG_DIR, "graph.graphml")
        nx.write_graphml(G, path)

        return path

    # ---------------- GES ----------------
    def run_ges():
        ALG_DIR = os.path.join(OUT_DIR, "GES")
        os.makedirs(ALG_DIR, exist_ok=True)

        result = ges(df_num.values)

        G = nx.DiGraph()
        cols = df_num.columns
        G.add_nodes_from(cols)

        for (i, j) in result["G"].graph.keys():
            G.add_edge(cols[i], cols[j])

        path = os.path.join(ALG_DIR, "graph.graphml")
        nx.write_graphml(G, path)

        return path

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
