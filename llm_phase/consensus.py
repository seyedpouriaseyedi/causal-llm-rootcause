# llm_phase/consensus.py
import os
import json
import pandas as pd
import networkx as nx


def build_consensus_edges(
    normalized_paths: dict,
    out_dir: str,
    min_alg_count: int = 2,
    min_support_mean: float = 0.30,
    conflict_penalty: float = 0.15,
    all_variables=None,
) -> dict:
    """
    Build a consensus edge table + graph from normalized per-algorithm edge tables.

    Outputs:
      - consensus_edges.csv
      - consensus_graph.graphml
      - consensus_summary.json

    Notes:
      - Adds all_variables as nodes (if provided) so target nodes exist even when no edges touch them.
      - Detects bidirectional conflicts and applies a penalty.
      - Resolves bidirectional pairs by keeping only the stronger direction (higher edge_score).
    """
    os.makedirs(out_dir, exist_ok=True)
    conflict_penalty = float(conflict_penalty)

    # ---- Load normalized edge tables ----
    frames = []
    for _alg, path in normalized_paths.items():
        df = pd.read_csv(path)
        frames.append(df)

    if not frames:
        raise ValueError("No normalized edge files provided.")

    all_edges = pd.concat(frames, ignore_index=True)

    # ---- Aggregate per directed edge ----
    agg = (
        all_edges
        .groupby(["source", "target"], as_index=False)
        .agg(
            algo_count=("alg", "nunique"),
            support_mean=("support", "mean"),
            support_max=("support", "max"),
            strength_mean=("strength", "mean"),
            algs=("alg", lambda x: sorted(list(set(x)))),
        )
    )

    # ---- Conflict detection: both directions exist somewhere ----
    pair_set = set(zip(agg["source"], agg["target"]))
    conflict_set = set()
    for s, t in pair_set:
        if (t, s) in pair_set:
            conflict_set.add((s, t))

    agg["conflict"] = agg.apply(lambda r: (r["source"], r["target"]) in conflict_set, axis=1)

    # ---- Edge scoring ----
    num_algs_total = int(all_edges["alg"].nunique())
    if num_algs_total <= 0:
        num_algs_total = 1

    agg["agreement"] = agg["algo_count"] / num_algs_total
    agg["edge_score"] = 0.7 * agg["support_mean"] + 0.3 * agg["agreement"]

    # apply conflict penalty
    agg.loc[agg["conflict"], "edge_score"] = agg.loc[agg["conflict"], "edge_score"] - conflict_penalty
    agg["edge_score"] = agg["edge_score"].clip(lower=0.0)

    # ---- Filter edges by stability + agreement ----
    keep = agg[
        (agg["algo_count"] >= min_alg_count) &
        (agg["support_mean"] >= min_support_mean)
    ].copy()

    keep = keep.sort_values(["edge_score", "support_mean", "algo_count"], ascending=False)

    # ---- Resolve bidirectional conflicts: keep only the stronger direction ----
    # This prevents circular explanations and reduces instability.
    keep_dict = {(r["source"], r["target"]): r for _, r in keep.iterrows()}
    resolved = {}

    for (s, t), r in keep_dict.items():
        if (t, s) in keep_dict:
            r_rev = keep_dict[(t, s)]
            if float(r["edge_score"]) >= float(r_rev["edge_score"]):
                resolved[(s, t)] = r
            else:
                resolved[(t, s)] = r_rev
        else:
            resolved[(s, t)] = r

    keep = pd.DataFrame(list(resolved.values()))
    keep = keep.sort_values(["edge_score", "support_mean", "algo_count"], ascending=False)

    # ---- Save consensus edges ----
    consensus_edges_path = os.path.join(out_dir, "consensus_edges.csv")
    keep.to_csv(consensus_edges_path, index=False)

    # ---- Build graph ----
    G = nx.DiGraph()

    if all_variables is not None:
        G.add_nodes_from(list(all_variables))

    for _, r in keep.iterrows():
        algs_val = r["algs"]
        if isinstance(algs_val, list):
            algs_str = ",".join(algs_val)
        else:
            algs_str = str(algs_val)

        G.add_edge(
            r["source"],
            r["target"],
            edge_score=float(r["edge_score"]),
            support_mean=float(r["support_mean"]),
            strength_mean=float(r["strength_mean"]),
            algo_count=int(r["algo_count"]),
            conflict=bool(r.get("conflict", False)),
            algs=algs_str,
        )

    consensus_graph_path = os.path.join(out_dir, "consensus_graph.graphml")
    nx.write_graphml(G, consensus_graph_path)

    # ---- Summary ----
    summary = {
        "min_alg_count": int(min_alg_count),
        "min_support_mean": float(min_support_mean),
        "conflict_penalty": float(conflict_penalty),
        "num_edges_kept": int(G.number_of_edges()),
        "num_nodes": int(G.number_of_nodes()),
        "num_algs_total": int(all_edges["alg"].nunique()),
    }

    consensus_summary_path = os.path.join(out_dir, "consensus_summary.json")
    with open(consensus_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "consensus_edges_path": consensus_edges_path,
        "consensus_graph_path": consensus_graph_path,
        "consensus_summary_path": consensus_summary_path,
    }
