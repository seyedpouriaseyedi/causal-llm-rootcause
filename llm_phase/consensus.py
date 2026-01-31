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
    all_variables: list | None = None
) -> dict:

    """
    Returns:
      - consensus_edges_path
      - consensus_graph_path
      - consensus_summary_path
    """
    os.makedirs(out_dir, exist_ok=True)
    conflict_penalty = float(conflict_penalty)


    frames = []
    for alg, path in normalized_paths.items():
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        raise ValueError("No normalized edge files provided.")

    all_edges = pd.concat(frames, ignore_index=True)

    agg = (
        all_edges
        .groupby(["source","target"], as_index=False)
        .agg(
            algo_count=("alg","nunique"),
            support_mean=("support","mean"),
            support_max=("support","max"),
            strength_mean=("strength","mean"),
            algs=("alg", lambda x: sorted(list(set(x)))),
        )
    )

    # conflict detection: if both directions exist anywhere
    pair_set = set(zip(agg["source"], agg["target"]))
    conflicts = []
    for s, t in pair_set:
        if (t, s) in pair_set:
            conflicts.append((s, t))
    conflict_set = set(conflicts)

    agg["conflict"] = agg.apply(lambda r: (r["source"], r["target"]) in conflict_set, axis=1)

    # edge score
    # weighted: support + algorithm agreement
    num_algs_total = all_edges["alg"].nunique()
    agg["agreement"] = agg["algo_count"] / max(1, num_algs_total)
    agg["edge_score"] = 0.7 * agg["support_mean"] + 0.3 * agg["agreement"]

    # apply conflict penalty
    agg.loc[agg["conflict"], "edge_score"] = agg.loc[agg["conflict"], "edge_score"] - conflict_penalty


    agg["edge_score"] = agg["edge_score"].clip(lower=0.0)

    # filter
    keep = agg[
        (agg["algo_count"] >= min_alg_count) &
        (agg["support_mean"] >= min_support_mean)
    ].copy()

    keep = keep.sort_values(["edge_score","support_mean","algo_count"], ascending=False)

    consensus_edges_path = os.path.join(out_dir, "consensus_edges.csv")
    keep.to_csv(consensus_edges_path, index=False)

    # build graph
    G = nx.DiGraph()
                            
    if all_variables is not None:
        G.add_nodes_from(all_variables)
                        
    for _, r in keep.iterrows():
        G.add_edge(
            r["source"], r["target"],
            edge_score=float(r["edge_score"]),
            support_mean=float(r["support_mean"]),
            strength_mean=float(r["strength_mean"]),
            algo_count=int(r["algo_count"]),
            conflict=bool(r["conflict"]),
            algs=",".join(r["algs"]),
        )

    consensus_graph_path = os.path.join(out_dir, "consensus_graph.graphml")
    nx.write_graphml(G, consensus_graph_path)

    summary = {
        "min_alg_count": min_alg_count,
        "min_support_mean": min_support_mean,
        "conflict_penalty": conflict_penalty,
        "num_edges_kept": int(G.number_of_edges()),
        "num_nodes": int(G.number_of_nodes()),
    }
    consensus_summary_path = os.path.join(out_dir, "consensus_summary.json")
    with open(consensus_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "consensus_edges_path": consensus_edges_path,
        "consensus_graph_path": consensus_graph_path,
        "consensus_summary_path": consensus_summary_path,
    }

