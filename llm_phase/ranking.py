import os
import json
import pandas as pd
import networkx as nx

def _path_score(G: nx.DiGraph, path: list) -> float:
    # robust path score: min edge_score along path
    scores = []
    for u, v in zip(path[:-1], path[1:]):
        scores.append(float(G.edges[u, v].get("edge_score", 0.0)))
    return min(scores) if scores else 0.0

def rank_root_causes(consensus_graph_path: str, target: str,
                     out_dir: str,
                     max_path_len: int = 4,
                     top_paths_per_cause: int = 3,
                     max_paths_searched: int = 50) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    G = nx.read_graphml(consensus_graph_path)

    if target not in G.nodes:
        return {
            "candidates_path": None,
            "top_paths_path": None,
            "candidates_df": pd.DataFrame(),
            "paths": {},
            "skipped": True,
            "reason": "target_not_in_graph",
            "target": target,
            "graph_nodes": list(G.nodes())[:50],
        }


    # candidates = ancestors of target
    candidates = list(nx.ancestors(G, target))

    rows = []
    paths_out = {}

    for c in candidates:
        # find paths c -> target
        found = []
        # limit search; graphs can explode
        try:
            for path in nx.all_simple_paths(G, source=c, target=target, cutoff=max_path_len):
                found.append(path)
                if len(found) >= max_paths_searched:
                    break
        except Exception:
            found = []

        if not found:
            continue

        scored = [(p, _path_score(G, p)) for p in found]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_path, best_score = scored[0]
        topk = scored[:top_paths_per_cause]

        paths_out[c] = [{
            "path": p,
            "path_score": float(s),
            "edges": [{
                "source": u, "target": v,
                "edge_score": float(G.edges[u, v].get("edge_score", 0.0)),
                "support_mean": float(G.edges[u, v].get("support_mean", 0.0)),
                "algo_count": int(G.edges[u, v].get("algo_count", 0)),
                "conflict": bool(G.edges[u, v].get("conflict", False)),
                "algs": str(G.edges[u, v].get("algs", "")),
            } for u, v in zip(p[:-1], p[1:])]
        } for p, s in topk]

        # candidate summary
        rows.append({
            "candidate": c,
            "best_path": " -> ".join(best_path),
            "best_path_score": float(best_score),
            "n_paths_found": int(len(found)),
        })

    cand_df = pd.DataFrame(rows).sort_values("best_path_score", ascending=False)
    candidates_path = os.path.join(out_dir, "root_cause_candidates.csv")
    cand_df.to_csv(candidates_path, index=False)

    paths_path = os.path.join(out_dir, "top_paths.json")
    with open(paths_path, "w") as f:
        json.dump(paths_out, f, indent=2)

    return {
        "candidates_path": candidates_path,
        "top_paths_path": paths_path,
        "candidates_df": cand_df,
        "paths": paths_out,
    }

