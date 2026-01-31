import os
import json
import pandas as pd
import networkx as nx

from .normalize import normalize_edges_from_outputs
from .consensus import build_consensus_edges
from .ranking import rank_root_causes
from .evidence import build_incident_evidence
from .validator import validate_llm_json

def generate_root_cause_report_inputs(outputs: dict, df_clean: pd.DataFrame,
                                      target: str, incident_index: int,
                                      work_dir: str,
                                      min_alg_count: int = 2,
                                      min_support_mean: float = 0.30,
                                      max_path_len: int = 4) -> dict:
    """
    Deterministic pre-LLM synthesis.
    """
    os.makedirs(work_dir, exist_ok=True)

    norm_dir = os.path.join(work_dir, "normalized")
    consensus_dir = os.path.join(work_dir, "consensus")
    ranking_dir = os.path.join(work_dir, "ranking")

    norm_paths = normalize_edges_from_outputs(outputs, norm_dir)
    cons = build_consensus_edges(norm_paths, consensus_dir,
                                min_alg_count=min_alg_count,
                                min_support_mean=min_support_mean)
    ranked = rank_root_causes(cons["consensus_graph_path"], target, ranking_dir, max_path_len=max_path_len)

    # allowed sets
    G = nx.read_graphml(cons["consensus_graph_path"])
    allowed_vars = set(G.nodes())
    allowed_edges = set((u, v) for u, v in G.edges())

    # take top candidates (e.g., 5)
    cand_df = ranked["candidates_df"].copy()
    top_candidates = cand_df.head(5)["candidate"].tolist()

    evidence = build_incident_evidence(df_clean, incident_index, top_candidates, target)

    return {
        "norm_paths": norm_paths,
        "consensus": cons,
        "ranking": {
            "candidates_path": ranked["candidates_path"],
            "top_paths_path": ranked["top_paths_path"],
            "top_candidates": top_candidates,
        },
        "allowed_vars": sorted(list(allowed_vars)),
        "allowed_edges": sorted([[a,b] for (a,b) in allowed_edges]),
        "incident_evidence": evidence,
    }

def build_prompt(prompt_template_path: str, payload: dict, target: str, incident_index: int) -> str:
    with open(prompt_template_path, "r", encoding="utf-8") as f:
        tmpl = f.read()

    # load candidates table and top paths
    cand_df = pd.read_csv(payload["ranking"]["candidates_path"]).head(10)
    ranked_candidates_table = cand_df.to_csv(index=False)

    with open(payload["ranking"]["top_paths_path"], "r", encoding="utf-8") as f:
        top_paths_json = f.read()

    allowed_edges_lines = "\n".join([f"{a} -> {b}" for a, b in (tuple(x) for x in payload["allowed_edges"])])

    prompt = tmpl.format(
        target=target,
        incident_index=incident_index,
        allowed_variables="\n".join(payload["allowed_vars"]),
        allowed_edges=allowed_edges_lines,
        ranked_candidates_table=ranked_candidates_table,
        top_paths_json=top_paths_json,
        incident_evidence_json=json.dumps(payload["incident_evidence"], indent=2),
    )
    return prompt

def run_llm_and_validate(call_llm_fn, prompt: str, allowed_vars: set, allowed_edges: set) -> dict:
    """
    call_llm_fn(prompt:str)->str must return JSON text.
    """
    raw = call_llm_fn(prompt)
    parsed = validate_llm_json(raw, allowed_vars=allowed_vars, allowed_edges=allowed_edges)
    return parsed

