# llm_phase/pipeline.py
import os
import json
import pandas as pd
import networkx as nx

from .normalize import normalize_edges_from_outputs
from .consensus import build_consensus_edges
from .ranking import rank_root_causes
from .evidence import build_incident_evidence
from .validator import validate_llm_json


def generate_root_cause_report_inputs(
    outputs: dict,
    df_clean: pd.DataFrame,
    target: str,
    incident_index: int,
    work_dir: str,
    min_alg_count: int = 2,
    min_support_mean: float = 0.30,
    max_path_len: int = 4,
) -> dict:
    """
    Deterministic pre-LLM synthesis:
      - normalize edges
      - build consensus graph
      - rank candidates toward target
      - compute incident evidence for top candidates
    """
    os.makedirs(work_dir, exist_ok=True)

    norm_dir = os.path.join(work_dir, "normalized")
    consensus_dir = os.path.join(work_dir, "consensus")
    ranking_dir = os.path.join(work_dir, "ranking")

    norm_paths = normalize_edges_from_outputs(outputs, norm_dir)

    # IMPORTANT: add all dataset columns as nodes so target exists even if no edges touch it
    all_vars = list(df_clean.columns)

    cons = build_consensus_edges(
        norm_paths,
        consensus_dir,
        min_alg_count=min_alg_count,
        min_support_mean=min_support_mean,
        all_variables=all_vars,
    )

    ranked = rank_root_causes(
        cons["consensus_graph_path"],
        target,
        ranking_dir,
        max_path_len=max_path_len,
    )

    # allowed sets come from consensus graph (nodes always exist now)
    G = nx.read_graphml(cons["consensus_graph_path"])
    allowed_vars = set(G.nodes())
    allowed_edges = set((u, v) for u, v in G.edges())

    # top candidates for evidence (if ranking empty, evidence will be empty)
    cand_df = ranked.get("candidates_df")
    if cand_df is not None and (not cand_df.empty):
        top_candidates = cand_df.head(5)["candidate"].tolist()
    else:
        top_candidates = []

    evidence = build_incident_evidence(df_clean, incident_index, top_candidates, target)

    return {
        "norm_paths": norm_paths,
        "consensus": cons,
        "ranking": {
            "candidates_path": ranked.get("candidates_path"),
            "top_paths_path": ranked.get("top_paths_path"),
            "top_candidates": top_candidates,
        },
        "allowed_vars": sorted(list(allowed_vars)),
        "allowed_edges": sorted([[a, b] for (a, b) in allowed_edges]),
        "incident_evidence": evidence,
    }


def build_prompt(prompt_template_path: str, payload: dict, target: str, incident_index: int) -> str:
    with open(prompt_template_path, "r", encoding="utf-8") as f:
        tmpl = f.read()

    candidates_path = payload["ranking"].get("candidates_path")
    top_paths_path = payload["ranking"].get("top_paths_path")

    if candidates_path and os.path.exists(candidates_path):
        cand_df = pd.read_csv(candidates_path).head(10)
        ranked_candidates_table = cand_df.to_csv(index=False)
    else:
        ranked_candidates_table = "No candidates found under current thresholds."

    if top_paths_path and os.path.exists(top_paths_path):
        with open(top_paths_path, "r", encoding="utf-8") as f:
            top_paths_json = f.read()
    else:
        top_paths_json = "{}"

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
    raw = call_llm_fn(prompt)
    return validate_llm_json(raw, allowed_vars=allowed_vars, allowed_edges=allowed_edges)
