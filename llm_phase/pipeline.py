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
      1) normalize edges from algorithms
      2) build consensus graph
      3) rank candidates toward target
      4) build incident evidence for top candidates
    """
    os.makedirs(work_dir, exist_ok=True)

    norm_dir = os.path.join(work_dir, "normalized")
    consensus_dir = os.path.join(work_dir, "consensus")
    ranking_dir = os.path.join(work_dir, "ranking")

    # 1) normalize
    norm_paths = normalize_edges_from_outputs(outputs, norm_dir)

    # 2) consensus (add all variables as nodes so target exists even with no edges)
    all_vars = list(df_clean.columns)
    cons = build_consensus_edges(
        norm_paths,
        consensus_dir,
        min_alg_count=min_alg_count,
        min_support_mean=min_support_mean,
        all_variables=all_vars,
    )

    # 3) ranking (safe even if no paths exist)
    ranked = rank_root_causes(
        cons["consensus_graph_path"],
        target,
        ranking_dir,
        max_path_len=max_path_len,
    )

    # Allowed sets from consensus graph
    G = nx.read_graphml(cons["consensus_graph_path"])
    allowed_vars = set(G.nodes())
    allowed_edges = set((u, v) for u, v in G.edges())

    # Load ranked candidates (if exists)
    cand_path = ranked.get("candidates_path")
    cand_df = None
    if cand_path and os.path.exists(cand_path):
        try:
            cand_df = pd.read_csv(cand_path)
        except Exception:
            cand_df = None

    if cand_df is not None and (not cand_df.empty) and ("candidate" in cand_df.columns):
        top_candidates = cand_df.head(5)["candidate"].tolist()
    else:
        top_candidates = []

    # 4) evidence
    evidence = build_incident_evidence(
        df_clean=df_clean,
        incident_index=int(incident_index),
        variables=top_candidates,
        target=target,
    )

    return {
        "norm_paths": norm_paths,
        "consensus": cons,
        "ranking": {
            "candidates_path": ranked.get("candidates_path"),
            "top_paths_path": ranked.get("top_paths_path"),
            "top_candidates": top_candidates,
        },
        "ranking_meta": {
            "skipped": bool(ranked.get("skipped", False)),
            "reason": ranked.get("reason", None),
            "target": target,
            "n_graph_edges": ranked.get("n_graph_edges", None),
            "n_graph_nodes": ranked.get("n_graph_nodes", None),
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

    allowed_edges_lines = "\n".join(
        [f"{a} -> {b}" for a, b in (tuple(x) for x in payload.get("allowed_edges", []))]
    )

    try:
        prompt = tmpl.format(
            target=target,
            incident_index=incident_index,
            allowed_variables="\n".join(payload.get("allowed_vars", [])),
            allowed_edges=allowed_edges_lines,
            ranked_candidates_table=ranked_candidates_table,
            top_paths_json=top_paths_json,
            incident_evidence_json=json.dumps(payload.get("incident_evidence", {}), indent=2),
        )
    except KeyError as e:
        raise KeyError(
            f"Prompt template contains an unknown placeholder {e}. "
            f"If you included literal JSON braces {{ }} in the template, escape them as {{ {{ and }} }}."
        )

    return prompt


def run_llm_and_validate(call_llm_fn, prompt: str, allowed_vars: set, allowed_edges: set) -> dict:
    raw = call_llm_fn(prompt)
    return validate_llm_json(raw, allowed_vars=allowed_vars, allowed_edges=allowed_edges)
