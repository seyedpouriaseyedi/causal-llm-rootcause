# Stabilizing LLM-Based Root Cause Explanations through Reliable Causal Discovery in Manufacturing Data

This project builds a reproducible pipeline to generate **stable, grounded root-cause reports** for manufacturing KPI deviations by:
1) learning causal graphs with multiple causal discovery algorithms (bootstrapped),
2) aggregating them into a **consensus graph**,
3) ranking candidate root causes via directed paths to a chosen KPI target,
4) using a **manual LLM step** to produce a structured JSON report,
5) validating the JSON strictly against allowed variables and directed edges.

## Repo structure

- `Streamlit_app.py` — Streamlit UI (Causal Discovery tab + Manual LLM Report tab + Prompt Generator tab)
- `preprocessing.py` — dataset cleaning + type detection
- `causal_discovery.py` — bootstrapped NOTEARS / GES / LiNGAM / DAG-GNN / PC orchestration + outputs
- `llm_phase/` — deterministic consensus + ranking + evidence + prompt builder + validator
- `notebooks/` — development notebooks (exploration; not required for running the app)
- `Sample_Dataset/` — demo dataset 

## Quickstart (local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run Streamlit_app.py

