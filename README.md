# Stabilizing LLM-Based Root Cause Explanations through Reliable Causal Discovery in Manufacturing Data

**Live app:** https://causal-llm-rootcause.streamlit.app/  
Repository: https://github.com/seyedpouriaseyedi/causal-llm-rootcause

This project builds an end-to-end pipeline to generate **stable, grounded root-cause reports** for manufacturing KPI deviations by combining:

1) **Multi-algorithm causal discovery** (bootstrapped): NOTEARS, GES, LiNGAM, DAG-GNN, PC  
2) **Consensus graph construction** (edge aggregation + conflict handling)  
3) **Deterministic candidate ranking** (directed paths into a target KPI)  
4) **Manual LLM report generation** (JSON-only) + **strict validation** against allowed variables and directed edges  
5) A **Streamlit app** that wraps the full workflow for interactive use

---

## Demo workflow (what the app does)

### Tab 1 — Causal Discovery
- Upload a CSV
- Run preprocessing
- Configure bootstrapping and thresholds
- Run causal discovery algorithms
- View and download outputs (edges/graphs/summaries)

### Tab 2 — LLM Root Cause Report (Manual)
- Select a **target KPI** (e.g., Rotational speed or Torque)
- Select an **incident** by row index or by top-|z| deviations
- Build:
  - consensus graph
  - ranked candidate causes
  - top supported causal paths
  - incident evidence
- Copy the prompt into ChatGPT and request **JSON only**
- Paste JSON back into the app → validate → download the validated report

### Tab 3 — LLM Q&A (Manual Prompt Generator)
- Generates a grounded prompt using your outputs
- You can paste it into ChatGPT for analysis/discussion (manual)

---

## Project structure

- `Streamlit_app.py` — Streamlit UI (Causal Discovery + Manual LLM report)
- `preprocessing.py` — preprocessing + variable type metadata
- `causal_discovery.py` — multi-algorithm bootstrapped causal discovery + outputs
- `llm_phase/` — deterministic LLM preparation:
  - consensus, ranking, evidence, prompt builder, validator
- `notebooks/` — exploration and development notebooks

---

## Installation (local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run Streamlit_app.py
