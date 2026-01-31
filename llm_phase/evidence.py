import numpy as np
import pandas as pd

def build_incident_evidence(df_clean: pd.DataFrame, incident_index: int,
                            variables: list, target: str) -> dict:
    """
    Evidence = deterministic stats (z-score, percentile, abnormal flag).
    """
    row = df_clean.iloc[incident_index]

    evidence_rows = []
    for v in variables:
        if v not in df_clean.columns:
            continue
        x = df_clean[v]
        val = row[v]

        # numeric evidence
        if pd.api.types.is_numeric_dtype(x):
            mu = float(np.nanmean(x.values))
            sd = float(np.nanstd(x.values)) if float(np.nanstd(x.values)) > 1e-12 else 1.0
            z = float((val - mu) / sd)
            pct = float((x.rank(pct=True).iloc[incident_index]) * 100.0)
            abnormal = bool(abs(z) >= 2.0)
            evidence_rows.append({
                "variable": v,
                "value": float(val),
                "mean": mu,
                "std": sd,
                "z_score": z,
                "percentile": pct,
                "abnormal": abnormal,
            })
        else:
            # fallback
            evidence_rows.append({
                "variable": v,
                "value": str(val),
                "abnormal": None,
            })

    # target context
    t_val = row[target] if target in df_clean.columns else None

    # sort by abnormal first, then |z|
    def sort_key(r):
        z = abs(r.get("z_score", 0.0)) if r.get("z_score") is not None else 0.0
        ab = 1 if r.get("abnormal") else 0
        return (ab, z)

    evidence_rows = sorted(evidence_rows, key=sort_key, reverse=True)

    return {
        "incident_index": int(incident_index),
        "target": target,
        "target_value": None if t_val is None else (float(t_val) if isinstance(t_val, (int,float,np.number)) else str(t_val)),
        "evidence": evidence_rows,
    }

