import os
import pandas as pd

def _sign_from_weight(x: float):
    if x > 0: return "+"
    if x < 0: return "-"
    return "0"

def normalize_edges_from_outputs(outputs: dict, out_dir: str) -> dict:
    """
    Takes your run_causal_discovery() outputs and writes normalized edge tables per algorithm.
    Returns dict: {alg_key: normalized_csv_path}
    """
    os.makedirs(out_dir, exist_ok=True)

    runs = outputs.get("runs", {})
    norm_paths = {}

    # NOTEARS
    if "NOTEARS" in runs and runs["NOTEARS"].get("edges_path"):
        df = pd.read_csv(runs["NOTEARS"]["edges_path"])
        df["alg"] = "NOTEARS"
        df["support"] = df["freq"].astype(float)
        df["strength"] = df["weight_median"].abs().astype(float)
        df["sign"] = df["weight_median"].apply(_sign_from_weight)
        df["notes"] = ""
        norm = df[["source","target","alg","support","strength","sign","notes"]]
        path = os.path.join(out_dir, "notears_edges_normalized.csv")
        norm.to_csv(path, index=False)
        norm_paths["NOTEARS"] = path

    # LiNGAM
    if "LiNGAM" in runs and runs["LiNGAM"].get("edges_path"):
        df = pd.read_csv(runs["LiNGAM"]["edges_path"])
        df["alg"] = "LiNGAM"
        df["support"] = df["freq"].astype(float)
        df["strength"] = df["weight_median"].abs().astype(float)
        df["sign"] = df["weight_median"].apply(_sign_from_weight)
        df["notes"] = ""
        norm = df[["source","target","alg","support","strength","sign","notes"]]
        path = os.path.join(out_dir, "lingam_edges_normalized.csv")
        norm.to_csv(path, index=False)
        norm_paths["LiNGAM"] = path

    # DAG-GNN
    if "DAGGNN" in runs and runs["DAGGNN"].get("edges_path") and not runs["DAGGNN"].get("skipped"):
        df = pd.read_csv(runs["DAGGNN"]["edges_path"])
        # might already have weight_median and freq
        df["alg"] = "DAG-GNN"
        df["support"] = df["freq"].astype(float)
        df["strength"] = df["weight_median"].abs().astype(float)
        df["sign"] = df["weight_median"].apply(_sign_from_weight)
        df["notes"] = ""
        norm = df[["source","target","alg","support","strength","sign","notes"]]
        path = os.path.join(out_dir, "daggnn_edges_normalized.csv")
        norm.to_csv(path, index=False)
        norm_paths["DAG-GNN"] = path

    # PC
    if "PC" in runs and runs["PC"].get("edges_path") and not runs["PC"].get("skipped"):
        df = pd.read_csv(runs["PC"]["edges_path"])
        df["alg"] = "PC"
        df["support"] = df["freq"].astype(float)
        df["strength"] = df["corr"].abs().astype(float)
        df["sign"] = df["corr"].apply(_sign_from_weight)
        df["notes"] = ""
        norm = df[["source","target","alg","support","strength","sign","notes"]]
        path = os.path.join(out_dir, "pc_edges_normalized.csv")
        norm.to_csv(path, index=False)
        norm_paths["PC"] = path

    # GES stable edges (if present)
    if "GES" in runs and runs["GES"].get("stable_edges_path"):
        try:
            df = pd.read_csv(runs["GES"]["stable_edges_path"])
            if not df.empty:
                df["alg"] = "GES"
                df["support"] = df["freq"].astype(float)
                df["strength"] = df["median_score"].abs().astype(float)
                df["sign"] = "?"  # score sign not reliable here
                df["notes"] = ""
                norm = df[["source","target","alg","support","strength","sign","notes"]]
                path = os.path.join(out_dir, "ges_edges_normalized.csv")
                norm.to_csv(path, index=False)
                norm_paths["GES"] = path
        except Exception:
            pass

    return norm_paths

