import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler

# ================= CONFIG =================

COL_MISSING_DROP_THRESHOLD = 0.30
ROW_MISSING_DROP_THRESHOLD = 0.50
WINSOR_LIMITS = (0.01, 0.01)
HIGH_CORR_THRESHOLD = 0.98

# ================= VARIABLE TYPE DETECTION =================

def detect_variable_type(
    s: pd.Series,
    n_rows: int,
    cat_threshold: int = 20,
    cat_ratio_threshold: float = 0.01
):
    non_na = s.dropna()
    if non_na.empty:
        return "constant"

    unique_vals = pd.unique(non_na)
    n_unique = len(unique_vals)

    if n_unique <= 1:
        return "constant"

    try:
        as_num = pd.to_numeric(non_na, errors="coerce").dropna()
        unique_num = set(as_num.unique())
    except Exception:
        unique_num = set()

    if unique_num and unique_num.issubset({0, 1}):
        return "binary"

    if set(non_na.unique()).issubset({True, False}):
        return "binary"

    if not pd.api.types.is_numeric_dtype(s):
        return "categorical"

    unique_ratio = n_unique / max(n_rows, 1)

    if n_unique <= cat_threshold and unique_ratio <= cat_ratio_threshold:
        return "categorical"

    if n_unique <= cat_threshold:
        return "discrete_numeric"

    return "continuous"

# ================= MAIN PIPELINE =================

def run_preprocessing(df: pd.DataFrame):
    """
    Main preprocessing pipeline.
    Input:
        df (raw dataframe from Streamlit upload)
    Returns:
        df_clean
        metadata dict (var types, scaler info, etc.)
    """

    df_clean = df.copy()

    # ---------- Variable Type Detection (Original) ----------
    var_types = {}
    n_unique_map = {}

    n_rows = df.shape[0]

    for col in df.columns:
        vtype = detect_variable_type(df[col], n_rows)
        var_types[col] = vtype
        n_unique_map[col] = df[col].nunique(dropna=True)

    # ---------- Drop High Missing Columns ----------
    col_missing_frac = df_clean.isnull().mean()
    high_na_cols = col_missing_frac[col_missing_frac > COL_MISSING_DROP_THRESHOLD].index.tolist()
    df_clean.drop(columns=high_na_cols, inplace=True)

    # ---------- Drop Constant Columns ----------
    constant_cols = [c for c in df_clean.columns if df_clean[c].nunique(dropna=False) <= 1]
    df_clean.drop(columns=constant_cols, inplace=True)

    # ---------- Drop High Missing Rows ----------
    row_missing_frac = df_clean.isnull().mean(axis=1)
    df_clean = df_clean.loc[row_missing_frac <= ROW_MISSING_DROP_THRESHOLD].copy()

    # ---------- Recompute Types ----------
    var_types_clean = {}
    for col in df_clean.columns:
        var_types_clean[col] = detect_variable_type(df_clean[col], len(df_clean))

    # ---------- Imputation ----------
    for col in df_clean.columns:
        col_type = var_types_clean[col]

        if df_clean[col].isnull().any():
            if col_type in ("continuous", "discrete_numeric"):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif col_type == "binary":
                df_clean[col] = df_clean[col].fillna(0)
            else:
                mode_val = df_clean[col].mode(dropna=True)
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Missing"
                df_clean[col] = df_clean[col].fillna(fill_val)

    # ---------- Winsorize + Scale ----------
    continuous_cols = [
        col for col, t in var_types_clean.items()
        if t == "continuous" and col in df_clean.columns
    ]

    scaler_info = None

    if len(continuous_cols) > 0:
        for col in continuous_cols:
            values = df_clean[col].astype(float).values
            df_clean[col] = np.asarray(winsorize(values, limits=WINSOR_LIMITS), dtype=float)

        scaler = StandardScaler()
        df_clean[continuous_cols] = scaler.fit_transform(df_clean[continuous_cols])

        scaler_info = {
            "columns": continuous_cols,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        }

    # ---------- Correlation Pruning ----------
    continuous_cols_clean = [c for c in continuous_cols if c in df_clean.columns]

    if len(continuous_cols_clean) >= 2:
        corr_matrix = df_clean[continuous_cols_clean].corr().abs().fillna(0)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop_corr = [
            col for col in upper.columns
            if (upper[col] > HIGH_CORR_THRESHOLD).any()
        ]

        df_clean.drop(columns=to_drop_corr, inplace=True)

    # ---------- Metadata ----------
    metadata = {
        "var_types_original": var_types,
        "var_types_clean": var_types_clean,
        "scaler_info": scaler_info
    }

    return df_clean, metadata
