#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


SPIRO_DIR_DEFAULT = "/export02/projects/LCS/03_spirometry"

import re

def _norm_col(x: str) -> str:
    # strip spaces + remove BOM/zero-width chars that sometimes appear in CSV headers
    s = str(x).strip().replace("\ufeff", "").replace("\u200b", "")
    return s

def adjacent_pred_column(df, base_col: str, pred_prefix: str = "_PRED") -> str | None:
    cols = list(df.columns)
    try:
        i = cols.index(base_col)
    except ValueError:
        return None
    if i + 1 >= len(cols):
        return None

    nxt = _norm_col(cols[i + 1])

    # Accept: "_PRED", "_PRED.1", "_PRED.23", etc.
    if re.match(rf"^{re.escape(pred_prefix)}(\.\d+)?$", nxt, flags=re.IGNORECASE):
        return cols[i + 1]

    return None

def pick_latest_spiro_csv(spiro_dir: Path) -> Path:
    """Pick most up-to-date CSV starting with spiro_data_*.csv (by mtime)."""
    cands = list(spiro_dir.glob("spiro_data_*.csv"))
    if not cands:
        raise FileNotFoundError(f"No spiro_data_*.csv found in {spiro_dir}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def normalize_sub_key(x: str) -> str:
    """Normalize subject id strings to help matching."""
    s = str(x).strip()
    s = s.replace("_", "-")
    s = s.lower()
    return s


def find_subject_rows(df: pd.DataFrame, sub_code: str, id_col: str | None = None) -> pd.DataFrame:
    """
    Find rows matching subject code. Accepts formats like:
      sub-2062, sub_2062, 2062, etc.
    """
    # Choose ID column
    if id_col and id_col in df.columns:
        col = id_col
    else:
        # Try common names
        candidates = [c for c in df.columns if str(c).lower() in ("sub_id", "subject", "subject_id", "participant", "participant_id", "id")]
        col = candidates[0] if candidates else df.columns[0]

    series = df[col].astype(str).map(normalize_sub_key)

    sub_code_clean = normalize_sub_key(sub_code)
    # Accept "2062" or "sub-2062"
    if sub_code_clean.startswith("sub-"):
        code_num = sub_code_clean.replace("sub-", "")
    else:
        code_num = sub_code_clean
    patterns = [
        rf"^sub-{re.escape(code_num)}$",
        rf"^sub{re.escape(code_num)}$",
        rf"^{re.escape(code_num)}$",
        rf"^sub[-_]?{re.escape(code_num)}$",
    ]
    # Also allow "contains" as fallback (some sheets have extra text)
    exact_match = series.apply(lambda s: any(re.match(p, s) for p in patterns))
    if exact_match.any():
        return df.loc[exact_match].copy()

    contains_match = series.str.contains(rf"{re.escape(code_num)}", na=False)
    return df.loc[contains_match].copy()


def find_col(df: pd.DataFrame, variants: list[str]) -> str | None:
    """Find a column by case-insensitive exact match among variants."""
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for v in variants:
        key = v.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def to_numeric(series: pd.Series) -> np.ndarray:
    """Convert series to numeric, handling commas as decimal separators if present."""
    s = series.astype(str).str.strip()
    # Replace comma decimals (e.g., "3,41") -> "3.41"
    s = s.str.replace(",", ".", regex=False)
    # Remove non-numeric except dot, minus, exp
    s = s.str.replace(r"[^0-9eE\.\-\+]", "", regex=True)
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def nanmax_or_nan(x: np.ndarray) -> float:
    return float(np.nanmax(x)) if np.isfinite(x).any() else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Extract spirometry metrics for a subject and save to derived/.")
    ap.add_argument("--spiro_dir", default=SPIRO_DIR_DEFAULT, help="Spirometry folder (default LCS/03_spirometry)")
    ap.add_argument("--sub", required=True, help="Subject code like 2062 (or sub-2062)")
    ap.add_argument("--ses", default="1", help="Session label used only for output path (default 1 -> ses-1)")
    ap.add_argument("--id_col", default=None, help="Optional column name that contains subject IDs")
    ap.add_argument("--out_root", default="derived", help="Output root (default derived)")
    args = ap.parse_args()

    spiro_dir = Path(args.spiro_dir)
    out_root = Path(args.out_root)
    sub_id = args.sub if args.sub.startswith("sub-") else f"sub-{args.sub}"
    ses_id = f"ses-{args.ses}"

    csv_path = pick_latest_spiro_csv(spiro_dir)
    print(f"[INFO] Using spirometry CSV: {csv_path}")

    # Read CSV with semicolon separator
    df = pd.read_csv(csv_path, sep=";", engine="python")
    print(f"[INFO] CSV shape: {df.shape[0]} rows x {df.shape[1]} cols")

    sub_rows = find_subject_rows(df, sub_id, id_col=args.id_col)
    if sub_rows.empty:
        raise RuntimeError(f"No rows found for subject {sub_id} in {csv_path.name}")

    print(f"[INFO] Found {len(sub_rows)} spirometry row(s) for {sub_id}")

    col_fev1 = find_col(sub_rows, ["FEV1", "FEV 1"])
    col_fvc  = find_col(sub_rows, ["FVC"])
    col_pef  = find_col(sub_rows, ["PEF"])

    # FEV1 value should come from the _PRED column immediately after FEV1
    col_fev1_pred_adj = None
    if col_fev1 is not None:
        col_fev1_pred_adj = adjacent_pred_column(df, col_fev1)

    if col_fev1_pred_adj is None:
        print("[WARN] Could not find adjacent _PRED column right after FEV1. "
            "Will fall back to FEV1 column (may be zero).")



    # ratio can appear either way
    col_fvc_fev1 = find_col(sub_rows, ["FVC/FEV1", "FVC / FEV1", "FVC/FEV 1", "FVC_FEV1"])
    col_fev1_fvc = find_col(sub_rows, ["FEV1/FVC", "FEV1 / FVC", "FEV 1/FVC", "FEV1_FVC"])

    # FEV1 values: use adjacent _PRED if available; otherwise use FEV1 itself
    if col_fev1_pred_adj is not None:
        fev1_vals = to_numeric(sub_rows[col_fev1_pred_adj])
        fev1_source = "FEV1_adjacent__PRED"
    else:
        fev1_vals = to_numeric(sub_rows[col_fev1]) if col_fev1 else np.array([np.nan])
        fev1_source = "FEV1"

    fev1_max = nanmax_or_nan(fev1_vals)

    fvc_vals  = to_numeric(sub_rows[col_fvc])  if col_fvc  else np.array([np.nan])
    pef_vals  = to_numeric(sub_rows[col_pef])  if col_pef  else np.array([np.nan])

    fvc_max  = nanmax_or_nan(fvc_vals)
    pef_max  = nanmax_or_nan(pef_vals)

    print(f"[INFO] FEV1 source: {fev1_source}")




    # Ratio: prefer whichever exists; otherwise compute from maxima (and also provide both)
    fev1_over_fvc_max = (fev1_max / fvc_max) if np.isfinite(fev1_max) and np.isfinite(fvc_max) and fvc_max != 0 else float("nan")
    fvc_over_fev1_max = (fvc_max / fev1_max) if np.isfinite(fev1_max) and np.isfinite(fvc_max) and fev1_max != 0 else float("nan")


    print("\n===== SPIROMETRY (max across repeats) =====")
    print(f"FEV1 max: {fev1_max:.4g}")
    print(f"FVC  max: {fvc_max:.4g}")
    print(f"PEF  max: {pef_max:.4g}")
    print(f"FEV1/FVC max: {fev1_over_fvc_max:.4g}")
    print(f"FVC/FEV1 max: {fvc_over_fev1_max:.4g}")


    # Save to derived/
    task_out = out_root / sub_id / ses_id / "spirometry"
    task_out.mkdir(parents=True, exist_ok=True)
    out_mat = task_out / "spirometry_metrics.mat"

    metrics = {
        "sub_id": sub_id,
        "ses_id": ses_id,
        "source_csv": str(csv_path),
        "n_rows_found": int(len(sub_rows)),
        "FEV1_max": float(fev1_max),
        "FVC_max": float(fvc_max),
        "PEF_max": float(pef_max),
        "FEV1_over_FVC_max": float(fev1_over_fvc_max),
        "FVC_over_FEV1_max": float(fvc_over_fev1_max),
    }

    savemat(str(out_mat), metrics, do_compression=True)
    print(f"\n[OK] Saved: {out_mat}")


if __name__ == "__main__":
    main()
