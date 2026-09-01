#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ----------------------------
# Helpers copied from run_subject_all.py
# ----------------------------
def find_task_metrics_mat(out_root: Path, sub: str, ses: str, task: str) -> Path | None:
    """
    Standard location:
      {out_root}/sub-XXXX/ses-Y/{task}/{task}_metrics.mat
    """
    p = out_root / f"sub-{sub}" / f"ses-{ses}" / task / f"{task}_metrics.mat"
    return p if p.exists() else None


def valsalva_acq_exists(root: Path, sub: str, ses: str) -> bool:
    sub_id = f"sub-{sub}"
    ses_id = f"ses-{ses}"
    ses_dir = root / sub_id / ses_id
    if not ses_dir.exists():
        return False

    patterns = [
        f"{sub_id}_{ses_id}_task-valsalva_physio.acq",
        f"{sub_id}_{ses_id}_task-valsalva*physio*.acq",
        "*task-valsalva*physio*.acq",
        "*valsalva*.acq",
    ]
    for pat in patterns:
        if list(ses_dir.glob(pat)):
            return True
    return False


def make_placeholder_valsalva(out_root: Path, sub: str, ses: str) -> None:
    """
    Create a placeholder valsalva folder with:
      - valsalva_metrics.mat containing NaNs
      - valsalva_best_rep_hr.png containing 'Missing'
    """
    task_out = out_root / f"sub-{sub}" / f"ses-{ses}" / "valsalva"
    task_out.mkdir(parents=True, exist_ok=True)

    fig_path = task_out / "valsalva_best_rep_hr.png"
    mat_path = task_out / "valsalva_metrics.mat"

    # placeholder plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 4))
        ax = plt.gca()
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Valsalva task missing\n(no .acq found)",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
        )
        fig.savefig(fig_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"[OK] Wrote placeholder valsalva plot: {fig_path}")
    except Exception as e:
        print(f"[WARN] Could not write placeholder plot: {e}")

    # placeholder metrics
    from scipy.io import savemat

    metrics = {
        "valsalva_ratio": np.nan,
        "ratios": np.array([], dtype=float),
        "best_rep": -1,
        "starts_s": np.array([], dtype=float),
        "rep_hr_max": np.array([], dtype=float),
        "rep_hr_min": np.array([], dtype=float),
        "best_start_s": np.nan,
        "best_hr_max": np.nan,
        "best_hr_min": np.nan,
        "best_hr_max_t_s": np.nan,
        "best_hr_min_t_s": np.nan,
        "fs": np.nan,
        "sub_id": f"sub-{sub}",
        "ses_id": f"ses-{ses}",
        "acq_path": "",
        "figure_path": str(fig_path),
        "note": "Valsalva missing; placeholder created",
    }
    savemat(str(mat_path), metrics, do_compression=True)
    print(f"[OK] Wrote placeholder valsalva metrics: {mat_path}")


def _sanitize_for_matlab(x):
    """Best-effort conversion to MATLAB-friendly values."""
    if x is None:
        return np.nan

    if isinstance(x, (np.floating, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return float(x)
        return float(x)

    if isinstance(x, (np.integer, int)):
        return int(x)

    if isinstance(x, (bool, np.bool_)):
        return int(x)

    if isinstance(x, str):
        return x

    if isinstance(x, Path):
        return str(x)

    if isinstance(x, np.ndarray):
        return x

    if isinstance(x, (list, tuple)):
        try:
            return np.array(x)
        except Exception:
            return np.array([str(v) for v in x], dtype=object)

    if isinstance(x, dict):
        return {k: _sanitize_for_matlab(v) for k, v in x.items()}

    return str(x)


def _first_existing(*paths: Path) -> str:
    """Return first existing path (as str), else return the first one (as str)."""
    for p in paths:
        if p.exists():
            return str(p)
    return str(paths[0]) if paths else ""


# Spirometry results CSV — authoritative for both the measured FEV1/FVC/ratio and
# their predicted LLN/ULN range. Supersedes run_spirometry_extract.py's device-CSV
# extraction, which has a known column-matching bug for some subjects (it can pick
# up a predicted/reference value instead of the actual repeat measurement). Usually
# one row per subject; subjects with multiple rows are disambiguated by the
# "Session_Number" column (see _load_spiro_from_csv).
SPIRO_PREDICTED_CSV_PATH = Path("/export02/projects/LCS/03_spirometry/FVC_results.csv")

_SPIRO_CSV_COLUMNS = {
    # whole_key:            csv column
    "FEV1": "FEV1",
    "FVC": "FVC",
    "FEV1_over_FVC": "FEV1/FVC",
    "FEV1_LLN": "FEV1_LLN",
    "FEV1_ULN": "FEV1_ULN",
    "FVC_LLN": "FVC_LLN",
    "FVC_ULN": "FVC_ULN",
    "FEV1_over_FVC_LLN": "FEV1/FVC_LLN",
    "FEV1_over_FVC_ULN": "FEV1/FVC_ULN",
}


def _load_spiro_from_csv(csv_path: Path, sub_id: str, ses: str | None = None) -> dict:
    """Look up measured + predicted-range spirometry values for one subject/session.

    Returns {} if the CSV or the subject's row is missing, so callers can fall
    back to the older device-CSV extraction rather than erroring.

    Most subjects have exactly one row. If a subject has more than one (multiple
    sessions logged), we try to disambiguate using a "Session_Number" column matched
    against `ses`; if that column is absent or doesn't resolve to a single row,
    we fall back to the first matching row and print a warning, since silently
    picking a row from a different session could show the wrong values.
    """
    if not csv_path.exists():
        return {}
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if "Subject_ID" not in df.columns:
        return {}

    # Match case-insensitively: FVC_results.csv uses "Sub-1053" while callers pass
    # "sub-1053", so a case-sensitive compare silently drops every subject.
    matches = df[df["Subject_ID"].astype(str).str.strip().str.lower() == str(sub_id).strip().lower()]
    if matches.empty:
        return {}

    if len(matches) > 1 and "Session_Number" in df.columns and ses is not None:
        ses_matches = matches[matches["Session_Number"].astype(str).str.strip() == str(ses).strip()]
        if not ses_matches.empty:
            matches = ses_matches
        else:
            print(
                f"[WARN] {sub_id} has {len(matches)} rows in {csv_path.name} but none "
                f"match Session_Number={ses!r}; using the first row."
            )

    if len(matches) > 1:
        print(
            f"[WARN] {sub_id} has {len(matches)} ambiguous rows in {csv_path.name}; "
            f"using the first row."
        )

    row = matches.iloc[0]

    out = {}
    for whole_key, csv_col in _SPIRO_CSV_COLUMNS.items():
        if csv_col not in row or pd.isna(row[csv_col]):
            out[whole_key] = np.nan
            continue
        try:
            out[whole_key] = float(row[csv_col])
        except Exception:
            out[whole_key] = np.nan
    return out


def _first_present(d: dict, keys, default=np.nan):
    """Return the first present key from a dict, else default."""
    if not isinstance(d, dict):
        return default
    for key in keys:
        if key in d:
            return d[key]
    return default


def _to_jsonable(x):
    """Convert numpy / matlab-ish values to clean JSON-serializable Python types."""
    if x is None:
        return None

    if isinstance(x, (np.floating, float)):
        val = float(x)
        return val if math.isfinite(val) else None

    if isinstance(x, (np.integer, int)):
        return int(x)

    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    if isinstance(x, str):
        return x

    if isinstance(x, Path):
        return str(x)

    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return _to_jsonable(x.item())
        return [_to_jsonable(v) for v in x.tolist()]

    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    return str(x)


def merge_to_all_metrics(out_root: Path, sub: str, ses: str) -> Path:
    """
    Build combined MAT + JSON bundles that contain:
      - metrics_by_task: dict (rest/sts/valsalva/breathing/spirometry)
      - whole: flattened fields used by the legacy MATLAB report code
      - figures: paths to key figures
    """
    from scipy.io import loadmat, savemat

    base = out_root / f"sub-{sub}" / f"ses-{ses}"
    if not base.exists():
        raise FileNotFoundError(f"Cannot find session folder: {base}")

    tasks = ["rest", "sts", "valsalva", "breathing", "spirometry", "gas"]

    metrics_by_task = {}
    for task in tasks:
        p = find_task_metrics_mat(out_root, sub, ses, task)
        if p is None:
            metrics_by_task[task] = {"present": 0, "note": "missing metrics file"}
            continue

        d = loadmat(str(p), simplify_cells=True)
        d = {k: v for k, v in d.items() if not k.startswith("__")}
        d = _sanitize_for_matlab(d)
        d.setdefault("present", 1)
        d["metrics_path"] = str(p)
        metrics_by_task[task] = d

    whole = {}

    # REST
    rest = metrics_by_task.get("rest", {})
    whole["mean_MAP"] = _first_present(rest, ["mean_MAP", "mean_map"], np.nan)
    whole["mean_sysBP"] = _first_present(rest, ["mean_sysBP", "mean_sysbp", "mean_sbp"], np.nan)
    whole["mean_diaBP"] = _first_present(rest, ["mean_diaBP", "mean_diabp", "mean_dbp"], np.nan)
    whole["mean_pulseBP"] = _first_present(rest, ["mean_pulseBP", "mean_pulsebp"], np.nan)
    whole["mean_RR"] = _first_present(rest, ["mean_RR", "mean_rr"], np.nan)
    whole["mean_HR"] = _first_present(rest, ["mean_HR", "mean_hr"], np.nan)
    whole["RMSSD"] = _first_present(rest, ["RMSSD_ms", "RMSSD", "rmssd_ms"], np.nan)
    whole["LF_HF_ratio"] = _first_present(rest, ["LF_HF", "LF_HF_ratio", "lf_hf", "LF_HF_power"], np.nan)
    whole["ecg_p_duration_ms"] = _first_present(rest, ["ecg_p_duration_ms"], np.nan)
    whole["ecg_qrs_duration_ms"] = _first_present(rest, ["ecg_qrs_duration_ms"], np.nan)
    whole["ecg_pq_time_ms"] = _first_present(rest, ["ecg_pq_time_ms"], np.nan)
    whole["ecg_qt_time_ms"] = _first_present(rest, ["ecg_qt_time_ms"], np.nan)
    whole["mean_br"] = _first_present(rest, ["mean_br", "breathing_rate_bpm"], np.nan)
    whole["mean_etco2"] = _first_present(rest, ["mean_etco2", "etco2_mean"], np.nan)
    whole["mean_tidal_volume_ml"] = _first_present(rest, ["mean_tidal_volume_ml", "mean_tidal_volume"], np.nan)
    whole["mean_tidal_volume_l"] = _first_present(rest, ["mean_tidal_volume_l"], np.nan)
    whole["mean_minute_ventilation"] = _first_present(rest, ["mean_minute_ventilation"], np.nan)
    whole["doppler_mean_peak"] = _first_present(rest, ["doppler_mean_peak", "mean_peak"], np.nan)
    whole["doppler_mean_trough"] = _first_present(rest, ["doppler_mean_trough", "mean_trough"], np.nan)
    whole["doppler_mean_flow"] = _first_present(rest, ["doppler_mean_flow", "mean_flow", "mean_mbp"], np.nan)
    whole["doppler_mean_quality"] = _first_present(rest, ["doppler_mean_quality", "mean_quality"], np.nan)
    whole["doppler_noisy_percent"] = _first_present(rest, ["doppler_noisy_percent", "noisy_percent"], np.nan)

    # STS
    sts = metrics_by_task.get("sts", {})
    whole["baseline_HR"] = _first_present(
        sts,
        ["baseline_HR", "baseline_hr", "mean_HR_sup", "mean_hr_sup", "mean_HR_supine", "supine_mean_HR"],
        np.nan,
    )
    whole["plateau_HR"] = _first_present(
        sts,
        ["plateau_HR", "plateau_hr", "mean_HR_plt", "mean_hr_plt", "mean_HR_plateau", "plateau_mean_HR"],
        np.nan,
    )
    whole["delta_HR"] = _first_present(sts, ["delta_HR", "delta_hr", "dHR", "dhr"], np.nan)
    whole["baseline_BP"] = _first_present(
        sts,
        ["baseline_BP", "baseline_bp", "baseline_MAP", "baseline_map", "mean_MAP_sup", "mean_map_sup", "supine_mean_MAP"],
        np.nan,
    )
    whole["plateau_BP"] = _first_present(
        sts,
        ["plateau_BP", "plateau_bp", "plateau_MAP", "plateau_map", "mean_MAP_plt", "mean_map_plt", "plateau_mean_MAP"],
        np.nan,
    )
    whole["delta_BP"] = _first_present(sts, ["delta_BP", "delta_bp", "delta_MAP", "delta_map", "dMAP", "dmap"], np.nan)

    # Valsalva
    val = metrics_by_task.get("valsalva", {})
    whole["Valsalva_ratio"] = _first_present(val, ["valsalva_ratio", "Valsalva_ratio", "valsalvaRatio"], np.nan)
    whole["Valsalva_max_HR"] = _first_present(val, ["best_hr_max", "max_hr_task"], np.nan)
    whole["Valsalva_min_HR"] = _first_present(val, ["best_hr_min", "min_hr_recovery"], np.nan)
    whole["valsalva_baseline_sbp"] = _first_present(val, ["baseline_sbp"], np.nan)
    whole["valsalva_baseline_map"] = _first_present(val, ["baseline_map"], np.nan)
    whole["sbp_phase1_from_baseline"] = _first_present(val, ["sbp_phase1_from_baseline"], np.nan)
    whole["sbp_phase2_early_fall"] = _first_present(val, ["sbp_phase2_early_fall"], np.nan)
    whole["sbp_phase2_late_recovery"] = _first_present(val, ["sbp_phase2_late_recovery"], np.nan)
    whole["sbp_phase3_drop"] = _first_present(val, ["sbp_phase3_drop"], np.nan)
    whole["sbp_phase4_rise"] = _first_present(val, ["sbp_phase4_rise"], np.nan)
    whole["map_phase1_from_baseline"] = _first_present(val, ["map_phase1_from_baseline"], np.nan)
    whole["map_phase2_early_fall"] = _first_present(val, ["map_phase2_early_fall"], np.nan)
    whole["map_phase2_late_recovery"] = _first_present(val, ["map_phase2_late_recovery"], np.nan)
    whole["map_phase3_drop"] = _first_present(val, ["map_phase3_drop"], np.nan)
    whole["map_phase4_rise"] = _first_present(val, ["map_phase4_rise"], np.nan)
    whole["map_phase2_drop"] = _first_present(val, ["map_phase2_drop"], np.nan)
    whole["map_phase4_overshoot"] = _first_present(val, ["map_phase4_overshoot"], np.nan)

    # Breathing
    br = metrics_by_task.get("breathing", {})
    whole["E_I_ratio"] = _first_present(br, ["ratio", "E_I_ratio", "ei_ratio", "EIRatio"], np.nan)
    whole["delta_HR_responses"] = _first_present(
        br,
        ["diff_bpm", "delta_bpm", "HR_diff", "hr_diff", "HR_delta_RSA", "HR_delta"],
        np.nan,
    )

    # Spirometry
    sp = metrics_by_task.get("spirometry", {})

    # FVC_results.csv is authoritative when the subject is present in it (measured
    # values + predicted LLN/ULN, all from the same pipeline). Only fall back to
    # the device-CSV extraction (run_spirometry_extract.py) when the subject isn't
    # in FVC_results.csv at all — that extraction path has a known bug where it can
    # substitute a predicted/reference value for the actual measurement.
    spiro_csv = _load_spiro_from_csv(SPIRO_PREDICTED_CSV_PATH, f"sub-{sub}", ses)

    if spiro_csv:
        sp.update(spiro_csv)
        sp["present"] = 1
        sp["metrics_source"] = str(SPIRO_PREDICTED_CSV_PATH)
        if not np.isfinite(_first_present(sp, ["PEF_max", "PEF"], np.nan)):
            sp["note"] = "FEV1 and FVC loaded from FVC_results.csv; PEF is unavailable."
        elif str(sp.get("note", "")).startswith("Spirometry unavailable"):
            sp.pop("note", None)
        for key in _SPIRO_CSV_COLUMNS:
            whole[key] = spiro_csv.get(key, np.nan)
    else:
        whole["FEV1"] = _first_present(sp, ["FEV1_max", "FEV1"], np.nan)
        whole["FVC"] = _first_present(sp, ["FVC_max", "FVC"], np.nan)
        for key in ("FEV1_LLN", "FEV1_ULN", "FVC_LLN", "FVC_ULN", "FEV1_over_FVC_LLN", "FEV1_over_FVC_ULN"):
            whole[key] = np.nan
        fev1 = whole["FEV1"]
        fvc = whole["FVC"]
        whole["FEV1_over_FVC"] = (fev1 / fvc) if (np.isfinite(fvc) and fvc != 0) else np.nan

    whole["PEF"] = _first_present(sp, ["PEF_max", "PEF"], np.nan)

    fev1 = whole["FEV1"]
    fvc = whole["FVC"]
    whole["FVC_over_FEV1"] = (fvc / fev1) if (np.isfinite(fev1) and fev1 != 0) else np.nan

    figs = {
        "REST_HR": _first_existing(base / "rest" / "resting_hr.png"),
        "REST_BP": _first_existing(base / "rest" / "resting_BP.png"),
        "REST_ECG_AVERAGE": _first_existing(base / "rest" / "resting_ecg_average.png"),
        "STS_HR_MAP": _first_existing(
            base / "sts" / "STS_HR_MAP.png",
            base / "sts" / "STS_HR_MAP_plot.png",
            base / "sts" / "STS_HR_MAP.jpg",
        ),
        "Valsalva_plot": _first_existing(
            base / "valsalva" / "valsalva_best_rep_hr_bp.png",
            base / "valsalva" / "valsalva_best_rep_hr.png",
            base / "valsalva" / "valsalva_best_rep_hr.jpg",
            base / "valsalva" / "valsalva_best_rep_hr.tif",
        ),
        "Valsalva_debug_plot": _first_existing(base / "valsalva" / "valsalva_debug_full_hr.png"),
        "DeepBreathing_plot": _first_existing(
            base / "breathing" / "deep_breathing_HR_plot.png",
            base / "breathing" / "breathing_hr_7to8min.png",
            base / "breathing" / "breathing_hr_8to9min.png",
            base / "breathing" / "breathing_hr_window.png",
            base / "breathing" / "breathing_hr_plot.png",
        ),
        "Spirometry_plot": _first_existing(base / "spirometry" / "spirometry_summary.png"),
        "Gas_plot": _first_existing(base / "gas" / "gas_signals.png"),
    }

    out_mat = base / f"sub-{sub}_ses-{ses}_all_metrics.mat"
    out_json = base / f"sub-{sub}_ses-{ses}_all_metrics.json"
    payload = {
        "metrics_by_task": _sanitize_for_matlab(metrics_by_task),
        "whole": _sanitize_for_matlab(whole),
        "figures": _sanitize_for_matlab(figs),
        "sub_id": f"sub-{sub}",
        "ses_id": f"ses-{ses}",
    }

    savemat(str(out_mat), payload, do_compression=True)
    print(f"[OK] Saved combined MAT bundle: {out_mat}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)
    print(f"[OK] Saved combined JSON bundle: {out_json}")

    return out_mat


# ----------------------------
# Main: MERGE ONLY (no task reruns)
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="MERGE ONLY: Create sub-*_ses-*_all_metrics.{mat,json} from existing per-task metrics."
    )
    ap.add_argument("--out_root", default="derived", help="Output root folder (e.g., derived)")
    ap.add_argument("--sub", required=True, help="Subject code like 2062")
    ap.add_argument("--ses", default="1", help="Session number like 1 or 2")

    # Optional: only used to decide whether to create valsalva placeholder
    ap.add_argument("--root", default=None, help="Physio root (only used to check if valsalva .acq exists)")
    ap.add_argument(
        "--make_valsalva_placeholder_if_missing",
        action="store_true",
        help="If valsalva_metrics.mat missing AND no valsalva .acq exists, create placeholder outputs.",
    )

    args = ap.parse_args()
    out_root = Path(args.out_root)

    # Optional placeholder for valsalva
    if args.make_valsalva_placeholder_if_missing:
        vm = find_task_metrics_mat(out_root, args.sub, args.ses, "valsalva")
        if vm is None:
            if args.root is None:
                print("[WARN] Valsalva metrics missing, but --root not provided; skipping placeholder check.")
            else:
                root = Path(args.root)
                if valsalva_acq_exists(root, args.sub, args.ses):
                    print("[INFO] Valsalva .acq exists but metrics missing; not creating placeholder.")
                else:
                    print("[WARN] Valsalva missing and no .acq found -> creating placeholder.")
                    make_placeholder_valsalva(out_root, args.sub, args.ses)

    # Merge
    merge_to_all_metrics(out_root, args.sub, args.ses)


if __name__ == "__main__":
    main()
