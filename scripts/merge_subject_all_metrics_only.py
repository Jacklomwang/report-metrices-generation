#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def merge_to_all_metrics(out_root: Path, sub: str, ses: str) -> Path:
    """
    Build a combined MAT that contains:
      - metrics_by_task: dict (rest/sts/valsalva/breathing/spirometry)
      - whole: flattened fields used by your MATLAB report code
      - figures: paths to key figures
    """
    from scipy.io import loadmat, savemat

    base = out_root / f"sub-{sub}" / f"ses-{ses}"
    if not base.exists():
        raise FileNotFoundError(f"Cannot find session folder: {base}")

    tasks = ["rest", "sts", "valsalva", "breathing", "spirometry"]

    metrics_by_task = {}
    for task in tasks:
        p = find_task_metrics_mat(out_root, sub, ses, task)
        if p is None:
            metrics_by_task[task] = {"present": 0, "note": "missing metrics file"}
            continue

        d = loadmat(str(p), simplify_cells=True)
        d = {k: v for k, v in d.items() if not k.startswith("__")}
        d = _sanitize_for_matlab(d)
        d["present"] = 1
        d["metrics_path"] = str(p)
        metrics_by_task[task] = d

    whole = {}

    # REST
    rest = metrics_by_task.get("rest", {})
    whole["mean_MAP"] = rest.get("mean_MAP", np.nan)
    whole["mean_sysBP"] = rest.get("mean_sysBP", np.nan)
    whole["mean_diaBP"] = rest.get("mean_diaBP", np.nan)
    whole["mean_pulseBP"] = rest.get("mean_pulseBP", np.nan)
    whole["mean_RR"] = rest.get("mean_RR", np.nan)
    whole["mean_HR"] = rest.get("mean_HR", np.nan)
    whole["RMSSD"] = rest.get("RMSSD_ms", np.nan)
    whole["LF_HF_ratio"] = rest.get("LF_HF", np.nan)

    # STS
    sts = metrics_by_task.get("sts", {})
    whole["baseline_HR"] = sts.get("mean_HR_sup", np.nan)
    whole["plateau_HR"] = sts.get("mean_RR_plt", np.nan)
    whole["delta_HR"] = sts.get("dHR", np.nan)
    whole["baseline_BP"] = sts.get("mean_MAP_sup", np.nan)
    whole["plateau_BP"] = sts.get("mean_MAP_plt", np.nan)
    whole["delta_BP"] = sts.get("dMAP", np.nan)

    # Valsalva
    val = metrics_by_task.get("valsalva", {})
    whole["Valsalva_ratio"] = val.get("valsalva_ratio", np.nan)

    # Breathing
    br = metrics_by_task.get("breathing", {})
    whole["E_I_ratio"] = br.get("ratio", np.nan)
    whole["delta_HR_responses"] = br.get("diff_bpm", np.nan)

    # Spirometry
    sp = metrics_by_task.get("spirometry", {})
    whole["FEV1"] = sp.get("FEV1_max", np.nan)
    whole["FVC"] = sp.get("FVC_max", np.nan)
    whole["PEF"] = sp.get("PEF_max", np.nan)

    fev1 = whole["FEV1"]
    fvc = whole["FVC"]
    whole["FVC_over_FEV1"] = (fvc / fev1) if (np.isfinite(fev1) and fev1 != 0) else np.nan
    whole["FEV1_over_FVC"] = (fev1 / fvc) if (np.isfinite(fvc) and fvc != 0) else np.nan

    # Figures (robust: try multiple names)
    figs = {
        "STS_HR_MAP": _first_existing(
            base / "sts" / "STS_HR_MAP.png",
            base / "sts" / "STS_HR_MAP_plot.png",
        ),
        "Valsalva_plot": str(base / "valsalva" / "valsalva_best_rep_hr.png"),
        "DeepBreathing_plot": str(base / "breathing" / "breathing_hr_8to9min.png"),
    }

    out_path = base / f"sub-{sub}_ses-{ses}_all_metrics.mat"
    payload = {
        "metrics_by_task": _sanitize_for_matlab(metrics_by_task),
        "whole": _sanitize_for_matlab(whole),
        "figures": _sanitize_for_matlab(figs),
        "sub_id": f"sub-{sub}",
        "ses_id": f"ses-{ses}",
    }

    savemat(str(out_path), payload, do_compression=True)
    print(f"[OK] Saved combined bundle: {out_path}")
    return out_path


# ----------------------------
# Main: MERGE ONLY (no task reruns)
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="MERGE ONLY: Create sub-*_ses-*_all_metrics.mat from existing per-task metrics."
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
