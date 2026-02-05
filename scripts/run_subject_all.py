#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import math
import numpy as np


def run(cmd: list[str], allow_fail: bool = False) -> int:
    print("\n[RUN]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        if allow_fail:
            print(f"[WARN] Command failed (continuing): {e}")
            return int(e.returncode) if e.returncode is not None else 1
        raise


def script_path(name: str) -> str:
    p = Path(__file__).resolve().parent / name
    if not p.exists():
        raise FileNotFoundError(f"Missing script: {p}")
    return str(p)


def find_task_metrics_mat(out_root: Path, sub: str, ses: str, task: str) -> Path | None:
    """
    Standard location we have been using:
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
    This keeps downstream report generation from crashing.
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
    """
    Convert Python objects into MATLAB-friendly values:
      - None -> NaN
      - bool/int/float -> float or int ok
      - str -> str
      - list/tuple -> np.array (best-effort)
      - dict -> recursive dict (but savemat will store as struct only if wrapped carefully)
    For our use, we keep dict-of-scalars/arrays and let savemat handle it.
    """
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
        # try numeric array first, else string array
        try:
            return np.array(x)
        except Exception:
            return np.array([str(v) for v in x], dtype=object)

    if isinstance(x, dict):
        return {k: _sanitize_for_matlab(v) for k, v in x.items()}

    # last resort: string
    return str(x)


def merge_to_all_metrics(out_root: Path, sub: str, ses: str) -> Path:
    """
    Build a combined MAT that contains:
      - metrics_by_task: struct-like dict: rest/sts/valsalva/breathing/spirometry
      - whole: flattened fields used by your MATLAB report code
    """
    from scipy.io import loadmat, savemat

    base = out_root / f"sub-{sub}" / f"ses-{ses}"
    base.mkdir(parents=True, exist_ok=True)

    tasks = ["rest", "sts", "valsalva", "breathing", "spirometry"]

    metrics_by_task = {}
    for task in tasks:
        p = find_task_metrics_mat(out_root, sub, ses, task)
        if p is None:
            metrics_by_task[task] = {"present": 0, "note": "missing metrics file"}
            continue

        d = loadmat(str(p), simplify_cells=True)
        # keep everything except MATLAB headers
        d = {k: v for k, v in d.items() if not k.startswith("__")}
        # sanitize
        d = _sanitize_for_matlab(d)
        d["present"] = 1
        d["metrics_path"] = str(p)
        metrics_by_task[task] = d

    # ---- Build "whole" (fields your compile_whole expects)
    whole = {}

    # REST mapping (from your rest script bundle)
    rest = metrics_by_task.get("rest", {})
    # Use safe .get because rest may be "missing metrics file"
    whole["mean_MAP"] = rest.get("mean_MAP", np.nan)
    whole["mean_sysBP"] = rest.get("mean_sysBP", np.nan)
    whole["mean_diaBP"] = rest.get("mean_diaBP", np.nan)
    whole["mean_pulseBP"] = rest.get("mean_pulseBP", np.nan)

    whole["mean_RR"] = rest.get("mean_RR", np.nan)
    whole["mean_HR"] = rest.get("mean_HR", np.nan)
    whole["RMSSD"] = rest.get("RMSSD_ms", np.nan)  # already ms in your printouts
    whole["LF_HF_ratio"] = rest.get("LF_HF", np.nan)

    # STS mapping (you said all in sts_metrics.mat under sts folder)
    sts = metrics_by_task.get("sts", {})
    whole["baseline_HR"] = sts.get("supine_mean_HR", np.nan)
    whole["plateau_HR"] = sts.get("plateau_mean_HR", np.nan)
    whole["delta_HR"] = sts.get("delta_HR", np.nan)

    # In your STS script you output MAP/SYS/DIA/PP; compile_whole uses baseline_BP etc.
    whole["baseline_BP"] = sts.get("supine_mean_MAP", np.nan)
    whole["plateau_BP"] = sts.get("plateau_mean_MAP", np.nan)
    whole["delta_BP"] = sts.get("delta_MAP", np.nan)

    # Valsalva mapping (NOW: only ratio needed)
    val = metrics_by_task.get("valsalva", {})
    whole["Valsalva_ratio"] = val.get("valsalva_ratio", np.nan)

    # Breathing mapping (your breathing script outputs mean_max/min, diff, ratio)
    br = metrics_by_task.get("breathing", {})
    # In your MATLAB you fill EIratio and delta_HR_responses:
    # we'll map EI ratio -> mean_max/mean_min ratio; delta -> diff
    whole["E_I_ratio"] = br.get("ratio", np.nan)
    whole["delta_HR_responses"] = br.get("diff_bpm", np.nan)

    # Spirometry mapping
    sp = metrics_by_task.get("spirometry", {})
    whole["FEV1"] = sp.get("FEV1_max", np.nan)
    whole["FVC"] = sp.get("FVC_max", np.nan)
    whole["PEF"] = sp.get("PEF_max", np.nan)
    # your Word table expects FVC/FEV1; you also said compute from extracted values
    fev1 = whole["FEV1"]
    fvc = whole["FVC"]
    whole["FVC_over_FEV1"] = (fvc / fev1) if (np.isfinite(fev1) and fev1 != 0) else np.nan
    whole["FEV1_over_FVC"] = (fev1 / fvc) if (np.isfinite(fvc) and fvc != 0) else np.nan

    # Paths to figures you insert in compile_whole
    # (keep these here so MATLAB can just read them)
    figs = {
        "STS_HR_MAP": str(base / "sts" / "STS_HR_MAP_plot.png"),  # if your STS script writes there
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


def main():
    ap = argparse.ArgumentParser(
        description="Run ALL tasks for one subject/session (rest, STS, valsalva, breathing, spirometry) + merge."
    )

    ap.add_argument("--root", default="/export02/projects/LCS/01_physio", help="Root folder containing sub-*")
    ap.add_argument("--sub", required=True, help="Subject code like 2062")
    ap.add_argument("--ses", default="1", help="Session number like 1 or 2")
    ap.add_argument("--out_root", default="derived", help="Output root folder (relative or absolute)")
    ap.add_argument("--one_based", action="store_true", help="Interpret channel numbers as 1-based")

    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["rest", "sts", "valsalva", "breathing", "spirometry"],
        help="Tasks to skip",
    )

    ap.add_argument("--rest_ecg_ch", type=int, default=4, help="Rest ECG channel (default 6)")
    ap.add_argument("--rest_bp_ch", type=int, default=10, help="Rest BP channel (default 10)")

    ap.add_argument("--sts_ecg_ch", type=int, default=4, help="STS ECG channel (default 4)")
    ap.add_argument("--sts_bp_ch", type=int, default=10, help="STS BP channel (default 10)")

    ap.add_argument("--val_ecg_ch", type=int, default=4, help="Valsalva ECG channel (default 4)")
    ap.add_argument("--breath_ecg_ch", type=int, default=4, help="Breathing ECG channel (default 4)")

    ap.add_argument("--force_ppg", action="store_true", help="Force PPG for valsalva+breathing")
    ap.add_argument("--ppg_ch", type=int, default=5, help="PPG channel (default 5)")

    ap.add_argument("--debug", action="store_true", help="Enable debug plots where supported")

    args = ap.parse_args()

    py = sys.executable
    out_root = Path(args.out_root)
    root = Path(args.root)

    common = []
    if args.one_based:
        common.append("--one_based")

    # REST
    if "rest" not in args.skip:
        cmd = [
            py, script_path("run_rest_acq.py"),
            "--root", args.root, "--sub", args.sub, "--ses", args.ses,
            *common,
            "--save", "--out_root", args.out_root,
            "--ecg_ch", str(args.rest_ecg_ch),
            "--bp_ch", str(args.rest_bp_ch),
        ]
        run(cmd)
    else:
        print("[SKIP] rest")

    # STS
    if "sts" not in args.skip:
        cmd = [
            py, script_path("run_sts_acq.py"),
            "--root", args.root, "--sub", args.sub, "--ses", args.ses,
            *common,
            "--save", "--out_root", args.out_root,
            "--ecg_ch", str(args.sts_ecg_ch),
            "--bp_ch", str(args.sts_bp_ch),
        ]
        run(cmd)
    else:
        print("[SKIP] sts")

    # VALSALVA (optional)
    if "valsalva" not in args.skip:
        if valsalva_acq_exists(root, args.sub, args.ses):
            cmd = [
                py, script_path("run_valsalva_acq.py"),
                "--root", args.root, "--sub", args.sub, "--ses", args.ses,
                *common,
                "--save", "--out_root", args.out_root,
                "--ecg_ch", str(args.val_ecg_ch),
            ]
            if args.force_ppg:
                cmd += ["--force_ppg", "--ppg_ch", str(args.ppg_ch)]
            if args.debug:
                cmd += ["--debug_plot", "--signal_debug_plot"]
            # allow_fail=False because if file exists it should work; if it fails you want to notice
            run(cmd)
        else:
            print("[WARN] valsalva file not found — creating placeholder valsalva outputs.")
            make_placeholder_valsalva(out_root, args.sub, args.ses)
    else:
        print("[SKIP] valsalva")

    # BREATHING
    if "breathing" not in args.skip:
        cmd = [
            py, script_path("run_breathing_acq.py"),
            "--root", args.root, "--sub", args.sub, "--ses", args.ses,
            *common,
            "--save", "--out_root", args.out_root,
            "--ecg_ch", str(args.breath_ecg_ch),
        ]
        if args.force_ppg:
            cmd += ["--force_ppg", "--ppg_ch", str(args.ppg_ch)]
        if args.debug:
            cmd += ["--debug_plot"]
        run(cmd)
    else:
        print("[SKIP] breathing")

    # SPIROMETRY
    if "spirometry" not in args.skip:
        cmd = [
            py, script_path("run_spirometry_extract.py"),
            "--sub", args.sub, "--ses", args.ses,
            "--out_root", args.out_root,
        ]
        run(cmd)
    else:
        print("[SKIP] spirometry")

    # MERGE
    merge_to_all_metrics(out_root, args.sub, args.ses)

    print("\n[DONE] All requested tasks finished.")
    print(f"[INFO] Outputs under: {out_root / f'sub-{args.sub}' / f'ses-{args.ses}'}")


if __name__ == "__main__":
    main()
