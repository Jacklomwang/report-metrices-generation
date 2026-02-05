#!/usr/bin/env python3
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import streamlit as st
import numpy as np

# Optional (nice-to-have) to preview metrics
try:
    import scipy.io as sio
    import pandas as pd
except Exception:
    sio = None
    pd = None


# -----------------------------
# Helpers
# -----------------------------

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

def _safe_cell(x, maxlen: int = 200):
    """Convert any MATLAB-ish / numpy-ish / mixed object into an Arrow-safe value (string or scalar)."""
    if x is None:
        return ""

    # unwrap 0-d arrays
    if isinstance(x, np.ndarray) and x.ndim == 0:
        try:
            x = x.item()
        except Exception:
            pass

    # arrays -> short summary (don’t dump huge vectors into the table)
    if isinstance(x, np.ndarray):
        if x.size == 1:
            try:
                return _safe_cell(x.item(), maxlen=maxlen)
            except Exception:
                pass
        return f"<array shape={x.shape} dtype={x.dtype}>"

    # bytes -> str
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", errors="replace")
        except Exception:
            return str(x)

    # pure scalars are fine
    if isinstance(x, (int, float, np.integer, np.floating, bool)):
        return x

    # everything else -> string (trim long ones)
    s = str(x)
    if len(s) > maxlen:
        s = s[:maxlen] + "…"
    return s


def scripts_dir() -> Path:
    # streamlit_app/app.py -> project_root/streamlit_app/app.py
    # project_root/scripts/...
    return Path(__file__).resolve().parents[1] / "scripts"


def script_path(name: str) -> str:
    p = scripts_dir() / name
    if not p.exists():
        raise FileNotFoundError(f"Missing script: {p}")
    return str(p)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    """Run command and capture stdout+stderr (combined)."""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def load_mat_struct(mat_path: Path) -> dict:
    """
    Load a .mat into a simple dict for display.
    Supports both:
      - {"whole": struct}
      - {"metrics_by_task": struct}
      - flat dicts
    """
    if sio is None:
        return {"error": "scipy not available in this env (pip install scipy)."}
    if not mat_path.exists():
        return {"error": f"File not found: {mat_path}"}

    m = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    def to_py(obj):
        # Convert MATLAB structs to dict recursively
        if hasattr(obj, "_fieldnames"):
            out = {}
            for f in obj._fieldnames:
                out[f] = to_py(getattr(obj, f))
            return out
        if isinstance(obj, (list, tuple)):
            return [to_py(x) for x in obj]
        return obj

    # drop private keys
    m2 = {k: v for k, v in m.items() if not k.startswith("__")}

    # prefer known containers
    if "whole" in m2:
        return {"whole": to_py(m2["whole"])}
    if "metrics_by_task" in m2:
        return {"metrics_by_task": to_py(m2["metrics_by_task"])}

    return {k: to_py(v) for k, v in m2.items()}


def dict_to_table(d: dict, prefix: str = "") -> "pd.DataFrame | None":
    """
    Flatten nested dict into a 2-col dataframe: key/value.
    Returns a DataFrame (never renders), or None if pandas unavailable.
    Forces value column to string to avoid Arrow dtype inference issues.
    """
    if pd is None:
        return None

    rows: list[tuple[str, object]] = []

    def rec(x, pfx: str):
        if isinstance(x, dict):
            for k, v in x.items():
                rec(v, f"{pfx}{k}.")
        elif isinstance(x, (list, tuple)):
            if len(x) > 50:
                rows.append((pfx[:-1], f"<list len={len(x)}>"))
            else:
                for i, v in enumerate(x):
                    rec(v, f"{pfx}{i}.")
        else:
            if isinstance(x, np.ndarray) and getattr(x, "size", 0) > 50:
                rows.append((pfx[:-1], f"<array shape={x.shape} dtype={x.dtype}>"))
            else:
                rows.append((pfx[:-1], x))

    rec(d, prefix)

    df = pd.DataFrame(rows, columns=["key", "value"])

    # Force consistent Arrow-friendly dtype
    df["key"] = df["key"].astype("string")
    df["value"] = df["value"].map(lambda x: str(_safe_cell(x))).astype("string")

    return df


def parse_patterns_csv(s: str) -> list[str]:
    # "trigger,trig, marker" -> ["trigger","trig","marker"]
    parts = [p.strip() for p in (s or "").split(",")]
    return [p for p in parts if p]


def show_status(task_key: str):
    rc = st.session_state.get(f"rc_{task_key}", None)
    if rc is None:
        st.caption("Status: not run yet.")
    elif rc == 0:
        st.success("Status: ✅ success")
    else:
        st.error(f"Status: ❌ failed (rc={rc})")


def show_log(task_key: str):
    log_txt = st.session_state.get(f"log_{task_key}", "")
    with st.expander("Run log", expanded=False):
        if log_txt:
            st.code(log_txt)
        else:
            st.caption("No output yet.")


def show_figures(fig_items: list[tuple[Path, str]]):
    if not fig_items:
        st.caption("No figures configured for this task.")
        return
    for p, cap in fig_items:
        if p.exists():
            st.image(str(p), caption=cap, width="stretch")
        else:
            st.info(f"Figure not found yet: {p}")


def show_metrics(mat_path: Path | None):
    if mat_path is None:
        st.caption("No metrics file configured for this task.")
        return
    if not mat_path.exists():
        st.info(f"Metrics not found yet: {mat_path}")
        return

    d = load_mat_struct(mat_path)
    df = dict_to_table(d)
    if df is None:
        st.json(d)
    else:
        st.dataframe(df, width="stretch", hide_index=True)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LCS Physio Metrics Runner", layout="wide")
st.title("LCS Physio Metrics Runner (REST / STS / Valsalva / Breathing / Spirometry)")

project_root = Path(__file__).resolve().parents[1]
py = sys.executable


# -----------------------------
# Global subject / paths (top)
# -----------------------------
with st.container(border=True):
    st.subheader("Subject / Paths")
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1], vertical_alignment="center")

    with c1:
        root = st.text_input("Physio root", value="/export02/projects/LCS/01_physio", key="root")
    with c2:
        out_root = st.text_input("Output root", value="derived", key="out_root")
    with c3:
        sub = st.text_input("Subject code", value="2062", key="sub")
    with c4:
        ses = st.text_input("Session", value="1", key="ses")
    with c5:
        one_based = st.checkbox("Channels 1-based", value=True, key="one_based")

sub_id = f"sub-{sub}"
ses_id = f"ses-{ses}"
base_out = Path(out_root) / sub_id / ses_id

# Where figures/metrics are expected (based on your scripts)
paths = {
    "rest_mat": base_out / "rest" / "rest_metrics.mat",
    "sts_mat": base_out / "sts" / "sts_metrics.mat",
    "rest_hr_fig": base_out / "rest" / "resting_hr.png",
    "rest_bp_fig": base_out / "rest" / "resting_BP.png",

    "val_mat": base_out / "valsalva" / "valsalva_metrics.mat",
    "breath_mat": base_out / "breathing" / "breathing_metrics.mat",
    "spiro_mat": base_out / "spirometry" / "spirometry_metrics.mat",
    "all_mat": base_out / f"{sub_id}_{ses_id}_all_metrics.mat",

    "sts_fig": base_out / "sts" / "STS_HR_MAP.png",
    "val_fig": base_out / "valsalva" / "valsalva_best_rep_hr.png",
    "val_debug_fig": base_out / "valsalva" / "valsalva_debug_full_hr.png",
    "breath_fig": base_out / "breathing" / "deep_breathing_HR_plot.png",
    "spiro_fig": base_out / "spirometry" / "spirometry_summary.png",
}

def common_flags():
    return ["--one_based"] if one_based else []


# ============================================================
# REST SECTION
# ============================================================
with st.container(border=True):
    st.markdown("## REST")

    pcol, runcol, statcol = st.columns([2, 1, 2], vertical_alignment="center")

    with pcol:
        st.caption("Parameters")
        rest_ecg_ch = st.number_input("REST ECG channel", min_value=1, value=6, step=1, key="rest_ecg_ch")
        rest_bp_ch  = st.number_input("REST BP channel",  min_value=1, value=10, step=1, key="rest_bp_ch")

    with runcol:
        if st.button("Run REST", key="btn_rest", width="stretch"):
            cmd = [
                py, script_path("run_rest_acq.py"),
                "--root", root, "--sub", sub, "--ses", ses,
                *common_flags(),
                "--save",
                "--out_root", out_root,
                "--ecg_ch", str(int(rest_ecg_ch)),
                "--bp_ch", str(int(rest_bp_ch)),
            ]
            rc, out = run_cmd(cmd, cwd=project_root)
            st.session_state["rc_rest"] = rc
            st.session_state["log_rest"] = out

    with statcol:
        show_status("rest")

    show_log("rest")
    st.divider()

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Figures")
        show_figures([
            (paths["rest_hr_fig"], "REST: Derived HR"),
            (paths["rest_bp_fig"], "REST: Derived SBP/DBP/MBP"),
        ])

    with right:
        st.subheader("Metrics")
        show_metrics(paths["rest_mat"])


# ============================================================
# STS SECTION
# ============================================================
with st.container(border=True):
    st.markdown("## STS")

    pcol, runcol, statcol = st.columns([2, 1, 2], vertical_alignment="center")

    with pcol:
        st.caption("Parameters")
        sts_ecg_ch = st.number_input("STS ECG channel", min_value=1, value=4, step=1, key="sts_ecg_ch")
        sts_bp_ch  = st.number_input("STS BP channel",  min_value=1, value=10, step=1, key="sts_bp_ch")
        # NEW: Height Adjustment Input
        use_height_corr = st.toggle("Apply Height Adjustment", value=False)
        sts_height = 0.0
        if use_height_corr:
            sts_height = st.number_input("Subject Height (cm)", min_value=0.0, max_value=250.0, value=170.0)
            st.caption(f"Correction: -{0.4 * sts_height:.1f} mmHg to standing MAP")
    with runcol:
            if st.button("Run STS", key="btn_sts", width="stretch"):
                cmd = [
                    py, script_path("run_sts_acq.py"),
                    "--root", root, "--sub", sub, "--ses", ses,
                    *common_flags(),
                    "--save",
                    "--out_root", out_root,
                    "--ecg_ch", str(int(sts_ecg_ch)),
                    "--bp_ch", str(int(sts_bp_ch)),
                    "--height", str(sts_height),  # <--- ADD THIS LINE
                ]
                rc, out = run_cmd(cmd, cwd=project_root)
                st.session_state["rc_sts"] = rc
                st.session_state["log_sts"] = out

    with statcol:
        show_status("sts")

    show_log("sts")
    st.divider()

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Figures")
        show_figures([(paths["sts_fig"], "STS: HR/MAP")])
    with right:
        st.subheader("Metrics")
        show_metrics(paths["sts_mat"])


# ============================================================
# VALSALVA SECTION
# ============================================================
with st.container(border=True):
    st.markdown("## Valsalva")

    # Parameters row (more knobs, so use expander)
    with st.expander("Parameters", expanded=True):
        cA, cB, cC = st.columns(3, gap="large")

        with cA:
            val_ecg_ch = st.number_input("ECG channel (--ecg_ch)", min_value=1, value=4, step=1, key="val_ecg_ch")
            val_trig_ch = st.number_input(
                "Trigger channel (--trig_ch) [0 = auto]",
                min_value=0, value=0, step=1, key="val_trig_ch"
            )
            trig_patterns_csv = st.text_input(
                "Trigger patterns (--trig_patterns), comma-separated",
                value="trigger,trig,marker,event,sync",
                key="val_trig_patterns",
            )

        with cB:
            val_ppg_ch = st.number_input("PPG channel (--ppg_ch)", min_value=1, value=5, step=1, key="val_ppg_ch")
            val_force_ppg = st.checkbox("Force PPG (--force_ppg)", value=False, key="val_force_ppg")
            val_fallback_ppg = st.checkbox("Fallback to PPG if ECG bad (--fallback_ppg)", value=False, key="val_fallback_ppg")

        with cC:
            val_hr_smooth_sec = st.number_input(
                "HR smoothing sec for max/min (--hr_smooth_sec)",
                min_value=0.0, value=0.0, step=0.5, key="val_hr_smooth_sec"
            )
            val_debug_plot = st.checkbox("Save debug plot (--debug_plot)", value=True, key="val_debug_plot")
            val_ecg_debug_plot = st.checkbox("Save ECG/PPG peaks debug plot (--ecg_debug_plot)", value=False, key="val_ecg_debug_plot")

    runcol, statcol = st.columns([1, 2], vertical_alignment="center")
    with runcol:
        if st.button("Run VALSALVA", key="btn_val", width="stretch"):
            cmd = [
                py, script_path("run_valsalva_acq.py"),
                "--root", root, "--sub", sub, "--ses", ses,
                *common_flags(),
                "--save",
                "--out_root", out_root,
                "--ecg_ch", str(int(val_ecg_ch)),
                "--ppg_ch", str(int(val_ppg_ch)),
            ]

            # trigger channel: if 0, omit to auto-detect
            if int(val_trig_ch) > 0:
                cmd += ["--trig_ch", str(int(val_trig_ch))]

            # patterns
            pats = parse_patterns_csv(trig_patterns_csv)
            if pats:
                cmd += ["--trig_patterns", *pats]

            # smoothing
            cmd += ["--hr_smooth_sec", str(float(val_hr_smooth_sec))]

            # ppg selection
            if val_force_ppg:
                cmd += ["--force_ppg"]
            if val_fallback_ppg:
                cmd += ["--fallback_ppg"]

            # debug plots
            if val_debug_plot:
                cmd += ["--debug_plot"]
            if val_ecg_debug_plot:
                cmd += ["--ecg_debug_plot"]

            rc, out = run_cmd(cmd, cwd=project_root)
            st.session_state["rc_val"] = rc
            st.session_state["log_val"] = out

    with statcol:
        show_status("val")

    show_log("val")
    st.divider()

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Figures")
        show_figures([
            (paths["val_fig"], "Valsalva: best repetition HR"),
            (paths["val_debug_fig"], "Valsalva: debug full HR"),
        ])
    with right:
        st.subheader("Metrics")
        show_metrics(paths["val_mat"])


# ============================================================
# BREATHING SECTION
# ============================================================
with st.container(border=True):
    st.markdown("## Breathing")

    with st.expander("Parameters", expanded=True):
        cA, cB, cC = st.columns(3, gap="large")

        with cA:
            breath_ecg_ch = st.number_input("ECG channel (--ecg_ch)", min_value=1, value=4, step=1, key="breath_ecg_ch")
            breath_start_min = st.number_input("Window start (min) (--win_start_min)", min_value=0.0, value=8.0, step=0.5, key="breath_start_min")
            breath_end_min = st.number_input("Window end (min) (--win_end_min)", min_value=0.0, value=9.0, step=0.5, key="breath_end_min")

        with cB:
            breath_ppg_ch = st.number_input("PPG channel (--ppg_ch)", min_value=1, value=5, step=1, key="breath_ppg_ch")
            breath_force_ppg = st.checkbox("Force PPG (--force_ppg)", value=False, key="breath_force_ppg")

        with cC:
            breath_debug_plot = st.checkbox("Save debug plot (--debug_plot)", value=True, key="breath_debug_plot")

    runcol, statcol = st.columns([1, 2], vertical_alignment="center")
    with runcol:
        if st.button("Run BREATHING", key="btn_breath", width="stretch"):
            cmd = [
                py, script_path("run_breathing_acq.py"),
                "--root", root, "--sub", sub, "--ses", ses,
                *common_flags(),
                "--save",
                "--out_root", out_root,
                "--ecg_ch", str(int(breath_ecg_ch)),
                "--win_start_min", str(float(breath_start_min)),
                "--win_end_min", str(float(breath_end_min)),
                "--ppg_ch", str(int(breath_ppg_ch)),
            ]
            if breath_force_ppg:
                cmd += ["--force_ppg"]

            rc, out = run_cmd(cmd, cwd=project_root)
            st.session_state["rc_breath"] = rc
            st.session_state["log_breath"] = out

    with statcol:
        show_status("breath")

    show_log("breath")
    st.divider()

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Figures")
        show_figures([(paths["breath_fig"], "Breathing: HR window with peaks/troughs")])
    with right:
        st.subheader("Metrics")
        show_metrics(paths["breath_mat"])


# ============================================================
# SPIROMETRY SECTION
# ============================================================
with st.container(border=True):
    st.markdown("## Spirometry")

    runcol, statcol = st.columns([1, 2], vertical_alignment="center")
    with runcol:
        if st.button("Run SPIROMETRY", key="btn_spiro", width="stretch"):
            cmd = [
                py, script_path("run_spirometry_extract.py"),
                "--sub", sub, "--ses", ses,
                "--out_root", out_root,
            ]
            rc, out = run_cmd(cmd, cwd=project_root)
            st.session_state["rc_spiro"] = rc
            st.session_state["log_spiro"] = out

    with statcol:
        show_status("spiro")

    show_log("spiro")
    st.divider()

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Figures")
        show_figures([(paths["spiro_fig"], "Spirometry: summary")])
    with right:
        st.subheader("Metrics")
        show_metrics(paths["spiro_mat"])


# ============================================================
# ALL (META) SECTION
# ============================================================
# ============================================================
# MERGE RESULTS SECTION
# ============================================================
# ============================================================
# MERGE RESULTS SECTION (runs external script)
# ============================================================
with st.container(border=True):
    st.markdown("## Merge task results")
    st.caption("This merges per-task outputs into the single all-metrics .mat (no task re-processing).")

    runcol, statcol = st.columns([1, 2], vertical_alignment="center")

    with runcol:
        if st.button("merge_tasks_results", key="btn_merge", width="stretch"):
            cmd = [
                py,
                script_path("merge_subject_all_metrics_only.py"),
                "--out_root", out_root,
                "--sub", sub,
                "--ses", ses,
            ]

            # OPTIONAL: only if you want placeholder valsalva behavior
            # cmd += [
            #     "--root", root,
            #     "--make_valsalva_placeholder_if_missing",
            # ]

            rc, out = run_cmd(cmd, cwd=project_root)
            st.session_state["rc_merge"] = rc
            st.session_state["log_merge"] = out

    with statcol:
        show_status("merge")   # your helper: reads st.session_state["rc_merge"]

    show_log("merge")         # your helper: reads st.session_state["log_merge"]
    st.divider()

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Outputs")
        st.caption("Per-task figures are shown above in each task section.")
        st.caption(f"Expected merged MAT: {paths['all_mat']}")
    with right:
        st.subheader("Merged metrics")
        show_metrics(paths["all_mat"])
