from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import bioread
import neurokit2 as nk

# ---- NumPy 2.x compatibility for NeuroKit2
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bp_processing import (
    detect_calibration_artifacts,
    detect_bp_peaks_custom,
    detect_bp_troughs,
    compute_bp_derived_from_peaks,
)


def save_resting_hr_figure(out_png: Path, t: np.ndarray, hr_ts: np.ndarray):
    import os
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 4))
    ax = plt.gca()
    ax.plot(t, hr_ts, linewidth=0.8, label="HR (bpm)")
    ax.set_title("Resting: Derived Heart Rate")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HR (bpm)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_resting_bp_figure(out_png: Path, t: np.ndarray, sbp: np.ndarray, dbp: np.ndarray, mbp: np.ndarray):
    import os
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 4))
    ax = plt.gca()
    ax.plot(t, sbp, linewidth=0.8, label="SBP")
    ax.plot(t, dbp, linewidth=0.8, label="DBP")
    ax.plot(t, mbp, linewidth=0.9, label="MBP")
    ax.set_title("Resting: Derived Blood Pressure (SBP/DBP/MBP)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (mmHg)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)





def build_rest_acq_path(root: Path, sub_code: str, ses_num: str) -> Path:
    sub_id = f"sub-{sub_code}"
    ses_id = f"ses-{ses_num}"
    ses_dir = root / sub_id / ses_id

    expected = ses_dir / f"{sub_id}_{ses_id}_task-rest_physio.acq"
    if expected.exists():
        return expected

    cands = sorted(ses_dir.glob(f"{sub_id}_{ses_id}_task-rest*physio*.acq"))
    if cands:
        return cands[0]

    cands = sorted(ses_dir.glob("*task-rest*physio*.acq"))
    if cands:
        return cands[0]

    cands = sorted(ses_dir.glob("*rest*.acq"))
    if cands:
        return cands[0]

    raise FileNotFoundError(f"No resting .acq found under: {ses_dir}")


def pick_channel_by_index(channels, ch_num: int, one_based: bool):
    idx = ch_num - 1 if one_based else ch_num
    if idx < 0 or idx >= len(channels):
        raise RuntimeError(
            f"Channel out of range: requested {ch_num} ({'1-based' if one_based else '0-based'}), "
            f"but file has {len(channels)} channels."
        )
    return channels[idx], idx


def plot_qc(sub_id: str, ses_id: str, fs: float,
            ecg: np.ndarray, rpeaks: np.ndarray,
            bp_raw: np.ndarray, bp_filt: np.ndarray,
            sys_peaks: np.ndarray, dia_troughs: np.ndarray,
            cal_idx1: list[int], cal_idx2: list[int],
            max_seconds: float = 60.0,
            outdir: Path | None = None):
    """
    QC plot: first `max_seconds` seconds of ECG and BP with detected points.
    If display not available, save png and print path.
    """
    import os
    import matplotlib.pyplot as plt

    n = len(ecg)
    nmax = min(n, int(max_seconds * fs))
    t = np.arange(nmax) / fs

    ecg_seg = ecg[:nmax]
    bp_raw_seg = bp_raw[:nmax]
    bp_filt_seg = bp_filt[:nmax]

    rpeaks_seg = rpeaks[(rpeaks >= 0) & (rpeaks < nmax)]
    sys_seg = sys_peaks[(sys_peaks >= 0) & (sys_peaks < nmax)]
    dia_seg = dia_troughs[(dia_troughs >= 0) & (dia_troughs < nmax)]

    fig = plt.figure(figsize=(14, 8))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t, ecg_seg, linewidth=0.8)
    if len(rpeaks_seg):
        ax1.plot(rpeaks_seg / fs, ecg[rpeaks_seg], "rx", markersize=6, label="R-peaks")
        ax1.legend(loc="upper right")
    ax1.set_title(f"{sub_id} {ses_id} - ECG (first {max_seconds:.0f}s)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("ECG")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(t, bp_raw_seg, linewidth=0.6, label="BP raw")
    ax2.plot(t, bp_filt_seg, linewidth=0.9, label="BP filt (40Hz)")
    if len(sys_seg):
        ax2.plot(sys_seg / fs, bp_filt[sys_seg], "r^", markersize=5, label="SBP peaks")
    if len(dia_seg):
        ax2.plot(dia_seg / fs, bp_filt[dia_seg], "bv", markersize=5, label="DBP troughs")

    # shade calibration segments (only those that overlap nmax)
    for s, e in zip(cal_idx1, cal_idx2):
        if s >= nmax:
            continue
        ss = max(0, s) / fs
        ee = min(nmax, e) / fs
        ax2.axvspan(ss, ee, alpha=0.15)

    ax2.set_title(f"{sub_id} {ses_id} - BP with detections (first {max_seconds:.0f}s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("BP")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()

    # Try show; if no DISPLAY, save fallback
    has_display = bool(os.environ.get("DISPLAY"))
    if has_display:
        plt.show()
    else:
        if outdir is None:
            outdir = Path("outputs")
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"qc_{sub_id}_{ses_id}.png"
        fig.savefig(outpath, dpi=150)
        print(f"[INFO] No DISPLAY detected; saved QC plot to: {outpath}")


def main():
    ap = argparse.ArgumentParser(description="Resting-state: derive indices from .acq")
    ap.add_argument("--root", default="/export02/projects/LCS/01_physio")
    ap.add_argument("--sub", required=True, help="Subject code like 2062")
    ap.add_argument("--ses", default="1", help="Session number like 1 or 2")

    ap.add_argument("--ecg_ch", type=int, required=True, help="ECG channel number")
    ap.add_argument("--bp_ch", type=int, required=True, help="BP channel number")
    ap.add_argument("--one_based", action="store_true", help="Interpret channel numbers as 1-based")

    ap.add_argument("--bp_prominence", type=float, default=20.0)
    ap.add_argument("--cal_thr", type=float, default=0.1)
    ap.add_argument("--cal_min_dur", type=float, default=0.65)

    ap.add_argument("--no_save", action="store_true", help="Do not save any .mat files")
    ap.add_argument("--out_root", default="derived", help="Root folder to save outputs under this project")
    ap.add_argument("--save", action="store_true", help="Save metrics to a single .mat file")

    ap.add_argument("--plot", action="store_true", help="Show QC plots (falls back to saving png if no DISPLAY)")
    ap.add_argument("--plot_seconds", type=float, default=60.0, help="Seconds to plot for QC")
    ap.add_argument("--plot_outdir", default="outputs", help="Where to save plot if no DISPLAY")

    args = ap.parse_args()

    root = Path(args.root)
    sub_id = f"sub-{args.sub}"
    ses_id = f"ses-{args.ses}"
    out_root = Path(args.out_root)
    task_out = out_root / sub_id / ses_id / "rest"
    task_out.mkdir(parents=True, exist_ok=True)
    out_mat = task_out / "rest_metrics.mat"


    acq_path = build_rest_acq_path(root, args.sub, args.ses)
    print(f"[INFO] Subject: {sub_id}  Session: {ses_id}")
    print(f"[INFO] Rest ACQ: {acq_path}")

    d = bioread.read_file(str(acq_path))
    fs = float(d.samples_per_second)
    print(f"[INFO] Sampling rate: {fs} Hz  |  Channels: {len(d.channels)}")

    ecg_ch, ecg_idx = pick_channel_by_index(d.channels, args.ecg_ch, args.one_based)
    bp_ch, bp_idx = pick_channel_by_index(d.channels, args.bp_ch, args.one_based)
    print(f"[INFO] ECG channel index={ecg_idx} name={getattr(ecg_ch,'name','')}")
    print(f"[INFO]  BP channel index={bp_idx} name={getattr(bp_ch,'name','')}")

    ecg = np.asarray(ecg_ch.data, dtype=float)
    bp_raw = np.asarray(bp_ch.data, dtype=float)

    # ---- BP
    idx1, idx2, bp_filt, dp = detect_calibration_artifacts(
        bp_raw, fs, thr=args.cal_thr, min_duration_sec=args.cal_min_dur, debug=False
    )
    mask_cal = np.zeros(len(bp_filt), dtype=bool)
    for s, e in zip(idx1, idx2):
        mask_cal[s:e + 1] = True

    peaks = detect_bp_peaks_custom(
        bp_filt, fs, mask_cal=mask_cal, prominence=args.bp_prominence, min_distance_sec=0.4
    )
    troughs = detect_bp_troughs(bp_filt, fs, peaks)

    t = np.arange(len(bp_filt)) / fs
    derived = compute_bp_derived_from_peaks(bp_filt, t, peaks, troughs)

    map_interp = derived["MBP"]
    systolic_interp = derived["SBP"]
    diastolic_interp = derived["DBP"]

    mean_MAP = float(np.nanmean(map_interp)) if len(map_interp) else np.nan
    mean_sysBP = float(np.nanmean(systolic_interp)) if len(systolic_interp) else np.nan
    mean_diaBP = float(np.nanmean(diastolic_interp)) if len(diastolic_interp) else np.nan
    mean_pulseBP = float(mean_sysBP - mean_diaBP) if np.isfinite(mean_sysBP) and np.isfinite(mean_diaBP) else np.nan

    # ---- ECG/HRV
    signals_ecg, info = nk.ecg_process(ecg, sampling_rate=fs)
    rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)

    # sample-aligned HR time series (same length as ECG)
    hr_ts = np.asarray(signals_ecg["ECG_Rate"], dtype=float) if "ECG_Rate" in signals_ecg else np.full(len(ecg), np.nan)
    t_ecg = np.arange(len(ecg)) / fs

    if len(rpeaks) < 3:
        mean_RR = np.nan
        mean_HR = np.nan
        RMSSD_ms = np.nan
        LF_HF_ratio = np.nan
    else:
        rr_s = np.diff(rpeaks) / fs
        rr_s = rr_s[(rr_s >= 0.3) & (rr_s <= 2.0)]
        mean_RR = float(np.nanmean(rr_s)) if len(rr_s) else np.nan
        mean_HR = float(60.0 / mean_RR) if np.isfinite(mean_RR) and mean_RR > 0 else np.nan

        hrv_time = nk.hrv_time(rpeaks, sampling_rate=fs, show=False)
        RMSSD_ms = float(hrv_time["HRV_RMSSD"].iloc[0]) if "HRV_RMSSD" in hrv_time else np.nan

        hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=fs, show=False)
        lf = float(hrv_freq["HRV_LF"].iloc[0]) if "HRV_LF" in hrv_freq else np.nan
        hf = float(hrv_freq["HRV_HF"].iloc[0]) if "HRV_HF" in hrv_freq else np.nan
        LF_HF_ratio = float(lf / hf) if np.isfinite(lf) and np.isfinite(hf) and hf != 0 else np.nan

    # ---- Print derived indices
    print("\n===== REST DERIVED INDICES =====")
    print(f"BP:  mean_MAP={mean_MAP:.2f}  mean_sysBP={mean_sysBP:.2f}  mean_diaBP={mean_diaBP:.2f}  mean_pulseBP={mean_pulseBP:.2f}")
    print(f"BP:  n_peaks={len(peaks)}  n_troughs={len(troughs)}  cal_segments={len(idx1)}")
    print(f"ECG: mean_RR={mean_RR:.4f} s  mean_HR={mean_HR:.2f} bpm  RMSSD={RMSSD_ms:.2f} ms  LF/HF={LF_HF_ratio:.2f}")
    print(f"ECG: n_rpeaks={len(rpeaks)}")

    # ---- QC Plot
    if args.plot:
        plot_qc(
            sub_id=sub_id, ses_id=ses_id, fs=fs,
            ecg=ecg, rpeaks=rpeaks,
            bp_raw=bp_raw, bp_filt=bp_filt,
            sys_peaks=peaks, dia_troughs=troughs,
            cal_idx1=idx1, cal_idx2=idx2,
            max_seconds=args.plot_seconds,
            outdir=Path(args.plot_outdir),
        )
    # ---- Save figures (requested)
    hr_png = task_out / "resting_hr.png"
    bp_png = task_out / "resting_BP.png"

     # BP uses t derived from bp_filt
    # (you already defined t = np.arange(len(bp_filt))/fs earlier)
    if args.save:
        save_resting_hr_figure(hr_png, t_ecg, hr_ts)
        save_resting_bp_figure(bp_png, t, systolic_interp, diastolic_interp, map_interp)
        print(f"[OK] Saved resting HR figure: {hr_png}")
        print(f"[OK] Saved resting BP figure: {bp_png}")
    if args.no_save:
        print("\n[INFO] --no_save set: not writing any .mat files.")
        return

    # Saving intentionally omitted for now.
    if args.save:
        from scipy.io import savemat

        metrics = {
            # Scalars (store as float)
            "mean_MAP": mean_MAP,
            "mean_sysBP": mean_sysBP,
            "mean_diaBP": mean_diaBP,
            "mean_pulseBP": mean_pulseBP,
            "mean_RR": mean_RR,
            "mean_HR": mean_HR,
            "RMSSD_ms": RMSSD_ms,
            "LF_HF": LF_HF_ratio,

            # Counts
            "n_peaks": int(len(peaks)),
            "n_troughs": int(len(troughs)),
            "n_rpeaks": int(len(rpeaks)),
            "n_cal_segments": int(len(idx1)),

            # Indices arrays (good for debugging)
            "bp_peaks": peaks.astype(np.int32),
            "bp_troughs": troughs.astype(np.int32),
            "rpeaks": rpeaks.astype(np.int32),
            "cal_idx1": np.array(idx1, dtype=np.int32),
            "cal_idx2": np.array(idx2, dtype=np.int32),

            # Metadata
            "fs": float(fs),
            "sub_id": sub_id,
            "ses_id": ses_id,
            "acq_path": str(acq_path),
        }

        savemat(str(out_mat), metrics, do_compression=True)
        print(f"[OK] Saved metrics bundle: {out_mat}")

if __name__ == "__main__":
    main()
