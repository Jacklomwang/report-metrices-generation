#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import bioread
import neurokit2 as nk
from scipy.signal import find_peaks


# ----------------------------
# Path / file finding
# ----------------------------
def build_breathing_acq_path(root: Path, sub_code: str, ses_num: str) -> Path:
    """
    Expected layout:
      {root}/sub-2066/ses-1/sub-2066_ses-1_task-breath_physio.acq

    Falls back to glob patterns if naming differs (deep breathing, etc.).
    """
    sub_id = f"sub-{sub_code}"
    ses_id = f"ses-{ses_num}"
    ses_dir = root / sub_id / ses_id

    patterns = [
        # Exact expected (your case)
        f"{sub_id}_{ses_id}_task-breath_physio.acq",

        # Common variants
        f"{sub_id}_{ses_id}_task-breath*physio*.acq",
        f"{sub_id}_{ses_id}_task-breathing*physio*.acq",

        "*task-breath*physio*.acq",
        "*task-breathing*physio*.acq",

        "*breath*physio*.acq",
        "*breathing*physio*.acq",
        "*deepbreath*physio*.acq",
        "*deepbreathing*physio*.acq",

        "*breath*.acq",
        "*breathing*.acq",
        "*deepbreath*.acq",
        "*deepbreathing*.acq",
    ]

    for pat in patterns:
        cands = sorted(ses_dir.glob(pat))
        if cands:
            return cands[0]

    raise FileNotFoundError(f"No breathing .acq found under: {ses_dir}")


# ----------------------------
# Channel selection
# ----------------------------
def pick_channel_by_index(channels, ch_num: int, one_based: bool):
    idx = ch_num - 1 if one_based else ch_num
    if idx < 0 or idx >= len(channels):
        raise RuntimeError(
            f"Channel out of range: requested {ch_num} ({'1-based' if one_based else '0-based'}), "
            f"but file has {len(channels)} channels."
        )
    return channels[idx], idx


# ----------------------------
# HR helpers
# ----------------------------
def moving_average_seconds(x, fs, win_sec=1.0, pad_mode="edge"):
    """
    Centered moving average with padding to avoid edge artifacts.
    """
    x = np.asarray(x, dtype=float)
    n = int(round(win_sec * fs))
    if n <= 1:
        return x

    idx = np.arange(len(x))
    good = np.isfinite(x)
    if not good.any():
        return x
    x_filled = x.copy()
    x_filled[~good] = np.interp(idx[~good], idx[good], x[good])

    kernel = np.ones(n, dtype=float) / n
    half = n // 2
    x_pad = np.pad(x_filled, (half, n - 1 - half), mode=pad_mode)
    y = np.convolve(x_pad, kernel, mode="valid")
    return y


def ecg_hr_quality_ok(hr_ts: np.ndarray, rpeaks: np.ndarray, fs: float) -> bool:
    """
    Simple sanity checks to avoid nonsense HR from very noisy ECG.
    """
    if rpeaks is None or len(rpeaks) < 50:
        return False

    hr = np.asarray(hr_ts, dtype=float)
    finite = np.isfinite(hr)
    if finite.mean() < 0.8:
        return False

    hr_med = np.nanmedian(hr)
    if not (35.0 <= hr_med <= 160.0):
        return False

    rr = np.diff(np.asarray(rpeaks, dtype=float)) / fs
    if rr.size < 10:
        return False
    rr_med = np.median(rr)
    if not (0.3 <= rr_med <= 2.0):  # 30–200 bpm
        return False

    out = np.mean((rr < 0.25) | (rr > 3.0))
    if out > 0.10:
        return False

    return True


def derive_hr_from_ppg(ppg: np.ndarray, fs: float):
    """
    Returns:
      hr_ts (len N), peaks (indices)
    """
    ppg = np.asarray(ppg, dtype=float)
    signals, info = nk.ppg_process(ppg, sampling_rate=fs)

    # Prefer sample-aligned HR if present
    hr_ts = None
    for key in ("PPG_Rate", "PPG_Rate_BPM", "PPG_Rate_Hz"):
        if key in signals.columns:
            hr_ts = np.asarray(signals[key], dtype=float)
            if "Hz" in key:
                hr_ts = hr_ts * 60.0
            break

    peaks = np.asarray(info.get("PPG_Peaks", []), dtype=int)

    # Fallback: build HR from peaks if needed
    if hr_ts is None or len(hr_ts) != len(ppg):
        N = len(ppg)
        hr_ts = np.full(N, np.nan, dtype=float)
        if peaks.size >= 3:
            tpk = peaks / fs
            ibi = np.diff(tpk)
            hr_inst = 60.0 / ibi
            idx = peaks[1:]
            hr_ts[idx] = hr_inst

            good = np.isfinite(hr_ts)
            if good.any():
                hr_ts[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), hr_ts[good])

    return hr_ts, peaks


# ----------------------------
# Plotting
# ----------------------------
def save_breathing_hr_window_plot(
    out_png: Path,
    t_win: np.ndarray,
    hr_win: np.ndarray,
    peak_idx: np.ndarray,
    trough_idx: np.ndarray,
    mean_max: float,
    mean_min: float,
    title: str = "Deep Breathing Heart Rate Responses",
):
    import os
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    ax.plot(t_win, hr_win, linewidth=1.2)

    # Mark peaks/troughs
    if peak_idx.size:
        ax.plot(t_win[peak_idx], hr_win[peak_idx], "r^", markersize=6, label="Max HR (top 5)")
    if trough_idx.size:
        ax.plot(t_win[trough_idx], hr_win[trough_idx], "gv", markersize=6, label="Min HR (bottom 5)")

    # Dashed horizontal lines
    if np.isfinite(mean_max):
        ax.axhline(mean_max, linestyle="--", linewidth=1.5)
        ax.text(t_win[0], mean_max, f"  Mean Max HR: {mean_max:.2f} bpm", va="bottom")
    if np.isfinite(mean_min):
        ax.axhline(mean_min, linestyle="--", linewidth=1.5)
        ax.text(t_win[0], mean_min, f"  Mean Min HR: {mean_min:.2f} bpm", va="bottom")

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Breathing task: HR max/min (top 5) in 8–9 min window (fallback to 3–4 min if too short) + plot + metrics"
    )

    ap.add_argument("--root", default="/export02/projects/LCS/01_physio", help="Root folder containing sub-*")
    ap.add_argument("--sub", required=True, help="Subject code like 2062")
    ap.add_argument("--ses", default="1", help="Session number like 1 or 2")

    ap.add_argument("--one_based", action="store_true", help="Interpret channel numbers as 1-based")
    ap.add_argument("--ecg_ch", type=int, default=4, help="ECG channel number (default 4)")
    ap.add_argument("--ppg_ch", type=int, default=5, help="PPG channel number (default 5)")
    ap.add_argument("--force_ppg", action="store_true", help="Always use PPG for HR (ignore ECG)")
    ap.add_argument("--fallback_ppg", action="store_true", help="If ECG HR looks bad, fall back to PPG")

    ap.add_argument("--win_start_min", type=float, default=8.0, help="Primary window start (minutes), default 8")
    ap.add_argument("--win_end_min", type=float, default=9.0, help="Primary window end (minutes), default 9")
    ap.add_argument("--top_k", type=int, default=5, help="Number of max/min points to use (default 5)")

    ap.add_argument(
        "--hr_smooth_sec",
        type=float,
        default=1.0,
        help="Smoothing window (sec) for HR used for peak/trough selection (default 1.0). Set 0 for none.",
    )
    ap.add_argument(
        "--min_peak_distance_sec",
        type=float,
        default=4.0,
        help="Min separation between peaks/troughs on HR curve (sec), default 4s",
    )

    ap.add_argument("--save", action="store_true", help="Save metrics (.mat) and plot (.png)")
    ap.add_argument("--out_root", default="derived", help="Root output folder under this project")

    args = ap.parse_args()

    root = Path(args.root)
    sub_id = f"sub-{args.sub}"
    ses_id = f"ses-{args.ses}"

    # Output folder
    out_root = Path(args.out_root)
    task_out = out_root / sub_id / ses_id / "breathing"
    task_out.mkdir(parents=True, exist_ok=True)

    # Locate file
    acq_path = build_breathing_acq_path(root, args.sub, args.ses)
    print(f"[INFO] Subject: {sub_id}  Session: {ses_id}")
    print(f"[INFO] Breathing ACQ: {acq_path}")

    # Load
    acq = bioread.read_file(str(acq_path))
    fs = float(acq.samples_per_second)
    print(f"[INFO] Sampling rate: {fs} Hz  |  Channels: {len(acq.channels)}")

    # ECG (for primary HR)
    ecg_ch, ecg_idx = pick_channel_by_index(acq.channels, args.ecg_ch, args.one_based)
    ecg = np.asarray(ecg_ch.data, dtype=float)
    print(f"[INFO] ECG channel index={ecg_idx} name={getattr(ecg_ch,'name','')}")

    N = len(ecg)
    t = np.arange(N) / fs
    duration_s = float(t[-1]) if len(t) else 0.0

    # --- Derive HR source
    source = "ECG"
    hr_ts = None
    peaks_for_debug = np.array([], dtype=int)
    ppg = None  # only set if used

    if not args.force_ppg:
        signals_ecg, info_ecg = nk.ecg_process(ecg, sampling_rate=fs)
        hr_ecg = np.asarray(signals_ecg["ECG_Rate"], dtype=float)
        rpeaks = np.asarray(info_ecg.get("ECG_R_Peaks", []), dtype=int)

        if args.fallback_ppg and (not ecg_hr_quality_ok(hr_ecg, rpeaks, fs)):
            print("[WARN] ECG HR quality looks bad -> falling back to PPG.")
            source = "PPG"
        else:
            hr_ts = hr_ecg
            peaks_for_debug = rpeaks
            source = "ECG"

    if args.force_ppg or source == "PPG":
        ppg_ch, ppg_idx = pick_channel_by_index(acq.channels, args.ppg_ch, args.one_based)
        ppg = np.asarray(ppg_ch.data, dtype=float)
        print(f"[INFO] Using PPG channel index={ppg_idx} name={getattr(ppg_ch,'name','')}")
        hr_ts, ppg_peaks = derive_hr_from_ppg(ppg, fs)
        peaks_for_debug = ppg_peaks
        source = "PPG"

    if hr_ts is None:
        raise RuntimeError("Failed to derive HR from both ECG and PPG.")

    print(f"[INFO] HR source: {source} | peaks_count={len(peaks_for_debug)}")

    # ----------------------------
    # Window selection rule:
    #   - primary: 8–9 min (480–540s)
    #   - if duration < 540s: use 3–4 min (180–240s)
    #   - safety: if even shorter than 240s: use last 60s
    # ----------------------------
    win_start = float(args.win_start_min) * 60.0
    win_end = float(args.win_end_min) * 60.0
    win_label = f"{args.win_start_min:.0f}–{args.win_end_min:.0f} min"

    if duration_s < 540.0:
        win_start, win_end = 180.0, 240.0
        win_label = "3–4 min (fallback)"
        print(f"[WARN] Breathing duration={duration_s:.1f}s < 540s. Using fallback window 180–240s (3–4 min).")

    if duration_s < win_end:
        # extreme fallback
        win_end = duration_s
        win_start = max(0.0, win_end - 60.0)
        win_label = "last 60 s (fallback)"
        print(f"[WARN] Breathing duration={duration_s:.1f}s < 240s. Using last 60s window {win_start:.1f}–{win_end:.1f}s.")

    mask = (t >= win_start) & (t < win_end)
    if not mask.any():
        raise RuntimeError(
            f"Selected window {win_start:.1f}-{win_end:.1f}s has no samples (duration={duration_s:.1f}s)."
        )

    t_win = t[mask]
    hr_win_raw = np.asarray(hr_ts[mask], dtype=float)

    # Smooth for peak selection (optional)
    if args.hr_smooth_sec and args.hr_smooth_sec > 0:
        hr_win = moving_average_seconds(hr_win_raw, fs, win_sec=args.hr_smooth_sec, pad_mode="edge")
    else:
        hr_win = hr_win_raw

    # Replace non-finite with interpolated values for peak detection stability
    idx = np.arange(len(hr_win))
    good = np.isfinite(hr_win)
    if good.any():
        hr_det = hr_win.copy()
        hr_det[~good] = np.interp(idx[~good], idx[good], hr_win[good])
    else:
        raise RuntimeError("HR window contains no finite values.")

    # --- Peak/trough detection on HR curve (within the window)
    dist = int(round(args.min_peak_distance_sec * fs))
    dist = max(1, dist)

    pk_idx, _ = find_peaks(hr_det, distance=dist)     # maxima
    tr_idx, _ = find_peaks(-hr_det, distance=dist)    # minima via inverted

    # pick top_k maxima/minima by value
    k = int(args.top_k)

    if pk_idx.size:
        pk_sorted = pk_idx[np.argsort(hr_det[pk_idx])[::-1]]  # descending
        pk_sel = np.sort(pk_sorted[:k])
    else:
        pk_sel = np.array([], dtype=int)

    if tr_idx.size:
        tr_sorted = tr_idx[np.argsort(hr_det[tr_idx])]        # ascending (lowest HR)
        tr_sel = np.sort(tr_sorted[:k])
    else:
        tr_sel = np.array([], dtype=int)

    mean_max = float(np.nanmean(hr_det[pk_sel])) if pk_sel.size else np.nan
    mean_min = float(np.nanmean(hr_det[tr_sel])) if tr_sel.size else np.nan

    diff = (mean_max - mean_min) if (np.isfinite(mean_max) and np.isfinite(mean_min)) else np.nan
    ratio = (mean_max / mean_min) if (np.isfinite(mean_max) and np.isfinite(mean_min) and mean_min > 0) else np.nan

    print(f"\n===== BREATHING RESULTS ({win_label}) =====")
    print(f"HR source: {source}")
    print(f"Window: {win_start:.1f}–{win_end:.1f} s  (duration={duration_s:.1f}s)")
    print(f"Found peaks: {pk_idx.size}  | selected top {min(k, pk_sel.size)}")
    print(f"Found troughs: {tr_idx.size} | selected bottom {min(k, tr_sel.size)}")
    print(f"Mean Max HR: {mean_max:.3f} bpm")
    print(f"Mean Min HR: {mean_min:.3f} bpm")
    print(f"Diff (MeanMax - MeanMin): {diff:.3f} bpm")
    print(f"Ratio (MeanMax / MeanMin): {ratio:.3f}")

    # Plot in that window
    plot_path = task_out / "deep_breathing_HR_plot.png"  # stable name for your report pipeline
    if args.save:
        save_breathing_hr_window_plot(
            out_png=plot_path,
            t_win=t_win,
            hr_win=hr_det,          # plot the same curve used for selection
            peak_idx=pk_sel,
            trough_idx=tr_sel,
            mean_max=mean_max,
            mean_min=mean_min,
            title=f"Deep Breathing Heart Rate Responses ({win_label})",
        )
        print(f"[OK] Saved breathing plot: {plot_path}")

    # Save metrics bundle
    if not args.save:
        print("\n[INFO] --save not set: not writing any .mat / .png outputs.")
        return

    from scipy.io import savemat

    metrics = {
        "hr_source": source,
        "duration_s": float(duration_s),
        "win_start_s": float(win_start),
        "win_end_s": float(win_end),
        "win_label": str(win_label),
        "top_k": int(k),

        "mean_max_hr_bpm": mean_max,
        "mean_min_hr_bpm": mean_min,
        "diff_bpm": diff,
        "ratio": ratio,

        # Indices within the *window* (0..len(window)-1)
        "peak_idx_win": pk_sel.astype(np.int32),
        "trough_idx_win": tr_sel.astype(np.int32),

        # Times (absolute seconds) of selected points
        "peak_times_s": t_win[pk_sel].astype(float) if pk_sel.size else np.array([], dtype=float),
        "trough_times_s": t_win[tr_sel].astype(float) if tr_sel.size else np.array([], dtype=float),

        "fs": float(fs),
        "sub_id": sub_id,
        "ses_id": ses_id,
        "acq_path": str(acq_path),
        "plot_path": str(plot_path),
        "ecg_channel_index": int(ecg_idx),
    }

    savemat(str(task_out / "breathing_metrics.mat"), metrics, do_compression=True)
    print(f"[OK] Saved breathing metrics bundle: {task_out / 'breathing_metrics.mat'}")


if __name__ == "__main__":
    main()
