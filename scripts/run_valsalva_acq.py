from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

import bioread
import neurokit2 as nk


# ----------------------------
# Path / file finding
# ----------------------------
def build_valsalva_acq_path(root: Path, sub_code: str, ses_num: str) -> Path:
    sub_id = f"sub-{sub_code}"
    ses_id = f"ses-{ses_num}"
    ses_dir = root / sub_id / ses_id

    if not ses_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {ses_dir}")

    # List all .acq files in the directory
    all_files = list(ses_dir.glob("*.acq"))

    # Filter files that contain 'valsalva' (case-insensitive)
    # and optionally 'physio'
    cands = [
        f for f in all_files 
        if "valsalva" in f.name.lower() and "physio" in f.name.lower()
    ]

    # If no "physio" matches, just look for valsalva
    if not cands:
        cands = [f for f in all_files if "valsalva" in f.name.lower()]

    if cands:
        # Sorting ensures we are deterministic (e.g., picking the first alphabetical match)
        return sorted(cands)[0]

    raise FileNotFoundError(f"No Valsalva .acq found under: {ses_dir}")

def ecg_hr_quality_ok(hr_ts: np.ndarray, rpeaks: np.ndarray, fs: float) -> bool:
    """Simple sanity checks so we don't use nonsense HR like 11 bpm."""
    if rpeaks is None or len(rpeaks) < 50:
        return False

    hr = np.asarray(hr_ts, dtype=float)
    finite = np.isfinite(hr)
    if finite.mean() < 0.8:
        return False

    # HR plausible range coverage
    hr_med = np.nanmedian(hr)
    if not (35.0 <= hr_med <= 160.0):
        return False

    # RR plausibility from rpeaks
    rr = np.diff(np.asarray(rpeaks, dtype=float)) / fs
    if rr.size < 10:
        return False
    rr_med = np.median(rr)
    if not (0.3 <= rr_med <= 2.0):  # 30–200 bpm
        return False

    # Too many impossible RR outliers
    out = np.mean((rr < 0.25) | (rr > 3.0))
    if out > 0.10:
        return False

    return True
def derive_hr_from_ppg(ppg: np.ndarray, fs: float):
    """
    Returns:
      hr_ts: sample-aligned HR (bpm), length N
      peaks: peak indices (PPG peaks)
      signals: neurokit signals (optional)
      info: neurokit info (optional)
    """
    ppg = np.asarray(ppg, dtype=float)

    # NeuroKit PPG pipeline
    signals, info = nk.ppg_process(ppg, sampling_rate=fs)

    # Prefer the sample-aligned HR if available
    hr_ts = None
    for key in ("PPG_Rate", "PPG_Rate_BPM", "PPG_Rate_Hz"):
        if key in signals.columns:
            hr_ts = np.asarray(signals[key], dtype=float)
            # if Hz, convert to bpm
            if "Hz" in key:
                hr_ts = hr_ts * 60.0
            break

    # Peaks
    peaks = np.asarray(info.get("PPG_Peaks", []), dtype=int)

    # If HR not present for some reason, build from peaks
    if hr_ts is None or len(hr_ts) != len(ppg):
        N = len(ppg)
        hr_ts = np.full(N, np.nan, dtype=float)
        if peaks.size >= 3:
            tpk = peaks / fs
            ibi = np.diff(tpk)              # seconds
            hr_inst = 60.0 / ibi            # bpm
            # timestamp at second peak of each interval
            idx = peaks[1:]
            hr_ts[idx] = hr_inst
            # interpolate to sample-aligned
            good = np.isfinite(hr_ts)
            if good.any():
                hr_ts[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), hr_ts[good])

    return hr_ts, peaks, signals, info

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
# Trigger extraction
# ----------------------------
def extract_valsalva_starts_from_trigger(trig: np.ndarray, fs: float) -> np.ndarray:
    """
    Supports:
      1) Discrete labels in trigger, e.g. 0,1,2,3 where 1/2/3 indicate repetition start.
         -> Returns first index of each nonzero label (first 3 labels).
      2) Pulse-like trigger:
         -> Threshold + rising edge detection + de-bounce -> first 3 edges.
    """
    x = np.asarray(trig, dtype=float)
    if x.size == 0:
        return np.array([], dtype=int)

    # Case 1: discrete labels
    rounded = np.round(x).astype(int)
    uniq = sorted([u for u in np.unique(rounded) if u != 0])
    # If it looks like a small set of integer labels, treat them as reps
    if 1 <= len(uniq) <= 10:
        starts = []
        for u in uniq:
            idx = np.where(rounded == u)[0]
            if idx.size:
                starts.append(int(idx[0]))
        return np.array(starts[:3], dtype=int)

    # Case 2: rising edges
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.array([], dtype=int)

    thr = 0.5 * (lo + hi)
    above = x > thr
    rising = np.where(np.diff(above.astype(int), prepend=0) == 1)[0]

    # de-bounce: enforce at least 5s separation
    min_sep = int(5.0 * fs)
    starts = []
    last = -10**12
    for r in rising:
        if r - last >= min_sep:
            starts.append(int(r))
            last = r

    return np.array(starts[:3], dtype=int)

def save_signal_peaks_debug_plot(out_png: Path, sig: np.ndarray, peaks: np.ndarray, fs: float, title: str):
    import os, matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = np.asarray(sig, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    peaks = peaks[(peaks >= 0) & (peaks < len(sig))]

    tt = np.arange(len(sig)) / fs
    fig = plt.figure(figsize=(14, 5))
    ax = plt.gca()
    ax.plot(tt, sig, linewidth=0.4)

    if peaks.size:
        ax.plot(tt[peaks], sig[peaks], "rx", markersize=3, label="Peaks")

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def autodetect_trigger_channel(acq, fs: float, trig_patterns: list[str]) -> np.ndarray | None:
    """
    Try:
      1) name-based match
      2) content-based: a channel that yields >=3 starts
    Returns trigger array or None.
    """
    pats = [p.lower() for p in trig_patterns]

    # name-based
    for ch in acq.channels:
        nm = (getattr(ch, "name", "") or "").lower()
        if any(p in nm for p in pats):
            return np.asarray(ch.data, dtype=float)

    # content-based
    for ch in acq.channels:
        cand = np.asarray(ch.data, dtype=float)
        starts = extract_valsalva_starts_from_trigger(cand, fs)
        if starts.size >= 3:
            return cand

    return None


# ----------------------------
# Smoothing (optional, small)
# ----------------------------
def moving_average_seconds(x, fs, win_sec=1.0, pad_mode="edge"):
    """
    Small moving average with padding to avoid edge artifacts.
    Default 1s smoothing (helps HR jitter).
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

def save_valsalva_debug_plot(
    out_png: Path,
    t: np.ndarray,
    hr_raw: np.ndarray,
    starts_s: np.ndarray,
    rep_max_t: np.ndarray,
    rep_max_v: np.ndarray,
    rep_min_t: np.ndarray,
    rep_min_v: np.ndarray,
):
    """
    Debug plot:
      - full HR (raw, no smoothing)
      - trigger starts as vertical lines
      - per rep max/min points in the 40s window after each start
    """
    import os
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 5))
    ax = plt.gca()

    ax.plot(t, hr_raw, linewidth=0.8, label="HR (raw)")

    # triggers
    for i, s in enumerate(starts_s):
        ax.axvline(float(s), linestyle="--", linewidth=1.5, label="Trigger" if i == 0 else None)

    # max/min markers
    # Max: red triangles, Min: blue inverted triangles
    good_max = np.isfinite(rep_max_t) & np.isfinite(rep_max_v)
    good_min = np.isfinite(rep_min_t) & np.isfinite(rep_min_v)

    if good_max.any():
        ax.plot(rep_max_t[good_max], rep_max_v[good_max], "r^", markersize=7, label="Max HR (0-40s)")
    if good_min.any():
        ax.plot(rep_min_t[good_min], rep_min_v[good_min], "bv", markersize=7, label="Min HR (0-40s)")

    ax.set_title("Valsalva Debug: Full HR (raw) + triggers + rep max/min (0–40s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HR (bpm)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# ----------------------------
# Figure
# ----------------------------
def save_valsalva_hr_figure(
    out_png: Path,
    t: np.ndarray,
    hr_ts: np.ndarray,
    start_t: float,
    hr_max_t: float,
    hr_min_t: float,
    hr_max: float,
    hr_min: float,
):
    import os
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # window: -10 to +40 seconds around start
    w = (t >= start_t - 10.0) & (t <= start_t + 70.0)
    tt = t[w] - start_t
    hh = hr_ts[w]

    fig = plt.figure(figsize=(14, 4))
    ax = plt.gca()

    ax.plot(tt, hh, color="k",linewidth=1.2)
    fig.tight_layout(pad=0.2)

    # Trigger
    ax.axvline(0.0, linestyle="--", linewidth=2)

    # Mark max/min with different colors
    ax.plot(hr_max_t - start_t, hr_max, "r^", markersize=9, label="Max HR")
    ax.plot(hr_min_t - start_t, hr_min, "bv", markersize=9, label="Min HR")
    ax.set_title("Valsalva Heart Rate (best repetition)", fontsize=16)
    ax.set_xlabel("Time relative to start (s)", fontsize=14)
    ax.set_ylabel("HR (bpm)", fontsize=14)

    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="best", fontsize=12, frameon=False)


    ax.grid(False, alpha=0.3)
    ax.legend(loc="best")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)

    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Valsalva: compute HR-only Valsalva ratio + figure")

    ap.add_argument("--root", default="/export02/projects/LCS/01_physio", help="Root folder containing sub-*")
    ap.add_argument("--sub", required=True, help="Subject code like 2062")
    ap.add_argument("--ses", default="1", help="Session number like 1 or 2")

    ap.add_argument("--one_based", action="store_true", help="Interpret channel numbers as 1-based")
    ap.add_argument("--ecg_ch", type=int, default=4, help="ECG channel number (default 4)")
    ap.add_argument("--trig_ch", type=int, default=None, help="Trigger channel number (optional; auto-detect if omitted)")
    ap.add_argument("--trig_patterns", nargs="+", default=["trigger", "trig", "marker", "event", "sync"],
                    help="Patterns to auto-detect trigger channel by name")

    ap.add_argument("--save", action="store_true", help="Save metrics (.mat) and best repetition figure (.png)")
    ap.add_argument("--out_root", default="derived", help="Root output folder under this project")

    ap.add_argument("--hr_smooth_sec", type=float, default=0.0,
                    help="Smoothing window (sec) for HR used in max/min (default 1.0). Set 0 for none.")
    ap.add_argument("--plot", action="store_true", help="Also try to show the figure (requires DISPLAY)")
    ap.add_argument("--debug_plot", action="store_true",
                help="Save debug plot: full HR (no smoothing) + triggers + max/min for each rep")

    ap.add_argument("--ecg_debug_plot", action="store_true",
                help="Save ECG debug plot: full ECG (downsampled) + detected R-peaks")
    ap.add_argument("--ecg_plot_ds", type=int, default=10,
                help="Downsample factor for ECG debug plot (default 10)")
    ap.add_argument("--ppg_ch", type=int, default=5, help="PPG channel number (default 5)")
    ap.add_argument("--force_ppg", action="store_true", help="Always use PPG for HR (ignore ECG)")
    ap.add_argument("--fallback_ppg", action="store_true", help="If ECG HR looks bad, fall back to PPG")

    args = ap.parse_args()

    root = Path(args.root)
    sub_id = f"sub-{args.sub}"
    ses_id = f"ses-{args.ses}"
    out_root = Path(args.out_root)
    task_out = out_root / sub_id / ses_id / "valsalva"
    task_out.mkdir(parents=True, exist_ok=True)

    # Locate file
    acq_path = build_valsalva_acq_path(root, args.sub, args.ses)
    print(f"[INFO] Subject: {sub_id}  Session: {ses_id}")
    print(f"[INFO] Valsalva ACQ: {acq_path}")

    # Load
    acq = bioread.read_file(str(acq_path))
    fs = float(acq.samples_per_second)
    print(f"[INFO] Sampling rate: {fs} Hz  |  Channels: {len(acq.channels)}")

    # ECG channel (channel 4 by your rule)
    ecg_ch, ecg_idx = pick_channel_by_index(acq.channels, args.ecg_ch, args.one_based)
    ecg = np.asarray(ecg_ch.data, dtype=float)
    print(f"[INFO] ECG channel index={ecg_idx} name={getattr(ecg_ch,'name','')}")

    # Trigger channel
    if args.trig_ch is not None:
        trig_ch, trig_idx = pick_channel_by_index(acq.channels, args.trig_ch, args.one_based)
        trig = np.asarray(trig_ch.data, dtype=float)
        print(f"[INFO] Trigger channel index={trig_idx} name={getattr(trig_ch,'name','')}")
    else:
        trig = autodetect_trigger_channel(acq, fs, args.trig_patterns)
        if trig is None:
            raise RuntimeError("Could not auto-detect trigger channel. Provide --trig_ch <N>.")
        print("[INFO] Trigger channel auto-detected.")

    # Extract repetition starts (3 expected)
    starts = extract_valsalva_starts_from_trigger(trig, fs)
    # Sort + drop starts that occur too early (e.g., < 10s)
    starts = np.sort(starts)
    starts = starts[starts >= int(10.0 * fs)]
    starts = starts[:3]

    if starts.size == 0:
        raise RuntimeError("Trigger found but no start events extracted.")
    if starts.size < 3:
        print(f"[WARN] Only found {starts.size} start(s). Proceeding with what we have.")
    else:
        print(f"[INFO] Found {starts.size} starts (expect 3).")

    starts_s = starts / fs
    print("[INFO] Starts (s):", np.array2string(starts_s, precision=3))

    t = np.arange(len(ecg)) / fs  # use signal length as reference

    use_ppg = bool(args.force_ppg)
    source = "ECG"

    # --- Try ECG first (unless forced PPG)
    if not use_ppg:
        signals_ecg, info_ecg = nk.ecg_process(ecg, sampling_rate=fs)
        hr_ecg = np.asarray(signals_ecg["ECG_Rate"], dtype=float)
        rpeaks = np.asarray(info_ecg.get("ECG_R_Peaks", []), dtype=int)

        if args.fallback_ppg and (not ecg_hr_quality_ok(hr_ecg, rpeaks, fs)):
            print("[WARN] ECG HR quality looks bad -> falling back to PPG.")
            use_ppg = True
        else:
            hr_ts = hr_ecg
            peaks_for_debug = rpeaks
            source = "ECG"

    # --- PPG fallback / forced
    if use_ppg:
        ppg_ch, ppg_idx = pick_channel_by_index(acq.channels, args.ppg_ch, args.one_based)
        ppg = np.asarray(ppg_ch.data, dtype=float)
        print(f"[INFO] Using PPG channel index={ppg_idx} name={getattr(ppg_ch,'name','')}")
        hr_ts, ppg_peaks, signals_ppg, info_ppg = derive_hr_from_ppg(ppg, fs)
        peaks_for_debug = ppg_peaks
        source = "PPG"

    if source == "ECG":
        print(f"[INFO] n_rpeaks={len(peaks_for_debug)}")
    else:
        print(f"[INFO] n_ppg_peaks={len(peaks_for_debug)}")

    # Peaks debug plot (works for ECG or PPG)
    if args.ecg_debug_plot:
        if source == "ECG":
            figp = task_out / "valsalva_ecg_rpeaks_full.png"
            save_signal_peaks_debug_plot(
                out_png=figp,
                sig=ecg,
                peaks=peaks_for_debug,
                fs=fs,
                title="ECG Debug: Full ECG with detected R-peaks",
            )
        else:
            figp = task_out / "valsalva_ppg_peaks_full.png"
            save_signal_peaks_debug_plot(
                out_png=figp,
                sig=ppg,
                peaks=peaks_for_debug,
                fs=fs,
                title="PPG Debug: Full PPG with detected peaks",
            )
        print(f"[OK] Saved peaks debug plot: {figp}")


    rep_max_t = np.full(len(starts), np.nan, dtype=float)
    rep_min_t = np.full(len(starts), np.nan, dtype=float)
    rep_max_v = np.full(len(starts), np.nan, dtype=float)
    rep_min_v = np.full(len(starts), np.nan, dtype=float)

    for i, s_idx in enumerate(starts):
        s_t = s_idx / fs
        w = (t >= s_t) & (t < s_t + 70.0)
        tw = t[w]
        hw = hr_ts[w]   # RAW HR (no smoothing)

        if hw.size == 0 or not np.isfinite(hw).any():
            continue

        i_max = int(np.nanargmax(hw))
        i_min = int(np.nanargmin(hw))

        rep_max_t[i] = float(tw[i_max])
        rep_min_t[i] = float(tw[i_min])
        rep_max_v[i] = float(hw[i_max])
        rep_min_v[i] = float(hw[i_min])

    # Optional smoothing to stabilize max/min
    if args.hr_smooth_sec and args.hr_smooth_sec > 0:
        hr_use = moving_average_seconds(hr_ts, fs, win_sec=args.hr_smooth_sec, pad_mode="edge")
    else:
        hr_use = hr_ts

    # Compute ratios for each repetition
    ratios = []
    rep_hr_max = []
    rep_hr_min = []

    for i, s_idx in enumerate(starts):
        s_t = s_idx / fs
        w = (t >= s_t) & (t < s_t + 70.0)  # 70s post-start
        hr_w = hr_use[w]

        if hr_w.size == 0 or not np.isfinite(hr_w).any():
            ratios.append(np.nan)
            rep_hr_max.append(np.nan)
            rep_hr_min.append(np.nan)
            continue

        hr_max = float(np.nanmax(hr_w))
        hr_min = float(np.nanmin(hr_w))
        ratio = (hr_max / hr_min) if (np.isfinite(hr_max) and np.isfinite(hr_min) and hr_min > 0) else np.nan

        ratios.append(ratio)
        rep_hr_max.append(hr_max)
        rep_hr_min.append(hr_min)

    ratios = np.array(ratios, dtype=float)
    rep_hr_max = np.array(rep_hr_max, dtype=float)
    rep_hr_min = np.array(rep_hr_min, dtype=float)

    if np.isfinite(ratios).any():
        best_rep = int(np.nanargmax(ratios))
        valsalva_ratio = float(np.nanmax(ratios))
    else:
        best_rep = -1
        valsalva_ratio = np.nan

    print("\n===== VALSALVA RESULTS =====")
    for i in range(len(ratios)):
        print(f"Rep {i+1}: HRmax={rep_hr_max[i]:.2f} HRmin={rep_hr_min[i]:.2f} ratio={ratios[i]:.3f}")
    print(f"[RESULT] Valsalva ratio (max over reps) = {valsalva_ratio:.3f}  | best rep = {best_rep+1 if best_rep>=0 else 'NA'}")


    fig_path = task_out / "valsalva_best_rep_hr.png"
    mat_path = task_out / "valsalva_metrics.mat"
    debug_fig_path = task_out / "valsalva_debug_full_hr.png"

    if args.debug_plot:
        save_valsalva_debug_plot(
            out_png=debug_fig_path,
            t=t,
            hr_raw=hr_ts,          # raw HR, no smoothing
            starts_s=starts_s,
            rep_max_t=rep_max_t,
            rep_max_v=rep_max_v,
            rep_min_t=rep_min_t,
            rep_min_v=rep_min_v,
        )
        print(f"[OK] Saved debug plot: {debug_fig_path}")

    # Create the best-repetition figure (always useful; save only if --save)
    hr_max_t = np.nan
    hr_min_t = np.nan
    hr_max = np.nan
    hr_min = np.nan
    best_start_t = np.nan

    if best_rep >= 0:
        best_start_t = starts[best_rep] / fs
        w70 = (t >= best_start_t) & (t < best_start_t + 70.0)
        tw = t[w70]
        hw = hr_use[w70]

        # indices of max/min within post-start 70s
        i_max = int(np.nanargmax(hw))
        i_min = int(np.nanargmin(hw))

        hr_max_t = float(tw[i_max])
        hr_min_t = float(tw[i_min])
        hr_max = float(hw[i_max])
        hr_min = float(hw[i_min])

        # Save figure if requested
        if args.save:
            save_valsalva_hr_figure(fig_path, t, hr_use, best_start_t, hr_max_t, hr_min_t, hr_max, hr_min)
            print(f"[OK] Saved Valsalva HR figure: {fig_path}")

            # If user also wants to view interactively
            if args.plot:
                import os
                if os.environ.get("DISPLAY"):
                    import matplotlib.pyplot as plt
                    import matplotlib.image as mpimg
                    img = mpimg.imread(fig_path)
                    plt.figure(figsize=(10, 4))
                    plt.imshow(img)
                    plt.axis("off")
                    plt.show()
                else:
                    print("[INFO] --plot requested but no DISPLAY available.")
    else:
        print("[WARN] No valid repetition ratio found; skipping figure generation.")

    # Save metrics bundle
    if not args.save:
        print("\n[INFO] --save not set: not writing any .mat / .png outputs.")
        return

    from scipy.io import savemat
    metrics = {
        "valsalva_ratio": float(valsalva_ratio),
        "ratios": ratios,
        "best_rep": int(best_rep + 1) if best_rep >= 0 else -1,
        "starts_s": starts_s,
        "rep_hr_max": rep_hr_max,
        "rep_hr_min": rep_hr_min,
        "best_start_s": float(best_start_t) if np.isfinite(best_start_t) else np.nan,
        "best_hr_max": float(hr_max) if np.isfinite(hr_max) else np.nan,
        "best_hr_min": float(hr_min) if np.isfinite(hr_min) else np.nan,
        "best_hr_max_t_s": float(hr_max_t) if np.isfinite(hr_max_t) else np.nan,
        "best_hr_min_t_s": float(hr_min_t) if np.isfinite(hr_min_t) else np.nan,
        "fs": float(fs),
        "sub_id": sub_id,
        "ses_id": ses_id,
        "acq_path": str(acq_path),
        "figure_path": str(fig_path) if best_rep >= 0 else "",
        "ecg_channel_index": int(ecg_idx),
    }

    savemat(str(mat_path), metrics, do_compression=True)
    print(f"[OK] Saved Valsalva metrics bundle: {mat_path}")


if __name__ == "__main__":
    main()
