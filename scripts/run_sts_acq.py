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
from src.physio_qc.metrics.doppler import process_doppler


DOPPLER_PARAMS = {
    "filter_method": "sg_wavelet", "filter_order": 3, "cutoff_freq": 25,
    "lowcut": 0.5, "highcut": 15.0, "apply_lowcut": True, "apply_highcut": True,
    "sg_win": 0.1, "wavelet": "db6", "level": 10, "alpha": 4.0,
    "drop_levels": 1, "trend_win": 2.0,
}


def moving_average_seconds(x, fs, win_sec=5.0, pad_mode="edge"):
    """
    Centered moving average with padding to avoid start/end artifacts.
    pad_mode: "edge" (repeat endpoints) or "reflect" (mirror)
    """
    x = np.asarray(x, dtype=float)
    n = int(round(win_sec * fs))
    if n <= 1:
        return x

    # Fill NaNs by linear interpolation
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if not good.any():
        return x
    x_filled = x.copy()
    x_filled[~good] = np.interp(idx[~good], idx[good], x[good])

    kernel = np.ones(n, dtype=float) / n

    half = n // 2
    # pad left/right so "valid" output matches original length
    x_pad = np.pad(x_filled, (half, n - 1 - half), mode=pad_mode)
    y = np.convolve(x_pad, kernel, mode="valid")  # length == len(x)

    return y


def save_sts_summary_figure(
    out_png: Path,
    t_hr: np.ndarray, hr_ts: np.ndarray,
    t_bp: np.ndarray, map_ts: np.ndarray,
    stand_onset: float,
    sup_start: float, sup_end: float,
    plt_start: float, plt_end: float,
    mean_hr_sup: float, mean_hr_plt: float, dhr: float,
    mean_map_sup: float, mean_map_plt: float, dmap: float, fs=250.0,
    t_doppler: np.ndarray | None = None,
    doppler_mean_ts: np.ndarray | None = None,
    show_bp: bool = True,
    ):
    import os
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_doppler = (
        t_doppler is not None and doppler_mean_ts is not None
        and len(t_doppler) and np.isfinite(np.asarray(doppler_mean_ts, dtype=float)).any()
    )
    n_panels = 1 + int(show_bp) + int(has_doppler)
    figure_height = {1: 4.5, 2: 6.2, 3: 8.0}[n_panels]
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, figure_height), sharex=True)
    axes = np.atleast_1d(axes)

    hr_smooth = moving_average_seconds(hr_ts, fs, win_sec=2.0)
    axes[0].plot(t_hr, hr_smooth, color="#C2413B", linewidth=1.35)
    axes[0].set_title("Heart Rate", loc="left")
    axes[0].set_ylabel("Heart rate (bpm)")

    panel_index = 1
    if show_bp:
        map_smooth = moving_average_seconds(map_ts, fs, win_sec=2.0)
        axes[panel_index].plot(t_bp, map_smooth, color="#1D4ED8", linewidth=1.35)
        axes[panel_index].set_title("Mean Arterial Pressure", loc="left")
        axes[panel_index].set_ylabel("MAP (mmHg)")
        panel_index += 1

    if has_doppler:
        doppler_values = np.asarray(doppler_mean_ts, dtype=float)
        doppler_time = np.asarray(t_doppler, dtype=float)
        if len(doppler_time) > 1:
            doppler_fs = 1.0 / np.nanmedian(np.diff(doppler_time))
            doppler_values = moving_average_seconds(doppler_values, doppler_fs, win_sec=2.0)
        axes[panel_index].plot(doppler_time, doppler_values, color="#047857", linewidth=1.35)
        axes[panel_index].set_title("Mean Doppler Velocity", loc="left")
        axes[panel_index].set_ylabel("Velocity (cm/s)")

    def interval_bracket(axis, start: float, end: float, label: str, color: str) -> None:
        y = 1.035
        transform = axis.get_xaxis_transform()
        axis.plot([start, end], [y, y], transform=transform, color=color, linewidth=2.0, clip_on=False)
        axis.plot([start, start], [y - 0.025, y + 0.025], transform=transform,
                  color=color, linewidth=2.0, clip_on=False)
        axis.plot([end, end], [y - 0.025, y + 0.025], transform=transform,
                  color=color, linewidth=2.0, clip_on=False)
        axis.text((start + end) / 2.0, y + 0.015, label, transform=transform,
                  color=color, fontsize=8, fontweight="bold", ha="center", va="bottom", clip_on=False)

    # The panels share time, so one set of brackets identifies the analysis
    # windows without drawing reference lines over each physiological trace.
    interval_bracket(axes[0], sup_start, sup_end, "Baseline window", "#2563EB")
    interval_bracket(axes[0], plt_start, plt_end, "Plateau window", "#047857")

    for index, axis in enumerate(axes):
        axis.axvline(stand_onset, color="#B45309", linestyle=":", linewidth=1.5)
        axis.grid(axis="y", alpha=0.18)
        if index == 0:
            axis.text(
                stand_onset, 1.025, "Stand onset", transform=axis.get_xaxis_transform(),
                color="#92400E", fontsize=9, fontweight="bold", ha="center", va="bottom",
                clip_on=False,
            )

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Supine-to-Stand Cardiovascular Response", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=1.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

def build_sts_acq_path(root: Path, sub_code: str, ses_num: str) -> Path:
    sub_id = f"sub-{sub_code}"
    ses_id = f"ses-{ses_num}"
    ses_dir = root / sub_id / ses_id

    # common expected filename
    expected = ses_dir / f"{sub_id}_{ses_id}_task-sts_physio.acq"
    if expected.exists():
        return expected

    # fallback patterns
    cands = sorted(ses_dir.glob("*task-sts*physio*.acq"))
    if cands:
        return cands[0]

    cands = sorted(ses_dir.glob("*STS*physio*.acq"))
    if cands:
        return cands[0]

    cands = sorted(ses_dir.glob("*sts*.acq"))
    if cands:
        return cands[0]

    raise FileNotFoundError(f"No STS .acq found under: {ses_dir}")



def pick_channel_by_index(channels, ch_num: int, one_based: bool):
    idx = ch_num - 1 if one_based else ch_num
    if idx < 0 or idx >= len(channels):
        raise RuntimeError(
            f"Channel out of range: requested {ch_num} ({'1-based' if one_based else '0-based'}), "
            f"but file has {len(channels)} channels."
        )
    return channels[idx], idx


def detect_doppler_channel(channels):
    """Find the Doppler waveform using the standard LCS A6 naming convention."""
    patterns = ("doppler", "a6", "a 6", "a5", "a 5")
    for index, channel in enumerate(channels):
        name = str(getattr(channel, "name", "") or "").lower()
        if any(pattern in name for pattern in patterns):
            return channel, index
    return None, None


def mean_beat_quality_in_window(trough_indices, beat_scores, sampling_rate, start_s, end_s):
    """Return a duration-weighted mean quality for beats overlapping a time window."""
    troughs = np.asarray(trough_indices, dtype=float).ravel()
    scores = np.asarray(beat_scores, dtype=float).ravel()
    n_beats = min(len(scores), max(len(troughs) - 1, 0))
    if n_beats <= 0 or sampling_rate <= 0 or end_s <= start_s:
        return np.nan

    beat_starts = troughs[:n_beats] / sampling_rate
    beat_ends = troughs[1:n_beats + 1] / sampling_rate
    scores = scores[:n_beats]
    overlap = np.maximum(0.0, np.minimum(beat_ends, end_s) - np.maximum(beat_starts, start_s))
    valid = (overlap > 0) & np.isfinite(scores)
    if not np.any(valid):
        return np.nan
    return float(np.sum(scores[valid] * overlap[valid]) / np.sum(overlap[valid]))


def condition_median_bp_panel_allowed(
    time_s, map_values, supine_start_s, supine_end_s,
    standing_start_s, standing_end_s, lower=40.0, upper=200.0,
):
    """Check whether supine and standing-window median MAP values are plausible."""
    time_s = np.asarray(time_s, dtype=float).ravel()
    map_values = np.asarray(map_values, dtype=float).ravel()
    n_values = min(len(time_s), len(map_values))
    if n_values == 0:
        return False, np.nan, np.nan

    time_s = time_s[:n_values]
    map_values = map_values[:n_values]
    supine = map_values[(time_s >= supine_start_s) & (time_s < supine_end_s)]
    standing = map_values[(time_s >= standing_start_s) & (time_s < standing_end_s)]
    supine = supine[np.isfinite(supine)]
    standing = standing[np.isfinite(standing)]
    if supine.size == 0 or standing.size == 0:
        return False, np.nan, np.nan

    supine_median = float(np.median(supine))
    standing_median = float(np.median(standing))
    allowed = (
        lower <= supine_median <= upper
        and lower <= standing_median <= upper
    )
    return bool(allowed), supine_median, standing_median


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
    ap.add_argument("--doppler_ch", type=int, default=None, help="Optional Doppler fallback channel number")
    ap.add_argument("--one_based", action="store_true", help="Interpret channel numbers as 1-based")
    ap.add_argument("--ecg_method", default="neurokit", help="NeuroKit ECG processing method")

    ap.add_argument("--bp_prominence", type=float, default=20.0)
    ap.add_argument("--cal_thr", type=float, default=0.1)
    ap.add_argument("--cal_min_dur", type=float, default=0.65)

    ap.add_argument("--no_save", action="store_true", help="Do not save any .mat files")
    ap.add_argument("--out_root", default="derived", help="Root folder to save outputs under this project")
    ap.add_argument("--save", action="store_true", help="Save metrics to a single .mat file")

    ap.add_argument("--plot", action="store_true", help="Show QC plots (falls back to saving png if no DISPLAY)")
    ap.add_argument("--plot_seconds", type=float, default=60.0, help="Seconds to plot for QC")
    ap.add_argument("--plot_outdir", default="outputs", help="Where to save plot if no DISPLAY")
    # Around line 170
    ap.add_argument("--height", type=float, default=0.0, help="Subject height in cm for hydrostatic correction")
    ap.add_argument("--stand_onset_s", type=float, default=300.0, help="Standing onset in seconds (default 300)")
    ap.add_argument(
        "--doppler_quality_threshold", type=float, default=0.8,
        help="Minimum mean beat quality required in both supine and standing periods (default 0.8)",
    )
    args = ap.parse_args()

    root = Path(args.root)
    sub_id = f"sub-{args.sub}"
    ses_id = f"ses-{args.ses}"
    out_root = Path(args.out_root)
    task_out = out_root / sub_id / ses_id / "sts"
    task_out.mkdir(parents=True, exist_ok=True)
    out_mat = task_out / "sts_metrics.mat"


    acq_path = build_sts_acq_path(root, args.sub, args.ses)

    print(f"[INFO] Subject: {sub_id}  Session: {ses_id}")
    print(f"[INFO] Rest ACQ: {acq_path}")

    d = bioread.read_file(str(acq_path))
    fs = float(d.samples_per_second)


    print(f"[INFO] Sampling rate: {fs} Hz  |  Channels: {len(d.channels)}")

    ecg_ch, ecg_idx = pick_channel_by_index(d.channels, args.ecg_ch, args.one_based)
    bp_ch, bp_idx = pick_channel_by_index(d.channels, args.bp_ch, args.one_based)
    doppler_ch, doppler_idx = detect_doppler_channel(d.channels)
    if doppler_ch is None and args.doppler_ch is not None:
        doppler_ch, doppler_idx = pick_channel_by_index(d.channels, args.doppler_ch, args.one_based)
    print(f"[INFO] ECG channel index={ecg_idx} name={getattr(ecg_ch,'name','')}")
    print(f"[INFO]  BP channel index={bp_idx} name={getattr(bp_ch,'name','')}")
    if doppler_ch is not None:
        print(f"[INFO] Doppler channel index={doppler_idx} name={getattr(doppler_ch,'name','')}")
    else:
        print("[INFO] Doppler channel not found; STS figure will use HR and MAP panels only.")

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
        # STS windows (seconds)
    stand_onset = float(args.stand_onset_s)
    supine_start, supine_end = 60.0, 240.0      # min 1-4
    plateau_start, plateau_end = 600.0, 780.0   # min 6-13

    mask_supine = (t >= supine_start) & (t < supine_end)
    mask_plateau = (t >= plateau_start) & (t < plateau_end)
    derived = compute_bp_derived_from_peaks(bp_filt, t, peaks, troughs)

# 1. Get the raw derived signals
    map_interp = derived["MBP"]
    systolic_interp = derived["SBP"]
    diastolic_interp = derived["DBP"]

    # 2. APPLY OFFSET ONCE (The "Source of Truth")
    # Apply to the time-series itself for t > 5 mins (300s)
    if args.height > 0:
        bp_offset = 0.4*0.77* args.height
        plot_mask_standing = (t >= stand_onset)
        
        map_interp[plot_mask_standing] -= bp_offset
        systolic_interp[plot_mask_standing] -= bp_offset
        diastolic_interp[plot_mask_standing] -= bp_offset
        print(f"[INFO] Applied height correction: -{bp_offset:.2f} mmHg (t >= {stand_onset:g}s)")

    bp_plot_allowed, supine_map_median, standing_map_median = condition_median_bp_panel_allowed(
        t, map_interp,
        supine_start, supine_end,
        plateau_start, plateau_end,
    )
    bp_plot_status = "available; supine and standing median MAP within 40-200 mmHg"
    if not bp_plot_allowed:
        bp_plot_status = "supine or standing median MAP outside 40-200 mmHg; panel omitted"
        print(
            "[WARN] STS BP panel omitted: "
            f"supine median MAP={supine_map_median:.2f} mmHg, "
            f"standing median MAP={standing_map_median:.2f} mmHg"
        )

    # 3. CALCULATE MEANS (Pulling from the already adjusted map_interp)
    def mean_in_window(x, mask):
        x = np.asarray(x)
        xw = x[mask]
        if xw.size == 0:
            return np.nan
        return float(np.nanmean(xw))

    # Supine BP (remains original because t < 300s)
    mean_MAP_sup = mean_in_window(map_interp, mask_supine)
    mean_sys_sup = mean_in_window(systolic_interp, mask_supine)
    mean_dia_sup = mean_in_window(diastolic_interp, mask_supine)
    mean_pp_sup  = (mean_sys_sup - mean_dia_sup)

    # Plateau BP (automatically includes correction because map_interp was shifted)
    mean_MAP_plt = mean_in_window(map_interp, mask_plateau)
    mean_sys_plt = mean_in_window(systolic_interp, mask_plateau)
    mean_dia_plt = mean_in_window(diastolic_interp, mask_plateau)
    mean_pp_plt  = (mean_sys_plt - mean_dia_plt)

    # 4. DELTAS (Correctly calculated once)
    dMAP = mean_MAP_plt - mean_MAP_sup
    dSYS = mean_sys_plt - mean_sys_sup
    dDIA = mean_dia_plt - mean_dia_sup
    dPP  = mean_pp_plt  - mean_pp_sup

    # ---- Doppler mean velocity (optional)
    doppler_result = None
    doppler_time = np.array([], dtype=float)
    doppler_mean_velocity = np.array([], dtype=float)
    mean_doppler_sup = mean_doppler_plt = d_doppler = np.nan
    doppler_quality_supine = doppler_quality_standing = np.nan
    doppler_plot_allowed = False
    doppler_status = "channel not found"
    if doppler_ch is not None:
        try:
            doppler_result = process_doppler(np.asarray(doppler_ch.data, dtype=float), fs, DOPPLER_PARAMS)
            if not doppler_result:
                raise ValueError("insufficient Doppler peaks/troughs")
            doppler_time = np.asarray(doppler_result["time_4hz"], dtype=float)
            doppler_mean_velocity = np.asarray(doppler_result["map_4hz"], dtype=float)
            doppler_sup_mask = (doppler_time >= supine_start) & (doppler_time < supine_end)
            doppler_plt_mask = (doppler_time >= plateau_start) & (doppler_time < plateau_end)
            mean_doppler_sup = mean_in_window(doppler_mean_velocity, doppler_sup_mask)
            mean_doppler_plt = mean_in_window(doppler_mean_velocity, doppler_plt_mask)
            d_doppler = mean_doppler_plt - mean_doppler_sup
            recording_end = len(doppler_result["filtered"]) / fs
            doppler_quality_supine = mean_beat_quality_in_window(
                doppler_result["current_troughs"], doppler_result.get("beat_scores", []),
                fs, 0.0, min(stand_onset, recording_end),
            )
            doppler_quality_standing = mean_beat_quality_in_window(
                doppler_result["current_troughs"], doppler_result.get("beat_scores", []),
                fs, stand_onset, recording_end,
            )
            doppler_plot_allowed = bool(
                np.isfinite(doppler_quality_supine)
                and np.isfinite(doppler_quality_standing)
                and doppler_quality_supine >= args.doppler_quality_threshold
                and doppler_quality_standing >= args.doppler_quality_threshold
            )
            if doppler_plot_allowed:
                doppler_status = "available; quality threshold passed"
            else:
                doppler_status = "quality threshold failed; panel omitted"
                print(
                    "[WARN] STS Doppler panel omitted: "
                    f"supine quality={doppler_quality_supine:.3f}, "
                    f"standing quality={doppler_quality_standing:.3f}, "
                    f"required>={args.doppler_quality_threshold:.3f}"
                )
        except Exception as exc:
            print(f"[WARN] STS Doppler processing unavailable: {exc}")
            doppler_result = None
            doppler_time = np.array([], dtype=float)
            doppler_mean_velocity = np.array([], dtype=float)
            doppler_status = str(exc)


    # ---- ECG/HRV
    # Keep peak detection method fixed (neurokit); only vary cleaning method.
    ecg_clean = nk.ecg_clean(ecg, sampling_rate=fs, method=args.ecg_method)
    _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs, method="neurokit")
    rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)

    if len(rpeaks) >= 2:
        hr_ts = nk.signal_rate(rpeaks, sampling_rate=fs, desired_length=len(ecg))
    else:
        hr_ts = np.full(len(ecg), np.nan)
    t_hr = np.arange(len(hr_ts)) / fs

    
    def rr_in_window(rpeaks, fs, start_s, end_s):
        """
        Return RR intervals (seconds) whose timestamps fall within [start_s, end_s).
        We timestamp each RR at the time of the *second* R-peak (common convention).
        """
        rpeaks = np.asarray(rpeaks, dtype=float)
        if rpeaks.size < 3:
            return np.array([], dtype=float)

        t_r = rpeaks / fs              # R-peak times (s)
        rr_s = np.diff(t_r)            # RR intervals (s)
        rr_t = t_r[1:]                 # timestamp for each RR (s)

        m = (rr_t >= start_s) & (rr_t < end_s)
        return rr_s[m]


    def rmssd_ms(rr_s):
        rr_s = np.asarray(rr_s, dtype=float)
        if rr_s.size < 3:
            return np.nan
        diff = np.diff(rr_s)
        return float(np.sqrt(np.mean(diff * diff)) * 1000.0)



    rr_sup = rr_in_window(rpeaks, fs, supine_start, supine_end)
    rr_plt = rr_in_window(rpeaks, fs, plateau_start, plateau_end)

    mean_RR_sup = float(np.nanmean(rr_sup)) if rr_sup.size else np.nan
    mean_HR_sup = float(60.0/mean_RR_sup) if np.isfinite(mean_RR_sup) and mean_RR_sup > 0 else np.nan
    RMSSD_sup   = rmssd_ms(rr_sup)

    mean_RR_plt = float(np.nanmean(rr_plt)) if rr_plt.size else np.nan
    mean_HR_plt = float(60.0/mean_RR_plt) if np.isfinite(mean_RR_plt) and mean_RR_plt > 0 else np.nan
    RMSSD_plt   = rmssd_ms(rr_plt)

    dHR    = mean_HR_plt - mean_HR_sup
    dRMSSD = RMSSD_plt   - RMSSD_sup
    # Save STS summary figure (for compile_whole insertion)
    fig_path = task_out / "STS_HR_MAP.png"
    save_sts_summary_figure(
        out_png=fig_path,
        t_hr=t_hr, hr_ts=hr_ts,
        t_bp=t, map_ts=map_interp,
        stand_onset=stand_onset,
        sup_start=supine_start, sup_end=supine_end,
        plt_start=plateau_start, plt_end=plateau_end,
        mean_hr_sup=mean_HR_sup, mean_hr_plt=mean_HR_plt, dhr=dHR,
        mean_map_sup=mean_MAP_sup, mean_map_plt=mean_MAP_plt, dmap=dMAP,
        t_doppler=doppler_time if doppler_plot_allowed else None,
        doppler_mean_ts=doppler_mean_velocity if doppler_plot_allowed else None,
        show_bp=bp_plot_allowed,
    )
    print(f"[OK] Saved STS summary figure: {fig_path}")


    # ---- Print derived indices

    print("\n===== STS DERIVED INDICES =====")
    print(f"Supine (1-4min):  MAP={mean_MAP_sup:.2f} SYS={mean_sys_sup:.2f} DIA={mean_dia_sup:.2f} PP={mean_pp_sup:.2f} | HR={mean_HR_sup:.2f} RMSSD={RMSSD_sup:.2f}ms")
    print(f"Plateau (6-13min): MAP={mean_MAP_plt:.2f} SYS={mean_sys_plt:.2f} DIA={mean_dia_plt:.2f} PP={mean_pp_plt:.2f} | HR={mean_HR_plt:.2f} RMSSD={RMSSD_plt:.2f}ms")
    print(f"Delta (plt-sup):   dMAP={dMAP:.2f} dSYS={dSYS:.2f} dDIA={dDIA:.2f} dPP={dPP:.2f} | dHR={dHR:.2f} dRMSSD={dRMSSD:.2f}ms")
    if doppler_result:
        print(f"Doppler mean velocity: baseline={mean_doppler_sup:.2f} plateau={mean_doppler_plt:.2f} "
              f"delta={d_doppler:.2f} cm/s")
        print(
            f"Doppler quality: supine={doppler_quality_supine:.3f} "
            f"standing={doppler_quality_standing:.3f} "
            f"threshold={args.doppler_quality_threshold:.3f} "
            f"panel={'included' if doppler_plot_allowed else 'omitted'}"
        )


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

    if args.no_save:
        print("\n[INFO] --no_save set: not writing any .mat files.")
        return

    # Saving intentionally omitted for now.
    if args.save:
        from scipy.io import savemat

        metrics = {
        # ---- Window definitions (seconds)
        "supine_start_s": float(supine_start),
        "supine_end_s": float(supine_end),
        "plateau_start_s": float(plateau_start),
        "plateau_end_s": float(plateau_end),
        "stand_onset_s": float(stand_onset),
        "sts_bp_plot_included": int(bp_plot_allowed),
        "sts_bp_plot_status": bp_plot_status,
        "supine_map_median": float(supine_map_median),
        "standing_map_median": float(standing_map_median),

        # ---- BP: Supine
        "mean_MAP_sup": float(mean_MAP_sup),
        "mean_sys_sup": float(mean_sys_sup),
        "mean_dia_sup": float(mean_dia_sup),
        "mean_pp_sup":  float(mean_pp_sup),

        # ---- BP: Plateau
        "mean_MAP_plt": float(mean_MAP_plt),
        "mean_sys_plt": float(mean_sys_plt),
        "mean_dia_plt": float(mean_dia_plt),
        "mean_pp_plt":  float(mean_pp_plt),

        # ---- BP: Deltas (plateau - supine)
        "dMAP": float(dMAP),
        "dSYS": float(dSYS),
        "dDIA": float(dDIA),
        "dPP":  float(dPP),

        # ---- ECG: Supine
        "mean_RR_sup": float(mean_RR_sup),
        "mean_HR_sup": float(mean_HR_sup),
        "RMSSD_sup_ms": float(RMSSD_sup),

        # ---- ECG: Plateau
        "mean_RR_plt": float(mean_RR_plt),
        "mean_HR_plt": float(mean_HR_plt),
        "RMSSD_plt_ms": float(RMSSD_plt),

        # ---- ECG: Deltas
        "dHR": float(dHR),
        "dRMSSD_ms": float(dRMSSD),

        # ---- Doppler: optional mean velocity
        "mean_doppler_sup": float(mean_doppler_sup),
        "mean_doppler_plt": float(mean_doppler_plt),
        "d_doppler": float(d_doppler),
        "doppler_quality_supine": float(doppler_quality_supine),
        "doppler_quality_standing": float(doppler_quality_standing),
        "doppler_quality_threshold": float(args.doppler_quality_threshold),
        "doppler_plot_included": int(doppler_plot_allowed),
        "doppler_status": doppler_status,
        "doppler_channel_index": int(doppler_idx) if doppler_idx is not None else -1,

        # ---- Counts / QC
        "n_bp_peaks": int(len(peaks)),
        "n_bp_troughs": int(len(troughs)),
        "n_rpeaks": int(len(rpeaks)),
        "n_cal_segments": int(len(idx1)),
        "n_rr_sup": int(len(rr_sup)),
        "n_rr_plt": int(len(rr_plt)),

        # ---- Indices arrays (debug)
        "bp_peaks": peaks.astype(np.int32),
        "bp_troughs": troughs.astype(np.int32),
        "rpeaks": rpeaks.astype(np.int32),
        "cal_idx1": np.array(idx1, dtype=np.int32),
        "cal_idx2": np.array(idx2, dtype=np.int32),

        # ---- Metadata
        "fs": float(fs),
        "sub_id": sub_id,
        "ses_id": ses_id,    "acq_path": str(acq_path),
    }


        savemat(str(out_mat), metrics, do_compression=True)
        print(f"[OK] Saved metrics bundle: {out_mat}")

if __name__ == "__main__":
    main()
