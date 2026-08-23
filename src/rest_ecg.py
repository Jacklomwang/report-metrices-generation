from __future__ import annotations

from pathlib import Path

import neurokit2 as nk
import numpy as np


INTERVAL_SPECS = {
    "ecg_p_duration_ms": ("ECG_P_Onsets", "ECG_P_Offsets", 40.0, 180.0),
    "ecg_qrs_duration_ms": ("ECG_R_Onsets", "ECG_R_Offsets", 40.0, 200.0),
    "ecg_pq_time_ms": ("ECG_P_Onsets", "ECG_R_Onsets", 80.0, 320.0),
    "ecg_qt_time_ms": ("ECG_R_Onsets", "ECG_T_Offsets", 200.0, 700.0),
}

LANDMARK_KEYS = (
    "ECG_P_Onsets",
    "ECG_P_Offsets",
    "ECG_R_Onsets",
    "ECG_R_Offsets",
    "ECG_T_Offsets",
)


def calculate_interval_metrics(waves: dict, sampling_rate: float) -> dict[str, float]:
    """Return robust median ECG intervals from aligned NeuroKit landmarks."""
    metrics: dict[str, float] = {}
    valid_counts = []
    for metric, (start_key, end_key, minimum, maximum) in INTERVAL_SPECS.items():
        start = np.asarray(waves.get(start_key, []), dtype=float)
        end = np.asarray(waves.get(end_key, []), dtype=float)
        count = min(len(start), len(end))
        durations = (end[:count] - start[:count]) / float(sampling_rate) * 1000.0
        valid = np.isfinite(durations) & (durations >= minimum) & (durations <= maximum)
        metrics[metric] = float(np.median(durations[valid])) if valid.any() else np.nan
        valid_counts.append(int(valid.sum()))
    metrics["ecg_interval_n_beats"] = min(valid_counts) if valid_counts else 0
    return metrics


def average_ecg_waveform(
    cleaned_ecg,
    r_peaks,
    sampling_rate: float,
    pre_seconds: float = 0.35,
    post_seconds: float = 0.50,
) -> dict[str, np.ndarray | int]:
    """Build a quality-screened arithmetic average of R-aligned ECG beats."""
    signal = np.asarray(cleaned_ecg, dtype=float)
    peaks = np.asarray(r_peaks, dtype=int)
    pre = int(round(pre_seconds * sampling_rate))
    post = int(round(post_seconds * sampling_rate))
    beats = []
    for peak in peaks:
        start, end = peak - pre, peak + post + 1
        if start < 0 or end > len(signal):
            continue
        beat = signal[start:end].copy()
        baseline_samples = max(1, int(round(0.05 * sampling_rate)))
        beat -= np.median(beat[:baseline_samples])
        if np.isfinite(beat).all():
            beats.append(beat)
    if not beats:
        raise ValueError("no complete ECG beats available for averaging")

    beat_array = np.asarray(beats)
    template = np.median(beat_array, axis=0)
    template_centered = template - np.mean(template)
    template_norm = np.linalg.norm(template_centered)
    correlations = np.full(len(beat_array), np.nan)
    if template_norm > 0:
        for index, beat in enumerate(beat_array):
            centered = beat - np.mean(beat)
            norm = np.linalg.norm(centered)
            if norm > 0:
                correlations[index] = float(np.dot(centered, template_centered) / (norm * template_norm))
    keep = np.isfinite(correlations) & (correlations >= 0.80)
    if keep.sum() < min(3, len(beat_array)):
        keep = np.ones(len(beat_array), dtype=bool)
    accepted = beat_array[keep]
    time_ms = (np.arange(-pre, post + 1) / float(sampling_rate)) * 1000.0
    return {
        "time_ms": time_ms,
        "mean": np.mean(accepted, axis=0),
        "std": np.std(accepted, axis=0),
        "n_beats": int(len(accepted)),
    }


def calculate_resting_ecg_morphology(cleaned_ecg, r_peaks, sampling_rate: float) -> dict:
    """Delineate resting ECG and calculate intervals plus an average beat."""
    peaks = np.asarray(r_peaks, dtype=int)
    _, waves = nk.ecg_delineate(
        np.asarray(cleaned_ecg, dtype=float),
        rpeaks=peaks,
        sampling_rate=sampling_rate,
        method="dwt",
        show=False,
    )
    result = calculate_interval_metrics(waves, sampling_rate)
    result["average"] = average_ecg_waveform(cleaned_ecg, peaks, sampling_rate)
    for key in LANDMARK_KEYS:
        values = np.asarray(waves.get(key, []), dtype=float)
        count = min(len(values), len(peaks))
        relative = (values[:count] - peaks[:count]) / float(sampling_rate) * 1000.0
        result[f"{key}_relative_ms"] = float(np.nanmedian(relative)) if np.isfinite(relative).any() else np.nan
    return result


def save_average_ecg_figure(out_png: Path, morphology: dict) -> None:
    import os
    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    average = morphology["average"]
    time_ms = np.asarray(average["time_ms"], dtype=float)
    mean = np.asarray(average["mean"], dtype=float)
    std = np.asarray(average["std"], dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.fill_between(time_ms, mean - std, mean + std, color="#93C5FD", alpha=0.25, linewidth=0,
                    label="Between-beat SD")
    ax.plot(time_ms, mean, color="#1D4ED8", linewidth=2.2, label="Average ECG")
    waveform_min = float(np.nanmin(mean - std))
    waveform_max = float(np.nanmax(mean + std))
    waveform_span = max(waveform_max - waveform_min, 1e-6)
    p_duration_row = waveform_min - 0.10 * waveform_span
    qrs_duration_row = waveform_min - 0.24 * waveform_span
    pq_time_row = waveform_min - 0.38 * waveform_span
    qt_time_row = waveform_min - 0.52 * waveform_span
    bracket_color = "#0891B2"

    def landmark(key: str) -> float:
        return float(morphology.get(f"{key}_relative_ms", np.nan))

    def interval_bracket(start: float, end: float, y: float, label: str) -> None:
        if not (np.isfinite(start) and np.isfinite(end) and end > start):
            return
        cap = 0.035 * waveform_span
        ax.plot([start, end], [y, y], color=bracket_color, linewidth=2.0, solid_capstyle="butt")
        ax.plot([start, start], [y - cap, y + cap], color=bracket_color, linewidth=2.0)
        ax.plot([end, end], [y - cap, y + cap], color=bracket_color, linewidth=2.0)
        ax.text((start + end) / 2.0, y + 0.045 * waveform_span, label,
                color="#0E7490", fontsize=8, fontweight="bold", ha="center", va="bottom")

    p_onset = landmark("ECG_P_Onsets")
    p_offset = landmark("ECG_P_Offsets")
    qrs_onset = landmark("ECG_R_Onsets")
    qrs_offset = landmark("ECG_R_Offsets")
    t_offset = landmark("ECG_T_Offsets")
    interval_bracket(p_onset, p_offset, p_duration_row, "P duration")
    interval_bracket(qrs_onset, qrs_offset, qrs_duration_row, "QRS duration")
    interval_bracket(p_onset, qrs_onset, pq_time_row, "PQ time")
    interval_bracket(qrs_onset, t_offset, qt_time_row, "QT time")

    ax.axhline(0.0, color="#64748B", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.axvline(0.0, color="#1E293B", linewidth=0.7, alpha=0.35)
    ax.set_ylim(waveform_min - 0.62 * waveform_span, waveform_max + 0.08 * waveform_span)
    ax.set(title=f"Average Resting ECG ({average['n_beats']} beats)",
           xlabel="Time relative to R peak (ms)", ylabel="ECG amplitude (a.u.)")
    ax.minorticks_on()
    ax.grid(which="major", color="#F3B8B8", alpha=0.35, linewidth=0.7)
    ax.grid(which="minor", color="#F8DADA", alpha=0.28, linewidth=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
