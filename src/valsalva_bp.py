from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import medfilt


EPOCH_START_SECONDS = -15.0
TASK_END_SECONDS = 16.0
EPOCH_END_SECONDS = 45.0
MAP_MEDIAN_FILTER_SAMPLES = 5
HR_MEDIAN_FILTER_SAMPLES = 5
HR_OUTLIER_WINDOW_SAMPLES = 9
HR_OUTLIER_MIN_DEVIATION_BPM = 10.0
HR_MAX_END_SECONDS = TASK_END_SECONDS + 5.0


def prepare_hr_traces(hr_signal, sampling_rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample HR to 4 Hz and return raw and robust median-filtered traces."""
    hr_signal = np.asarray(hr_signal, dtype=float)
    source_time = np.arange(len(hr_signal)) / float(sampling_rate)
    time_4hz = np.arange(0.0, float(source_time[-1]) + 0.001, 0.25)
    finite = np.isfinite(hr_signal)
    if finite.sum() < 2:
        empty = np.full(len(time_4hz), np.nan)
        return time_4hz, empty, empty.copy()
    raw_4hz = np.interp(time_4hz, source_time[finite], hr_signal[finite])
    local_median = median_filter(raw_4hz, size=HR_OUTLIER_WINDOW_SAMPLES, mode="nearest")
    deviation = np.abs(raw_4hz - local_median)
    local_mad = median_filter(deviation, size=HR_OUTLIER_WINDOW_SAMPLES, mode="nearest")
    threshold = np.maximum(HR_OUTLIER_MIN_DEVIATION_BPM, 6.0 * 1.4826 * local_mad)
    artifact_rejected = raw_4hz.copy()
    artifact_rejected[deviation > threshold] = local_median[deviation > threshold]
    filtered_4hz = medfilt(artifact_rejected, kernel_size=HR_MEDIAN_FILTER_SAMPLES)
    return time_4hz, raw_4hz, filtered_4hz


def valsalva_hr_metrics(time_4hz, filtered_hr, onset_sec: float) -> dict[str, float]:
    """Calculate the group-analysis Valsalva ratio from the filtered HR trace."""
    time_4hz = np.asarray(time_4hz, dtype=float)
    filtered_hr = np.asarray(filtered_hr, dtype=float)
    relative = time_4hz - float(onset_sec)
    valid = np.isfinite(filtered_hr)
    task = (relative >= 0.0) & (relative <= HR_MAX_END_SECONDS) & valid
    recovery = (relative > TASK_END_SECONDS) & (relative <= EPOCH_END_SECONDS) & valid
    if not task.any() or not recovery.any():
        return {
            "valsalva_ratio": np.nan,
            "max_hr_task": np.nan,
            "min_hr_recovery": np.nan,
            "max_hr_time": np.nan,
            "min_hr_time": np.nan,
        }
    task_indices = np.flatnonzero(task)
    recovery_indices = np.flatnonzero(recovery)
    max_index = int(task_indices[np.argmax(filtered_hr[task_indices])])
    min_index = int(recovery_indices[np.argmin(filtered_hr[recovery_indices])])
    maximum = float(filtered_hr[max_index])
    minimum = float(filtered_hr[min_index])
    return {
        "valsalva_ratio": maximum / minimum if minimum > 0 else np.nan,
        "max_hr_task": maximum,
        "min_hr_recovery": minimum,
        "max_hr_time": float(relative[max_index]),
        "min_hr_time": float(relative[min_index]),
    }

def bp_phase_landmarks(bp_result: dict, onset_sec: float) -> dict[str, object]:
    """Identify MAP-defined Valsalva landmarks using the group-analysis windows."""
    required = ("time_4hz", "sbp_4hz", "map_4hz")
    if not bp_result or any(key not in bp_result for key in required):
        return {}

    time = np.asarray(bp_result["time_4hz"], dtype=float)
    sbp = np.asarray(bp_result["sbp_4hz"], dtype=float)
    map_values = np.asarray(bp_result["map_4hz"], dtype=float)
    if not (len(time) == len(sbp) == len(map_values)) or not len(time):
        return {}

    relative = time - float(onset_sec)
    valid = np.isfinite(relative) & np.isfinite(sbp) & np.isfinite(map_values)
    baseline = (relative >= EPOCH_START_SECONDS) & (relative < 0.0) & valid
    if not baseline.any():
        return {}

    filled_map = map_values.copy()
    finite = np.isfinite(filled_map)
    if not finite.any():
        return {}
    indices = np.arange(len(filled_map))
    filled_map[~finite] = np.interp(indices[~finite], indices[finite], filled_map[finite])
    smooth_map = medfilt(filled_map, kernel_size=MAP_MEDIAN_FILTER_SAMPLES)

    def extreme(start: float, end: float, mode: str) -> int | None:
        candidates = np.flatnonzero((relative >= start) & (relative <= end) & valid)
        if not len(candidates):
            return None
        offset = np.argmax(smooth_map[candidates]) if mode == "max" else np.argmin(smooth_map[candidates])
        return int(candidates[offset])

    phase1 = extreme(0.0, 3.0, "max")
    if phase1 is None:
        return {}
    phase2_nadir = extreme(float(relative[phase1]), 10.0, "min")
    if phase2_nadir is None:
        return {}
    phase2_late = extreme(max(13.5, float(relative[phase2_nadir])), 15.5, "max")
    if phase2_late is None:
        return {}
    phase3_nadir = extreme(float(relative[phase2_late]), float(relative[phase2_late]) + 3.0, "min")
    if phase3_nadir is None:
        return {}
    phase4_peak = extreme(float(relative[phase3_nadir]), float(relative[phase3_nadir]) + 10.0, "max")
    if phase4_peak is None:
        return {}

    event_indices = {
        "phase1_max": phase1,
        "phase2_nadir": phase2_nadir,
        "phase2_late_max": phase2_late,
        "phase3_nadir": phase3_nadir,
        "phase4_max": phase4_peak,
    }
    return {
        "baseline_sbp": float(np.mean(sbp[baseline])),
        "baseline_map": float(np.mean(map_values[baseline])),
        "events": {
            name: {
                "time": float(relative[index]),
                "sbp": float(sbp[index]),
                "map": float(map_values[index]),
            }
            for name, index in event_indices.items()
        },
    }


def phase_summary_metrics(landmarks: dict[str, object]) -> dict[str, float]:
    if not landmarks or "events" not in landmarks:
        return {
            "baseline_sbp": np.nan,
            "baseline_map": np.nan,
            "sbp_phase1_from_baseline": np.nan,
            "sbp_phase2_early_fall": np.nan,
            "sbp_phase2_late_recovery": np.nan,
            "sbp_phase3_drop": np.nan,
            "sbp_phase4_rise": np.nan,
            "map_phase1_from_baseline": np.nan,
            "map_phase2_early_fall": np.nan,
            "map_phase2_late_recovery": np.nan,
            "map_phase3_drop": np.nan,
            "map_phase4_rise": np.nan,
            "map_phase2_drop": np.nan,
            "map_phase4_overshoot": np.nan,
        }
    events = landmarks["events"]
    return {
        "baseline_sbp": float(landmarks["baseline_sbp"]),
        "baseline_map": float(landmarks["baseline_map"]),
        "sbp_phase1_from_baseline": float(events["phase1_max"]["sbp"] - landmarks["baseline_sbp"]),
        "sbp_phase2_early_fall": float(events["phase2_nadir"]["sbp"] - events["phase1_max"]["sbp"]),
        "sbp_phase2_late_recovery": float(events["phase2_late_max"]["sbp"] - events["phase2_nadir"]["sbp"]),
        "sbp_phase3_drop": float(events["phase3_nadir"]["sbp"] - events["phase2_late_max"]["sbp"]),
        "sbp_phase4_rise": float(events["phase4_max"]["sbp"] - events["phase3_nadir"]["sbp"]),
        "map_phase1_from_baseline": float(events["phase1_max"]["map"] - landmarks["baseline_map"]),
        "map_phase2_early_fall": float(events["phase2_nadir"]["map"] - events["phase1_max"]["map"]),
        "map_phase2_late_recovery": float(events["phase2_late_max"]["map"] - events["phase2_nadir"]["map"]),
        "map_phase3_drop": float(events["phase3_nadir"]["map"] - events["phase2_late_max"]["map"]),
        "map_phase4_rise": float(events["phase4_max"]["map"] - events["phase3_nadir"]["map"]),
        "map_phase2_drop": float(events["phase1_max"]["map"] - events["phase2_nadir"]["map"]),
        "map_phase4_overshoot": float(events["phase4_max"]["map"] - landmarks["baseline_map"]),
    }


def save_valsalva_hr_bp_figure(
    out_png: Path,
    hr_time_4hz,
    raw_hr_4hz,
    filtered_hr_4hz,
    raw_bp,
    bp_result: dict,
    sampling_rate: float,
    onset_sec: float,
    landmarks: dict[str, object],
) -> None:
    """Plot synchronized HR and BP traces for the selected repetition."""
    import os
    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hr_time = np.asarray(hr_time_4hz, dtype=float) - float(onset_sec)
    raw_hr = np.asarray(raw_hr_4hz, dtype=float)
    filtered_hr = np.asarray(filtered_hr_4hz, dtype=float)
    hr_mask = (hr_time >= EPOCH_START_SECONDS) & (hr_time <= EPOCH_END_SECONDS)

    raw_bp = np.asarray(raw_bp, dtype=float)
    raw_bp_time = np.arange(len(raw_bp)) / float(sampling_rate) - float(onset_sec)
    raw_bp_mask = (raw_bp_time >= EPOCH_START_SECONDS) & (raw_bp_time <= EPOCH_END_SECONDS)
    map_time = np.asarray(bp_result["time_4hz"], dtype=float) - float(onset_sec)
    map_values = np.asarray(bp_result["map_4hz"], dtype=float)
    map_mask = (map_time >= EPOCH_START_SECONDS) & (map_time <= EPOCH_END_SECONDS)
    if not hr_mask.any() or not raw_bp_mask.any() or not map_mask.any():
        raise ValueError("selected Valsalva BP window contains no samples")

    fig, (hr_ax, bp_ax) = plt.subplots(2, 1, figsize=(15.5, 8.4), sharex=True)
    hr_ax.plot(hr_time[hr_mask], raw_hr[hr_mask], color="#64748B", linewidth=1.1,
               alpha=0.22, label="Unfiltered HR")
    hr_ax.plot(hr_time[hr_mask], filtered_hr[hr_mask], color="#1D4ED8", linewidth=2.4,
               label="Artifact-rejected median HR")
    hr_ax.set_title("Heart Rate", fontsize=14, loc="left", pad=11)
    hr_ax.set_ylabel("Heart rate (bpm)", fontsize=11)
    hr_ax.grid(axis="y", alpha=0.18)
    hr_ax.legend(
        loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2,
        fontsize=10, frameon=False, borderaxespad=0.0,
    )

    events = landmarks.get("events", {}) if landmarks else {}
    boundaries = [
        (0.0, events.get("phase1_max", {}).get("time"), "I", "#FDE68A"),
        (events.get("phase1_max", {}).get("time"), events.get("phase2_nadir", {}).get("time"), "II early", "#BAE6FD"),
        (events.get("phase2_nadir", {}).get("time"), events.get("phase2_late_max", {}).get("time"), "II late", "#DDD6FE"),
        (events.get("phase2_late_max", {}).get("time"), events.get("phase3_nadir", {}).get("time"), "III", "#FECDD3"),
        (events.get("phase3_nadir", {}).get("time"), events.get("phase4_max", {}).get("time"), "IV", "#FED7AA"),
        (events.get("phase4_max", {}).get("time"), EPOCH_END_SECONDS, "Recovery", "#D1FAE5"),
    ]
    for index, (start, end, label, color) in enumerate(boundaries):
        if start is None or end is None or end <= start:
            continue
        bp_ax.axvspan(start, end, color=color, alpha=0.22, linewidth=0)
        text_y = 0.97 if index % 2 == 0 else 0.91
        bp_ax.text((start + end) / 2.0, text_y, label, transform=bp_ax.get_xaxis_transform(),
                   ha="center", va="top", fontsize=10, fontweight="bold")

    bp_ax.plot(raw_bp_time[raw_bp_mask], raw_bp[raw_bp_mask], color="#64748B", linewidth=0.75,
               alpha=0.22, label="Raw continuous BP")
    bp_ax.plot(map_time[map_mask], map_values[map_mask], color="#047857", linewidth=2.5,
               label="Derived MAP")
    for event in events.values():
        event_time = event["time"]
        bp_ax.plot(event_time, event["map"], "o", color="#0F766E", markersize=5, label="_nolegend_")

    for axis in (hr_ax, bp_ax):
        axis.axvline(0.0, color="#C2413B", linestyle="--", linewidth=1.5,
                     label="Strain start" if axis is bp_ax else None)
        axis.axvline(TASK_END_SECONDS, color="#B45309", linestyle="--", linewidth=1.5,
                     label="Strain release" if axis is bp_ax else None)
    hr_ax.text(
        0.0, 1.015, "Strain start", transform=hr_ax.get_xaxis_transform(),
        ha="center", va="bottom", color="#9F2D2A", fontsize=10,
        fontweight="bold", clip_on=False,
    )
    hr_ax.text(
        TASK_END_SECONDS, 1.015, "Release", transform=hr_ax.get_xaxis_transform(),
        ha="center", va="bottom", color="#92400E", fontsize=10,
        fontweight="bold", clip_on=False,
    )
    bp_ax.set_title("Blood Pressure and MAP-Defined Phases", fontsize=14, loc="left", pad=11)
    bp_ax.set_xlabel("Time relative to strain start (s)", fontsize=12)
    bp_ax.set_ylabel("Blood pressure (mmHg)", fontsize=11)
    bp_ax.set_xlim(EPOCH_START_SECONDS, EPOCH_END_SECONDS)
    bp_ax.grid(axis="y", alpha=0.18)
    bp_ax.legend(
        loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=4,
        fontsize=10, frameon=False, borderaxespad=0.0,
    )
    fig.suptitle("Valsalva Cardiovascular Response", fontsize=17, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.5)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
