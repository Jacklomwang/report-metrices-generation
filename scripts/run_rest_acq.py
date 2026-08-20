#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import bioread
import neurokit2 as nk
import numpy as np

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.physio_qc.metrics.blood_pressure import process_bp
from src.physio_qc.metrics.doppler import calculate_doppler_metrics, process_doppler
from src.physio_qc.metrics.ecg import process_ecg
from src.physio_qc.metrics.etco2 import process_etco2
from src.physio_qc.metrics.rsp import process_rsp
from src.physio_qc.metrics.spirometry import process_breathmetrics
from src.physio_qc.utils.conversions import convert_voltage_to_mmhg_co2


ECG_PARAMS = {
    "powerline": 60, "method": "neurokit", "peak_method": "neurokit",
    "correct_artifacts": False, "calculate_quality": False, "rate_method": "monotone_cubic",
    "lowcut": 0.5, "highcut": 45.0, "filter_type": "butterworth", "filter_order": 5,
    "apply_lowcut": True, "apply_highcut": True,
}
BP_PARAMS = {
    "filter_method": "bessel_25hz", "filter_order": 3, "cutoff_freq": 25,
    "peak_method": "delineator", "prominence": 10, "detect_calibration": True,
    "calibration_threshold": 0.1, "calibration_min_duration": 1.0, "calibration_padding": 0.4,
}
RSP_PARAMS = {
    "method": "khodadad2018", "peak_method": "scipy", "rate_method": "monotone_cubic",
    "amplitude_method": "robust", "lowcut": 0.05, "highcut": 3.0,
    "filter_type": "butterworth", "filter_order": 5, "apply_lowcut": True,
    "apply_highcut": True, "rvt_method": "none",
}
ETCO2_PARAMS = {
    "peak_method": "diff", "min_peak_distance_s": 3.0, "min_prominence": 3.0,
    "sg_window_s": 0.2, "sg_poly": 2, "prom_adapt": False, "smooth_peaks": 3,
}
SPIROMETER_PARAMS = {
    "data_type": "humanAirflow", "zscore": 0, "baseline_method": "sliding",
    "simplify": 1, "verbose": 0, "exclude_outliers": 0, "volume_outlier_sd": 3.0,
    "exclude_duration_outliers": 0, "duration_outlier_sd": 3.0,
}
DOPPLER_PARAMS = {
    "filter_method": "sg_wavelet", "filter_order": 3, "cutoff_freq": 25,
    "lowcut": 0.5, "highcut": 15.0, "apply_lowcut": True, "apply_highcut": True,
    "sg_win": 0.1, "wavelet": "db6", "level": 10, "alpha": 4.0,
    "drop_levels": 1, "trend_win": 2.0,
}


def build_rest_acq_path(root: Path, sub_code: str, ses_num: str) -> Path:
    sub_id, ses_id = f"sub-{sub_code}", f"ses-{ses_num}"
    ses_dir = root / sub_id / ses_id
    for pattern in (
        f"{sub_id}_{ses_id}_task-rest_physio.acq",
        f"{sub_id}_{ses_id}_task-rest*physio*.acq",
        "*task-rest*physio*.acq", "*rest*.acq",
    ):
        candidates = sorted(ses_dir.glob(pattern))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No resting .acq found under: {ses_dir}")


def _name(channel) -> str:
    return str(getattr(channel, "name", "") or "")


def _find_channel(channels, patterns, excluded=()):
    for index, channel in enumerate(channels):
        name = _name(channel).lower()
        if any(term in name for term in patterns) and not any(term in name for term in excluded):
            return channel, index
    return None, None


def detect_channels(channels):
    """Apply Physio-QC name patterns, including the study's A-channel fallbacks."""
    return {
        "ecg": _find_channel(channels, ("ecg", "ekg", "cardiac", "heart")),
        # Do not use a bare "bp" substring: it would select the NIBP rate channel
        # before the continuous A10 pressure waveform in standard LCS files.
        "bp": _find_channel(channels, ("blood_pressure", "arterial_pressure", "abp", "a10", "a 10")),
        "rsp": _find_channel(
            channels, ("rsp", "resp", "respiratory", "breathing", "breath"),
            excluded=("pneumotach", "respflow", "maskflow", "mask_flow"),
        ),
        "spirometer": _find_channel(
            channels, ("spirometer", "spiro", "pneumotach", "respflow", "maskflow", "mask_flow")
        ),
        "etco2": _find_channel(channels, ("co2", "etco2", "carbon_dioxide", "a8", "a 8")),
        "doppler": _find_channel(channels, ("doppler", "a6", "a 6", "a5", "a 5")),
    }


def _legacy_channel(channels, number: int | None, one_based: bool):
    if number is None:
        return None, None
    index = number - 1 if one_based else number
    return (channels[index], index) if 0 <= index < len(channels) else (None, None)


def _finite_mean(values) -> float:
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def compute_doppler_noisy_windows(
    signal_length, sampling_rate, trough_indices, beat_quality_scores,
    window_sec=10.0, step_sec=5.0, quality_threshold=0.8,
):
    """Classify windows using Physio-QC's time-weighted beat-quality rule."""
    noisy_mask = np.zeros(max(int(signal_length), 0), dtype=bool)
    if signal_length <= 0 or sampling_rate <= 0:
        return [], noisy_mask
    troughs = np.sort(np.asarray(trough_indices, dtype=int).ravel())
    troughs = troughs[(troughs >= 0) & (troughs < signal_length)]
    scores = np.asarray(beat_quality_scores, dtype=float).ravel()
    n_beats = min(len(scores), max(len(troughs) - 1, 0))
    if n_beats <= 0:
        return [], noisy_mask
    beat_starts = troughs[:n_beats] / sampling_rate
    beat_ends = troughs[1:n_beats + 1] / sampling_rate
    scores = scores[:n_beats]
    duration = max(0.0, (signal_length - 1.0) / sampling_rate)
    noisy_windows, start_t = [], 0.0
    while start_t < duration:
        end_t = min(start_t + window_sec, duration)
        overlap = np.maximum(0.0, np.minimum(beat_ends, end_t) - np.maximum(beat_starts, start_t))
        valid = (overlap > 0) & np.isfinite(scores)
        if np.any(valid):
            quality = float(np.sum(scores[valid] * overlap[valid]) / np.sum(overlap[valid]))
            if quality < quality_threshold:
                noisy_windows.append((start_t, end_t))
                first = max(0, int(np.floor(start_t * sampling_rate)))
                last = min(signal_length, int(np.ceil(end_t * sampling_rate)))
                noisy_mask[first:last] = True
        start_t += step_sec
    return noisy_windows, noisy_mask


def _clean_doppler_metrics(result, sampling_rate, quality_threshold):
    filtered = np.asarray(result["filtered"], dtype=float)
    peaks = np.asarray(result["current_peaks"], dtype=int)
    troughs = np.asarray(result["current_troughs"], dtype=int)
    scores = np.asarray(result.get("beat_scores", []), dtype=float)
    windows, noisy_mask = compute_doppler_noisy_windows(
        len(filtered), sampling_rate, troughs, scores, quality_threshold=quality_threshold
    )
    clean_peaks = peaks[~noisy_mask[peaks]]
    clean_troughs = troughs[~noisy_mask[troughs]]
    metrics = calculate_doppler_metrics(filtered, clean_peaks, clean_troughs, sampling_rate)
    n_beats = min(len(scores), max(len(troughs) - 1, 0))
    clean_scores = []
    for index in range(n_beats):
        start, end = max(0, troughs[index]), min(len(noisy_mask), troughs[index + 1])
        if end > start and not np.any(noisy_mask[start:end]) and np.isfinite(scores[index]):
            clean_scores.append(scores[index])
    metrics.update({
        "peaks": clean_peaks, "troughs": clean_troughs,
        "mean_quality": _finite_mean(clean_scores),
        "noisy_windows": np.asarray(windows, dtype=float).reshape((-1, 2)),
        "noisy_percent": float(100.0 * np.mean(noisy_mask)) if noisy_mask.size else np.nan,
    })
    return metrics


def save_resting_figures(task_out: Path, fs: float, ecg_result, bp_result):
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if ecg_result:
        time = np.arange(len(ecg_result["hr_interpolated"])) / fs
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(time, ecg_result["hr_interpolated"], linewidth=0.8)
        ax.set(title="Resting: Derived Heart Rate", xlabel="Time (s)", ylabel="HR (bpm)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(task_out / "resting_hr.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    if bp_result:
        fig, ax = plt.subplots(figsize=(14, 4))
        time = bp_result["time_4hz"]
        ax.plot(time, bp_result["sbp_4hz"], linewidth=0.8, label="SBP")
        ax.plot(time, bp_result["dbp_4hz"], linewidth=0.8, label="DBP")
        ax.plot(time, bp_result["map_4hz"], linewidth=0.9, label="MAP")
        ax.set(title="Resting: Derived Blood Pressure", xlabel="Time (s)", ylabel="Pressure (mmHg)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(task_out / "resting_BP.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Process all resting-state report metrics from an ACQ file")
    parser.add_argument("--root", default="/export02/projects/LCS/01_physio")
    parser.add_argument("--sub", required=True, help="Subject code, for example 2062")
    parser.add_argument("--ses", default="1", help="Session number")
    parser.add_argument("--out_root", default="derived")
    parser.add_argument("--save", action="store_true", help="Save MAT metrics and QC figures")
    parser.add_argument("--no_save", action="store_true", help="Process without writing outputs")
    parser.add_argument("--ecg_method", default="neurokit", help="ECG cleaning method; peak detection remains NeuroKit")
    parser.add_argument("--doppler_quality_threshold", type=float, default=0.8)
    parser.add_argument("--ecg_ch", type=int, help="Deprecated fallback when ECG name detection fails")
    parser.add_argument("--bp_ch", type=int, help="Deprecated fallback when BP name detection fails")
    parser.add_argument("--one_based", action="store_true", help="Interpret deprecated fallback channels as 1-based")
    args = parser.parse_args()

    sub_id, ses_id = f"sub-{args.sub}", f"ses-{args.ses}"
    task_out = Path(args.out_root) / sub_id / ses_id / "rest"
    acq_path = build_rest_acq_path(Path(args.root), args.sub, args.ses)
    print(f"[INFO] Subject: {sub_id}  Session: {ses_id}\n[INFO] Rest ACQ: {acq_path}")
    acq = bioread.read_file(str(acq_path))
    fs = float(acq.samples_per_second)
    detected = detect_channels(acq.channels)
    if detected["ecg"][0] is None:
        detected["ecg"] = _legacy_channel(acq.channels, args.ecg_ch, args.one_based)
    if detected["bp"][0] is None:
        detected["bp"] = _legacy_channel(acq.channels, args.bp_ch, args.one_based)
    print(f"[INFO] Sampling rate: {fs} Hz  |  Channels: {len(acq.channels)}")
    for modality, (channel, index) in detected.items():
        label = f"index={index} name={_name(channel)}" if channel is not None else "not found"
        print(f"[INFO] {modality.upper():10s} {label}")

    metrics = {"fs": fs, "sub_id": sub_id, "ses_id": ses_id, "acq_path": str(acq_path),
               "processing_source": "vendored_physio_qc_d56fa44"}
    for modality, (channel, _) in detected.items():
        metrics[f"channel_{modality}"] = _name(channel) if channel is not None else ""
    ecg_result = bp_result = None

    channel = detected["ecg"][0]
    try:
        if channel is None:
            raise ValueError("ECG channel not found")
        params = {**ECG_PARAMS, "method": args.ecg_method, "cleaning_method": args.ecg_method}
        ecg_result = process_ecg(np.asarray(channel.data, dtype=float), fs, params)
        if not ecg_result or ecg_result["n_peaks"] < 3:
            raise ValueError("insufficient R-peaks")
        rpeaks = np.asarray(ecg_result["current_r_peaks"], dtype=int)
        rr = np.diff(rpeaks) / fs
        rr = rr[(rr >= 0.3) & (rr <= 2.0)]
        hrv_time = nk.hrv_time(rpeaks, sampling_rate=fs, show=False)
        hrv_frequency = nk.hrv_frequency(rpeaks, sampling_rate=fs, show=False)
        metrics.update({"mean_RR": _finite_mean(rr), "mean_HR": _finite_mean(ecg_result["hr_bpm"]),
                        "RMSSD_ms": float(hrv_time["HRV_RMSSD"].iloc[0]),
                        "LF_HF": float(hrv_frequency["HRV_LFHF"].iloc[0]),
                        "n_rpeaks": int(len(rpeaks)), "rpeaks": rpeaks.astype(np.int32),
                        "ecg_status": "available"})
    except Exception as exc:
        print(f"[WARN] ECG processing unavailable: {exc}")
        metrics.update({"mean_RR": np.nan, "mean_HR": np.nan, "RMSSD_ms": np.nan, "LF_HF": np.nan,
                        "n_rpeaks": 0, "rpeaks": np.array([], dtype=np.int32), "ecg_status": str(exc)})

    channel = detected["bp"][0]
    try:
        if channel is None:
            raise ValueError("blood-pressure channel not found")
        bp_result = process_bp(np.asarray(channel.data, dtype=float), fs, BP_PARAMS)
        if not bp_result:
            raise ValueError("insufficient BP peaks/troughs")
        peaks = np.asarray(bp_result["current_peaks"], dtype=int)
        troughs = np.asarray(bp_result["current_troughs"], dtype=int)
        metrics.update({"mean_MAP": float(bp_result["mean_mbp"]), "mean_sysBP": float(bp_result["mean_sbp"]),
                        "mean_diaBP": float(bp_result["mean_dbp"]),
                        "mean_pulseBP": float(bp_result["mean_sbp"] - bp_result["mean_dbp"]),
                        "n_peaks": int(len(peaks)), "n_troughs": int(len(troughs)),
                        "bp_peaks": peaks.astype(np.int32), "bp_troughs": troughs.astype(np.int32),
                        "bp_status": "available"})
    except Exception as exc:
        print(f"[WARN] BP processing unavailable: {exc}")
        metrics.update({"mean_MAP": np.nan, "mean_sysBP": np.nan, "mean_diaBP": np.nan,
                        "mean_pulseBP": np.nan, "n_peaks": 0, "n_troughs": 0,
                        "bp_peaks": np.array([], dtype=np.int32), "bp_troughs": np.array([], dtype=np.int32),
                        "bp_status": str(exc)})

    channel = detected["rsp"][0]
    try:
        if channel is None:
            raise ValueError("respiration channel not found")
        result = process_rsp(np.asarray(channel.data, dtype=float), fs, RSP_PARAMS)
        if not result:
            raise ValueError("insufficient respiratory cycles")
        metrics.update({"mean_br": float(result["mean_br"]), "std_br": float(result["std_br"]),
                        "n_breaths_rsp": int(result["n_breaths"]), "rsp_status": "available"})
    except Exception as exc:
        print(f"[WARN] RSP processing unavailable: {exc}")
        metrics.update({"mean_br": np.nan, "std_br": np.nan, "n_breaths_rsp": 0, "rsp_status": str(exc)})

    channel = detected["etco2"][0]
    try:
        if channel is None:
            raise ValueError("CO2 channel not found")
        co2 = np.asarray(channel.data, dtype=float)
        if "mmhg" not in _name(channel).lower():
            co2 = convert_voltage_to_mmhg_co2(co2)
        result = process_etco2(co2, fs, ETCO2_PARAMS)
        metrics.update({"mean_etco2": _finite_mean(result["etco2_envelope"]),
                        "n_etco2_peaks": int(len(result["current_peaks"])), "etco2_status": "available"})
    except Exception as exc:
        print(f"[WARN] ETCO2 processing unavailable: {exc}")
        metrics.update({"mean_etco2": np.nan, "n_etco2_peaks": 0, "etco2_status": str(exc)})

    channel = detected["spirometer"][0]
    try:
        if channel is None:
            raise ValueError("pneumotach/spirometer channel not found")
        result = process_breathmetrics(np.asarray(channel.data, dtype=float), fs, SPIROMETER_PARAMS)
        if not result:
            raise ValueError("BreathMetrics processing failed")
        tidal_ml = float(result.get("mean_tidal_volume", np.nan))
        metrics.update({"mean_tidal_volume_ml": tidal_ml,
                        "mean_tidal_volume_l": tidal_ml / 1000.0 if np.isfinite(tidal_ml) else np.nan,
                        "mean_minute_ventilation": float(result.get("mean_minute_ventilation", np.nan)),
                        "n_breaths_flow": int(result.get("n_breaths", 0)), "spirometer_status": "available"})
    except Exception as exc:
        print(f"[WARN] Respiratory-flow processing unavailable: {exc}")
        metrics.update({"mean_tidal_volume_ml": np.nan, "mean_tidal_volume_l": np.nan,
                        "mean_minute_ventilation": np.nan, "n_breaths_flow": 0,
                        "spirometer_status": str(exc)})

    channel = detected["doppler"][0]
    try:
        if channel is None:
            raise ValueError("Doppler channel not found")
        result = process_doppler(np.asarray(channel.data, dtype=float), fs, DOPPLER_PARAMS)
        if not result:
            raise ValueError("insufficient Doppler peaks/troughs")
        clean = _clean_doppler_metrics(result, fs, args.doppler_quality_threshold)
        metrics.update({"doppler_mean_peak": float(clean["mean_peak"]),
                        "doppler_mean_trough": float(clean["mean_trough"]),
                        "doppler_mean_flow": float(clean["mean_mbp"]),
                        "doppler_mean_quality": float(clean["mean_quality"]),
                        "doppler_noisy_percent": float(clean["noisy_percent"]),
                        "doppler_quality_threshold": float(args.doppler_quality_threshold),
                        "doppler_n_peaks_clean": int(len(clean["peaks"])),
                        "doppler_n_troughs_clean": int(len(clean["troughs"])),
                        "doppler_peaks_clean": clean["peaks"].astype(np.int32),
                        "doppler_troughs_clean": clean["troughs"].astype(np.int32),
                        "doppler_noisy_windows": clean["noisy_windows"], "doppler_status": "available"})
    except Exception as exc:
        print(f"[WARN] Doppler processing unavailable: {exc}")
        metrics.update({"doppler_mean_peak": np.nan, "doppler_mean_trough": np.nan,
                        "doppler_mean_flow": np.nan, "doppler_mean_quality": np.nan,
                        "doppler_noisy_percent": np.nan, "doppler_quality_threshold": args.doppler_quality_threshold,
                        "doppler_n_peaks_clean": 0, "doppler_n_troughs_clean": 0,
                        "doppler_peaks_clean": np.array([], dtype=np.int32),
                        "doppler_troughs_clean": np.array([], dtype=np.int32),
                        "doppler_noisy_windows": np.empty((0, 2)), "doppler_status": str(exc)})

    print("\n===== REST DERIVED INDICES =====")
    print(f"ECG: HR={metrics['mean_HR']:.2f} bpm  RMSSD={metrics['RMSSD_ms']:.2f} ms  LF/HF={metrics['LF_HF']:.2f}")
    print(f"BP:  SBP={metrics['mean_sysBP']:.2f}  DBP={metrics['mean_diaBP']:.2f}  MAP={metrics['mean_MAP']:.2f} mmHg")
    print(f"RSP: BR={metrics['mean_br']:.2f}/min  ETCO2={metrics['mean_etco2']:.2f} mmHg  "
          f"TV={metrics['mean_tidal_volume_ml']:.2f} mL  MV={metrics['mean_minute_ventilation']:.2f} L/min")
    print(f"Doppler: peak={metrics['doppler_mean_peak']:.2f}  trough={metrics['doppler_mean_trough']:.2f}  "
          f"flow={metrics['doppler_mean_flow']:.2f} cm/s  quality={metrics['doppler_mean_quality']:.3f}  "
          f"noisy={metrics['doppler_noisy_percent']:.2f}%")
    if args.no_save:
        return
    if args.save:
        from scipy.io import savemat
        task_out.mkdir(parents=True, exist_ok=True)
        save_resting_figures(task_out, fs, ecg_result, bp_result)
        out_mat = task_out / "rest_metrics.mat"
        savemat(str(out_mat), metrics, do_compression=True)
        print(f"[OK] Saved metrics bundle: {out_mat}")


if __name__ == "__main__":
    main()
