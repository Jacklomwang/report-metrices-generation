"""
Blood pressure signal processing functions
Pure functions with no classes
"""

import neurokit2 as nk
import numpy as np
from scipy.signal import bessel, butter, filtfilt, find_peaks

from ..algorithms.bp_delineator import delineate_bp
from ..algorithms.quality_detection import detect_calibration_artifacts, filter_indices_outside_regions


def filter_bp(
    signal,
    sampling_rate,
    method="bessel_25hz",
    filter_order=3,
    cutoff_freq=25,
    filter_type="butterworth",
    lowcut=0.5,
    highcut=15.0,
    apply_lowcut=True,
    apply_highcut=True,
):
    """
    Filter blood pressure signal

    Parameters
    ----------
    signal : array
        Raw blood pressure signal
    sampling_rate : int
        Sampling rate in Hz
    method : str
        Filtering method ('bessel_25hz', 'butterworth', 'custom')
    filter_order : int
        Filter order
    cutoff_freq : float
        Cutoff frequency for lowpass methods (Hz)
    filter_type : str
        Filter type for custom filtering
    lowcut : float
        High-pass cutoff for custom filtering
    highcut : float
        Low-pass cutoff for custom filtering
    apply_lowcut : bool
        Apply high-pass filter (custom method only)
    apply_highcut : bool
        Apply low-pass filter (custom method only)

    Returns
    -------
    array
        Filtered blood pressure signal
    """
    if method == "bessel_25hz":
        wn = float(cutoff_freq) / (float(sampling_rate) / 2.0)
        wn = min(max(wn, 1e-6), 0.999999)
        b, a = bessel(filter_order, wn, btype="low", analog=False, output="ba", norm="phase")
        bp_filtered = filtfilt(b, a, signal)

    elif method == "butterworth":
        wn = float(cutoff_freq) / (float(sampling_rate) / 2.0)
        wn = min(max(wn, 1e-6), 0.999999)
        b, a = butter(filter_order, wn, btype="low", analog=False)
        bp_filtered = filtfilt(b, a, signal)

    elif method == "custom":
        bp_filtered = signal.copy()
        low = lowcut if apply_lowcut else None
        high = highcut if apply_highcut else None
        if (low is not None and low > 0) or (high is not None and high < sampling_rate / 2):
            bp_filtered = nk.signal_filter(
                bp_filtered,
                sampling_rate=sampling_rate,
                lowcut=low,
                highcut=high,
                method=filter_type,
                order=filter_order,
            )
    else:
        bp_filtered = signal.copy()

    return bp_filtered


def detect_bp_peaks(signal, sampling_rate, method="delineator", prominence=10):
    """
    Detect systolic peaks and diastolic troughs in blood pressure signal

    Parameters
    ----------
    signal : array
        Filtered blood pressure signal
    sampling_rate : int
        Sampling rate in Hz
    method : str
        Detection method ('delineator', 'prominence')
    prominence : float
        Prominence parameter for prominence method

    Returns
    -------
    dict
        Dictionary containing:
        - peaks: Systolic peak indices
        - troughs: Diastolic trough indices
        - dicrotic_notches: Dicrotic notch indices (delineator only, 0 if not found)
    """
    if method == "delineator":
        result = delineate_bp(signal, sampling_rate)
        return {"peaks": result["peaks"], "troughs": result["onsets"], "dicrotic_notches": result["dicrotic_notches"]}

    elif method == "prominence":
        peaks, _ = find_peaks(signal, prominence=prominence)

        inverted = -signal
        troughs, _ = find_peaks(inverted, prominence=prominence)

        return {"peaks": peaks, "troughs": troughs, "dicrotic_notches": np.array([], dtype=int)}

    return {
        "peaks": np.array([], dtype=int),
        "troughs": np.array([], dtype=int),
        "dicrotic_notches": np.array([], dtype=int),
    }


def calculate_bp_metrics(signal, peaks, troughs, sampling_rate, target_fs=4.0):
    time = np.arange(len(signal)) / sampling_rate
    duration = time[-1]

    # New time grid at 4Hz
    time_4hz = np.linspace(0, duration, int(duration * target_fs))

    sbp_values = signal[peaks]
    dbp_values = signal[troughs]

    # Interpolate to 4Hz grid
    sbp_4hz = np.interp(time_4hz, time[peaks], sbp_values)
    dbp_4hz = np.interp(time_4hz, time[troughs], dbp_values)

    # Calculate MAP on the 4Hz grid
    map_4hz = dbp_4hz + (sbp_4hz - dbp_4hz) / 3

    return {
        "time_4hz": time_4hz,
        "sbp_4hz": sbp_4hz,
        "dbp_4hz": dbp_4hz,
        "map_4hz": map_4hz,
        "mean_sbp": np.mean(sbp_values),
        "mean_dbp": np.mean(dbp_values),
        "mean_mbp": np.mean(map_4hz),
    }


def process_bp(signal, sampling_rate, params):
    """
    Complete blood pressure processing pipeline

    Parameters
    ----------
    signal : array
        Raw blood pressure signal
    sampling_rate : int
        Sampling rate in Hz
    params : dict
        Processing parameters

    Returns
    -------
    dict or None
        Dictionary containing all processing results, or None if insufficient peaks
        Includes:
        - raw: Raw signal
        - filtered: Filtered signal
        - auto_peaks: Auto-detected systolic peak indices (immutable)
        - current_peaks: Current systolic peak indices (mutable for editing)
        - auto_troughs: Auto-detected diastolic trough indices (immutable)
        - current_troughs: Current diastolic trough indices (mutable for editing)
        - dicrotic_notches: Dicrotic notch indices (if detected)
        - peaks_times: Peak times in seconds
        - troughs_times: Trough times in seconds
        - sbp, dbp, mbp: Blood pressure values
        - sbp_signal, dbp_signal, mbp_signal: Continuous BP signals
        - n_peaks, n_troughs: Counts
        - mean_sbp, mean_dbp, mean_mbp: Statistics
        - calibration_artifacts: Calibration artifact regions (if detected)
        - quality_dp: Derivative quality metrics (if detected)
    """
    signal = np.asarray(signal, dtype=float)

    bp_filtered = filter_bp(
        signal,
        sampling_rate,
        method=params.get("filter_method", "bessel_25hz"),
        filter_order=params.get("filter_order", 3),
        cutoff_freq=params.get("cutoff_freq", 25),
        filter_type=params.get("filter_type", "butterworth"),
        lowcut=params.get("lowcut", 0.5),
        highcut=params.get("highcut", 15.0),
        apply_lowcut=params.get("apply_lowcut", True),
        apply_highcut=params.get("apply_highcut", True),
    )

    peak_result = detect_bp_peaks(
        bp_filtered,
        sampling_rate,
        method=params.get("peak_method", "delineator"),
        prominence=params.get("prominence", 10),
    )

    peaks = peak_result["peaks"]
    troughs = peak_result["troughs"]
    dicrotic_notches = peak_result["dicrotic_notches"]

    result = {
        "raw": signal,
        "raw_signal": signal,
        "filtered": bp_filtered,
        "clean": bp_filtered,
        "cleaned_signal": bp_filtered,
        "sampling_rate": sampling_rate,
        "params": params.copy(),
        "dicrotic_notches": dicrotic_notches,
    }

    if params.get("detect_calibration", True):
        calib_result = detect_calibration_artifacts(
            signal,
            sampling_rate,
            thr_norm=params.get("calibration_threshold", 1),
            min_dur_s=params.get("calibration_min_duration", 1.0),
            pad_s=params.get("calibration_padding", 20),
        )
        result["calibration_artifacts"] = calib_result
        result["quality_dp"] = calib_result["dp_plot"]

        peaks = filter_indices_outside_regions(peaks, calib_result["starts"], calib_result["ends"])
        troughs = filter_indices_outside_regions(troughs, calib_result["starts"], calib_result["ends"])
    else:
        result["calibration_artifacts"] = None
        result["quality_dp"] = np.array([])

    if len(peaks) < 2 or len(troughs) < 2:
        return None

    result["auto_peaks"] = peaks.copy()
    result["current_peaks"] = peaks.copy()
    result["auto_troughs"] = troughs.copy()
    result["current_troughs"] = troughs.copy()
    result["peaks_times"] = peaks / sampling_rate
    result["troughs_times"] = troughs / sampling_rate
    result["n_peaks"] = len(peaks)
    result["n_troughs"] = len(troughs)

    bp_metrics = calculate_bp_metrics(bp_filtered, peaks, troughs, sampling_rate)
    result.update(bp_metrics)

    return result
