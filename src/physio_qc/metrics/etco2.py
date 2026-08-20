"""
End-Tidal CO2 (ETCO2) Processing Module

This module provides functions for extracting end-tidal CO2 traces from continuous
CO2 recordings. It detects breath-by-breath peaks and creates an upper envelope
representing maximum CO2 levels across the respiratory cycle.

Functions:
    detect_peaks_diff: Derivative-based peak detection with curvature filtering
    detect_peaks_prominence: Prominence-based peak detection using scipy
    extract_etco2_envelope: Main processing function to create ETCO2 trace
"""

import warnings
from typing import Dict, Optional

import numpy as np
from scipy.signal import find_peaks, medfilt, peak_prominences, savgol_filter


def _nearest_odd(n: int) -> int:
    """Ensure window size is odd for Savitzky-Golay filter."""
    return n if n % 2 == 1 else n + 1


def detect_peaks_diff(
    signal: np.ndarray,
    sampling_rate: float,
    min_peak_distance_s: float = 2.0,
    min_prominence: float = 1.0,
    sg_window_s: float = 0.3,
    sg_poly: int = 2,
    prom_adapt: bool = False,
) -> np.ndarray:
    """
    Detect CO2 peaks using derivative zero-crossings with curvature filtering.

    This method:
    1. Smooths signal with Savitzky-Golay filter for differentiability
    2. Finds zero-crossings where derivative goes from positive to negative
    3. Filters for negative curvature (concave down at peaks)
    4. Refines to nearest local maximum in raw signal (±150ms window)
    5. Enforces minimum temporal separation via greedy amplitude ordering
    6. Validates by prominence with optional adaptive thresholding

    Parameters
    ----------
    signal : np.ndarray
        Raw CO2 signal in mmHg
    sampling_rate : float
        Sampling frequency in Hz
    min_peak_distance_s : float, optional
        Minimum time between peaks in seconds (default: 2.0)
    min_prominence : float, optional
        Minimum peak prominence in mmHg (default: 1.0)
    sg_window_s : float, optional
        Savitzky-Golay window duration in seconds (default: 0.3)
    sg_poly : int, optional
        Savitzky-Golay polynomial order (default: 2)
    prom_adapt : bool, optional
        Enable adaptive prominence threshold using 25th percentile (default: False)

    Returns
    -------
    peaks_idx : np.ndarray
        Array of peak indices in the signal
    """
    # Smooth with Savitzky-Golay
    win_pts = max(5, int(round(sg_window_s * sampling_rate)))
    win_pts = _nearest_odd(win_pts)

    try:
        x_s = savgol_filter(signal, window_length=win_pts, polyorder=max(1, sg_poly))
    except ValueError:
        # Fallback for very short signals
        x_s = signal.copy()

    # Compute first and second derivatives
    d1 = np.gradient(x_s)
    d2 = np.gradient(d1)

    # Zero-crossings from positive to non-positive (peak candidates)
    zc = np.where((d1[:-1] > 0) & (d1[1:] <= 0))[0] + 1
    if zc.size == 0:
        return zc

    # Enforce negative curvature (concave down)
    cand = zc[d2[zc] < 0]
    if cand.size == 0:
        cand = zc

    # Refine to nearest local maximum in raw signal within ±150ms
    r = max(1, int(round(0.15 * sampling_rate)))
    refined = []
    n = len(signal)

    for idx in cand:
        lo = max(0, idx - r)
        hi = min(n, idx + r + 1)
        if hi - lo <= 0:
            continue
        local = signal[lo:hi]
        ridx = lo + int(np.argmax(local))
        refined.append(ridx)

    if not refined:
        return np.array([], dtype=int)

    refined = np.array(refined, dtype=int)

    # Enforce minimum distance using greedy selection by descending amplitude
    min_dist = int(round(min_peak_distance_s * sampling_rate))
    order = np.argsort(signal[refined])[::-1]  # Highest peaks first
    selected = []
    taken = np.zeros(n, dtype=bool)

    for i in order:
        idx = refined[i]
        lo = max(0, idx - min_dist)
        hi = min(n, idx + min_dist + 1)
        if not taken[lo:hi].any():
            selected.append(idx)
            taken[lo:hi] = True

    peaks_idx = np.array(sorted(selected), dtype=int)

    if peaks_idx.size == 0:
        return peaks_idx

    # Validate by prominence with optional adaptive threshold
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="some peaks have a prominence of 0", category=Warning)
            prom, _, _ = peak_prominences(signal, peaks_idx)
        thr = float(min_prominence)

        if prom_adapt and prom.size:
            # Use 25th percentile as adaptive floor
            thr = max(thr, float(np.quantile(prom, 0.25)))

        keep = prom >= thr
        peaks_idx = peaks_idx[keep]
    except Exception:
        pass

    return peaks_idx


def detect_peaks_prominence(
    signal: np.ndarray, sampling_rate: float, min_peak_distance_s: float = 2.0, min_prominence: float = 1.0
) -> np.ndarray:
    """
    Detect CO2 peaks using scipy's prominence-based peak detection.

    Simpler alternative to derivative method, relies on peak prominence
    and minimum temporal separation.

    Parameters
    ----------
    signal : np.ndarray
        Raw CO2 signal in mmHg
    sampling_rate : float
        Sampling frequency in Hz
    min_peak_distance_s : float, optional
        Minimum time between peaks in seconds (default: 2.0)
    min_prominence : float, optional
        Minimum peak prominence in mmHg (default: 1.0)

    Returns
    -------
    peaks_idx : np.ndarray
        Array of peak indices in the signal
    """
    min_dist_samples = int(min_peak_distance_s * sampling_rate)
    peaks_idx, _ = find_peaks(signal, prominence=min_prominence, distance=min_dist_samples)
    return peaks_idx


def extract_etco2_envelope(signal: np.ndarray, sampling_rate: float, params: Optional[Dict] = None) -> Dict:
    """
    Extract end-tidal CO2 envelope from continuous CO2 recording.

    Detects breath-by-breath CO2 peaks, applies median smoothing, and
    interpolates to create a continuous upper envelope trace representing
    maximum CO2 levels.

    Parameters
    ----------
    signal : np.ndarray
        Raw CO2 signal in mmHg
    sampling_rate : float
        Sampling frequency in Hz
    params : dict, optional
        Processing parameters:
        - peak_method: 'diff' or 'prominence' (default: 'diff')
        - min_peak_distance_s: Minimum time between peaks in seconds (default: 2.0)
        - min_prominence: Minimum peak prominence in mmHg (default: 1.0)
        - sg_window_s: Savitzky-Golay window in seconds (default: 0.3)
        - sg_poly: Savitzky-Golay polynomial order (default: 2)
        - prom_adapt: Enable adaptive prominence threshold (default: False)
        - smooth_peaks: Median filter kernel size in peaks (default: 5)

    Returns
    -------
    result : dict
        Dictionary containing:
        - raw_signal: Original CO2 signal
        - auto_peaks: Automatically detected peak indices
        - current_peaks: Current peaks (same as auto_peaks, can be edited)
        - etco2_envelope: Upper envelope trace (same length as signal)
        - peak_values: CO2 values at peak locations
        - smoothed_peak_values: Median-filtered peak values
        - time_vector: Time axis in seconds
        - sampling_rate: Sampling frequency
        - params: Processing parameters used
    """
    if params is None:
        params = {}

    # Default parameters
    peak_method = params.get("peak_method", "diff")
    min_peak_distance_s = params.get("min_peak_distance_s", 2.0)
    min_prominence = params.get("min_prominence", 1.0)
    sg_window_s = params.get("sg_window_s", 0.3)
    sg_poly = params.get("sg_poly", 2)
    prom_adapt = params.get("prom_adapt", False)
    smooth_peaks = params.get("smooth_peaks", 5)

    # Ensure smooth_peaks is odd
    if smooth_peaks % 2 == 0:
        smooth_peaks += 1

    # Detect peaks
    if peak_method == "diff":
        peaks_idx = detect_peaks_diff(
            signal,
            sampling_rate,
            min_peak_distance_s=min_peak_distance_s,
            min_prominence=min_prominence,
            sg_window_s=sg_window_s,
            sg_poly=sg_poly,
            prom_adapt=prom_adapt,
        )

        # Fallback to prominence if derivative method fails
        if peaks_idx.size == 0:
            peaks_idx = detect_peaks_prominence(
                signal, sampling_rate, min_peak_distance_s=min_peak_distance_s, min_prominence=min_prominence
            )
    else:
        peaks_idx = detect_peaks_prominence(
            signal, sampling_rate, min_peak_distance_s=min_peak_distance_s, min_prominence=min_prominence
        )

    # Create time vector
    time_vector = np.arange(len(signal)) / sampling_rate

    # Extract peak values and apply median smoothing
    if peaks_idx.size >= smooth_peaks:
        peak_values = signal[peaks_idx]
        smoothed_peak_values = medfilt(peak_values, kernel_size=smooth_peaks)
    elif peaks_idx.size > 0:
        peak_values = signal[peaks_idx]
        smoothed_peak_values = peak_values.copy()
    else:
        peak_values = np.array([])
        smoothed_peak_values = np.array([])

    # Compute SG-filtered signal for visualization (before/after comparison)
    win_pts = max(5, int(round(sg_window_s * sampling_rate)))
    win_pts = _nearest_odd(win_pts)
    try:
        sg_filtered_signal = savgol_filter(signal, window_length=win_pts, polyorder=max(1, sg_poly))
    except ValueError:
        sg_filtered_signal = signal.copy()

    # Create envelope by interpolation
    if peaks_idx.size >= 2:
        envelope = np.interp(time_vector, time_vector[peaks_idx], smoothed_peak_values)
    elif peaks_idx.size == 1:
        # Single peak: flat line at peak value
        envelope = np.full_like(time_vector, smoothed_peak_values[0], dtype=float)
    else:
        # No peaks: fallback to smoothed signal
        try:
            win_pts = _nearest_odd(max(5, int(round(sg_window_s * sampling_rate))))
            envelope = savgol_filter(signal, window_length=win_pts, polyorder=max(1, sg_poly))
        except Exception:
            envelope = signal.astype(float)

    result = {
        "raw_signal": signal,
        "sg_filtered_signal": sg_filtered_signal,
        "auto_peaks": peaks_idx.copy(),
        "current_peaks": peaks_idx.copy(),
        "etco2_envelope": envelope,
        "peak_values": peak_values,
        "smoothed_peak_values": smoothed_peak_values,
        "time_vector": time_vector,
        "sampling_rate": sampling_rate,
        "params": params.copy(),
    }
    # Backward/forward compatibility aliases for cross-metric consistency
    result["raw"] = result["raw_signal"]
    result["clean"] = result["etco2_envelope"]
    result["cleaned_signal"] = result["etco2_envelope"]
    return result


def process_etco2(signal: np.ndarray, sampling_rate: float, params: Optional[Dict] = None) -> Dict:
    """Wrapper to standardize per-metric entry-point naming."""
    return extract_etco2_envelope(signal=signal, sampling_rate=sampling_rate, params=params)
