"""
Quality and artifact detection algorithms for physiological signals
Pure functions with no classes
"""

import numpy as np
from scipy.signal import butter, filtfilt, medfilt


def _lowpass_filter(x, fs, cutoff_hz=30.0, order=2):
    """
    Apply Butterworth lowpass filter

    Parameters
    ----------
    x : array
        Input signal
    fs : float
        Sampling rate in Hz
    cutoff_hz : float
        Cutoff frequency in Hz
    order : int
        Filter order

    Returns
    -------
    array
        Filtered signal
    """
    x = np.asarray(x, dtype=float)

    nyq = 0.5 * fs
    w = cutoff_hz / nyq
    w = min(max(w, 1e-6), 0.999999)
    b, a = butter(order, w, btype="low")
    return filtfilt(b, a, x)


def detect_calibration_artifacts(signal, fs, thr_norm=0.03, min_dur_s=2.0,
                                  lp_cutoff_hz=15.0, dp_ma_s=0.05, dp_med_s=0.15,
                                  pad_s=0.10, normalize=True):
    """
    Detect calibration artifacts in blood pressure signal using derivative analysis

    Algorithm:
    1. Apply lowpass filter
    2. Calculate absolute derivative (mmHg/s)
    3. Smooth with moving average
    4. Optional median filter
    5. Robust normalization using median + MAD
    6. Detect regions where normalized derivative is below threshold (flat regions)
    7. Enforce minimum duration
    8. Add padding

    Parameters
    ----------
    signal : array
        Blood pressure signal
    fs : float
        Sampling rate in Hz
    thr_norm : float
        Threshold on normalized derivative (dimensionless, default 0.03)
    min_dur_s : float
        Minimum duration of artifact region in seconds
    lp_cutoff_hz : float
        Lowpass filter cutoff frequency
    dp_ma_s : float
        Moving average window for derivative smoothing (seconds)
    dp_med_s : float
        Median filter window for derivative (seconds, 0 to disable)
    pad_s : float
        Padding to add around detected regions (seconds)
    normalize : bool
        Apply robust normalization to derivative

    Returns
    -------
    dict
        Dictionary containing:
        - starts: Start indices of artifact regions
        - ends: End indices of artifact regions (inclusive)
        - dp: Raw derivative (mmHg/s)
        - dp_plot: Normalized or smoothed derivative for plotting
        - signal_lp: Lowpass filtered signal
        - threshold: Threshold value used
    """
    signal = np.asarray(signal, dtype=float)
    n = signal.size

    if n < 3:
        dp = np.array([], dtype=float)
        return {
            'starts': np.array([], dtype=int),
            'ends': np.array([], dtype=int),
            'dp': dp,
            'dp_plot': dp,
            'signal_lp': signal.copy(),
            'threshold': float(thr_norm)
        }

    signal_lp = _lowpass_filter(signal, fs, cutoff_hz=lp_cutoff_hz)

    dp = np.abs(np.diff(signal_lp)) * fs

    dp_ma_n = max(1, int(round(dp_ma_s * fs)))
    if dp_ma_n > 1:
        kernel = np.ones(dp_ma_n, dtype=float) / dp_ma_n
        dp_smooth = np.convolve(dp, kernel, mode="same")
    else:
        dp_smooth = dp

    if dp_med_s and dp_med_s > 0:
        dp_med_n = int(round(dp_med_s * fs))
        dp_med_n = max(1, dp_med_n | 1)
        if dp_med_n > 1:
            dp_smooth = medfilt(dp_smooth, kernel_size=dp_med_n)

    if normalize:
        med = float(np.median(dp_smooth))
        mad = float(np.median(np.abs(dp_smooth - med))) + 1e-12
        scale = med + 6.0 * mad
        scale = max(scale, 1e-6)
        dp_plot = dp_smooth / scale
        dp_plot = np.clip(dp_plot, 0.0, 1.0)
        thr_plot = float(thr_norm)
    else:
        dp_plot = dp_smooth
        thr_plot = float(thr_norm)

    plateau = dp_plot < thr_plot

    d = np.diff(plateau.astype(np.int8), prepend=0, append=0)
    starts_dp = np.where(d == 1)[0]
    ends_dp = np.where(d == -1)[0] - 1

    if starts_dp.size == 0:
        return {
            'starts': np.array([], dtype=int),
            'ends': np.array([], dtype=int),
            'dp': dp,
            'dp_plot': dp_plot,
            'signal_lp': signal_lp,
            'threshold': thr_plot
        }

    min_len = int(round(min_dur_s * fs))
    keep = (ends_dp - starts_dp + 1) >= min_len
    starts_dp = starts_dp[keep]
    ends_dp = ends_dp[keep]

    if starts_dp.size == 0:
        return {
            'starts': np.array([], dtype=int),
            'ends': np.array([], dtype=int),
            'dp': dp,
            'dp_plot': dp_plot,
            'signal_lp': signal_lp,
            'threshold': thr_plot
        }
# --- DELETE THIS OLD SECTION ---
    pad_n = int(round(pad_s * fs))
    starts_p = []
    ends_p = []
    for s_dp, e_dp in zip(starts_dp, ends_dp):
        s_p = int(s_dp)
        e_p = int(min(n - 1, e_dp + 1))
        if pad_n > 0:
            s_p = max(0, s_p - pad_n)
            e_p = min(n - 1, e_p + pad_n)
        starts_p.append(s_p)
        ends_p.append(e_p)
    # -------------------------------


    pad_n = int(round(pad_s * fs))
    mask = np.zeros(n, dtype=bool)
    for s, e in zip(starts_dp, ends_dp):
        s_pad = max(0, s - pad_n)
        e_pad = min(n - 1, e + pad_n)
        mask[s_pad : e_pad + 1] = True

    # This creates clean, non-overlapping starts and ends
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts_p = np.where(d == 1)[0]
    ends_p = np.where(d == -1)[0] - 1
    # ++++++++++++++++++++++++++++++++++++++

    return {
        'starts': np.asarray(starts_p, dtype=int),
        'ends': np.asarray(ends_p, dtype=int),
        'dp': dp,
        'dp_plot': dp_plot,
        'signal_lp': signal_lp,
        'threshold': thr_plot
    }

def detect_high_derivative_regions(dp_plot, fs, thr=0.95, min_dur_s=0.15, pad_s=0.01):
    """
    Detect regions where normalized derivative is high (noise/movement artifacts)

    Parameters
    ----------
    dp_plot : array
        Normalized derivative signal (length N-1 in derivative space)
    fs : float
        Sampling rate in Hz
    thr : float
        Threshold for high derivative (default 0.95)
    min_dur_s : float
        Minimum duration of region in seconds
    pad_s : float
        Padding around regions in seconds

    Returns
    -------
    dict
        Dictionary containing:
        - starts: Start indices in signal space
        - ends: End indices in signal space (inclusive)
    """
    dp_plot = np.asarray(dp_plot, dtype=float)
    m = dp_plot.size

    if m == 0:
        return {
            'starts': np.array([], dtype=int),
            'ends': np.array([], dtype=int)
        }

    mask = dp_plot > float(thr)

    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts_dp = np.where(d == 1)[0]
    ends_dp = np.where(d == -1)[0] - 1

    if starts_dp.size == 0:
        return {
            'starts': np.array([], dtype=int),
            'ends': np.array([], dtype=int)
        }

    min_len = int(round(min_dur_s * fs))
    keep = (ends_dp - starts_dp + 1) >= max(1, min_len)
    starts_dp = starts_dp[keep]
    ends_dp = ends_dp[keep]

    if starts_dp.size == 0:
        return {
            'starts': np.array([], dtype=int),
            'ends': np.array([], dtype=int)
        }

    pad_n = int(round(pad_s * fs))

    starts_p = []
    ends_p = []
    n_p = m + 1
    for s_dp, e_dp in zip(starts_dp, ends_dp):
        s_p = int(s_dp)
        e_p = int(min(n_p - 1, e_dp + 1))
        if pad_n > 0:
            s_p = max(0, s_p - pad_n)
            e_p = min(n_p - 1, e_p + pad_n)
        starts_p.append(s_p)
        ends_p.append(e_p)

    return {
        'starts': np.asarray(starts_p, dtype=int),
        'ends': np.asarray(ends_p, dtype=int)
    }


def filter_indices_outside_regions(indices, region_starts, region_ends):
    """
    Remove indices that fall inside any of the specified regions

    Parameters
    ----------
    indices : array
        Array of indices to filter
    region_starts : array
        Start indices of regions
    region_ends : array
        End indices of regions (inclusive)

    Returns
    -------
    array
        Filtered indices array
    """
    indices = np.asarray(indices, dtype=int)

    if indices.size == 0:
        return indices

    if region_starts is None or region_ends is None or len(region_starts) == 0:
        return indices

    region_starts = np.asarray(region_starts, dtype=int)
    region_ends = np.asarray(region_ends, dtype=int)

    keep = np.ones_like(indices, dtype=bool)
    for s, e in zip(region_starts, region_ends):
        keep &= ~((indices >= s) & (indices <= e))

    return indices[keep]
