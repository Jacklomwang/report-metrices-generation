"""
Doppler signal processing functions
Pure functions with no classes

This module is based on the blood pressure pipeline, but:
- Removes calibration-artifact detection and related filtering
- Renames BP-specific concepts/outputs to Doppler-friendly names
- Keeps the same peak/trough editing structure (auto_* vs current_*)
"""

import numpy as np
from scipy.signal import bessel, butter, filtfilt, find_peaks
import neurokit2 as nk

from ..algorithms.bp_delineator import delineate_bp  # reuse BP delineator for Doppler


import numpy as np
from scipy.signal import bessel, butter, filtfilt, find_peaks, savgol_filter
import scipy.ndimage
import scipy.interpolate
import neurokit2 as nk

try:
    import pywt
except ImportError:
    pywt = None

from ..algorithms.bp_delineator import delineate_bp

# ============================================================
# Filtering
# ============================================================

def get_robust_trend(data, fs, trend_win_s=2.0):
    sigma = (trend_win_s * fs) / 2.0
    return scipy.ndimage.gaussian_filter(data, sigma=sigma)

def wavelet_denoise_iqr(data, wavelet="db6", level=10, iqr_alpha=4.0, drop_levels=1):
    if pywt is None:
        return data # Fallback if not installed
        
    data = np.asarray(data).ravel()
    n = len(data)
    max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    
    level = min(level, max_level) if level > 0 else min(10, max_level)
    coeffs = pywt.wavedec(data, wavelet, level=level, mode='symmetric')
    coeffs_cleaned = []

    for i, c in enumerate(coeffs):
        current_detail = level - (i - 1) if i > 0 else 0
        
        if i == 0: # Approx
            coeffs_cleaned.append(c)
            continue
        if current_detail <= drop_levels: # Drop fine details
            coeffs_cleaned.append(np.zeros_like(c))
            continue

        # IQR cleaning
        q1, q3 = np.quantile(c, [0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - iqr_alpha*iqr, q3 + iqr_alpha*iqr
        
        c_fixed = c.copy()
        c_fixed[(c < lo) | (c > hi)] = 0.0
        coeffs_cleaned.append(c_fixed)

    return pywt.waverec(coeffs_cleaned, wavelet, mode='symmetric')[:n]

def extract_template_and_score(sig, troughs, target_len=100):
    """Shift-invariant correlation scoring."""
    beats_norm = []
    
    for i in range(len(troughs) - 1):
        start, end = troughs[i], troughs[i+1]
        length = end - start
        if length < 10 or length > 5000: continue
            
        segment = sig[start:end]
        
        # Resample
        x_old = np.linspace(0, 1, num=length)
        x_new = np.linspace(0, 1, num=target_len)
        f = scipy.interpolate.interp1d(x_old, segment, kind='linear')
        seg_resampled = f(x_new)
        
        # Z-Score (Automatically handles DC offset/baseline drift)
        std_val = np.std(seg_resampled)
        if std_val > 1e-9:
            seg_norm = (seg_resampled - np.mean(seg_resampled)) / std_val
        else:
            seg_norm = seg_resampled - np.mean(seg_resampled)
        
        beats_norm.append(seg_norm)

    if not beats_norm: return [], []
    
    beats_matrix = np.array(beats_norm) 
    template = np.mean(beats_matrix, axis=0)
    
    scores = []
    max_shift = 5 
    for beat in beats_matrix:
        best_corr = -1.0
        for shift in range(-max_shift, max_shift + 1):
            shifted_beat = np.roll(beat, shift)
            corr = np.corrcoef(template, shifted_beat)[0, 1]
            if corr > best_corr: best_corr = corr
        scores.append(max(0, best_corr))
        
    return template, scores


# ============================================================
# Filtering & Pipeline Updates
# ============================================================


def filter_doppler(signal, sampling_rate, method='sg_wavelet', filter_order=3,
                   cutoff_freq=25, filter_type='butterworth', lowcut=0.5, highcut=15.0,
                   apply_lowcut=True, apply_highcut=True,
                   sg_win=0.1, wavelet='db6', level=10, alpha=4.0, drop_levels=1, trend_win=2.0, **kwargs):
    """
    Filter Doppler signal
    """

    if method == 'sg_wavelet':
        # Use the parameters directly from the function signature and enforce types
        sg_win = float(sg_win)
        level = int(level)
        alpha = float(alpha)
        drop_levels = int(drop_levels)
        trend_win = float(trend_win)

        # 1. Detrend
        trend = get_robust_trend(signal, sampling_rate, trend_win)
        detrended = signal - trend
        
        # 2. Savitzky-Golay
        w_len = int(sg_win * sampling_rate) | 1 # ensure odd
        w_len = max(3, w_len)
        sg = savgol_filter(detrended, window_length=w_len, polyorder=3)
        
        # 3. Wavelet
        clean_ac = wavelet_denoise_iqr(sg, wavelet, level, alpha, drop_levels)
        
        # 4. Retrend
        doppler_filtered = clean_ac + trend

    elif method == 'bessel_25hz':
        wn = float(cutoff_freq) / (float(sampling_rate) / 2.0)
        wn = min(max(wn, 1e-6), 0.999999)
        b, a = bessel(filter_order, wn, btype="low", analog=False, output="ba", norm="phase")
        doppler_filtered = filtfilt(b, a, signal)

    elif method == 'butterworth':
        wn = float(cutoff_freq) / (float(sampling_rate) / 2.0)
        wn = min(max(wn, 1e-6), 0.999999)
        b, a = butter(filter_order, wn, btype="low", analog=False)
        doppler_filtered = filtfilt(b, a, signal)

    elif method == 'custom':
        doppler_filtered = signal.copy()
        low = lowcut if apply_lowcut else None
        high = highcut if apply_highcut else None
        if (low is not None and low > 0) or (high is not None and high < sampling_rate / 2):
            doppler_filtered = nk.signal_filter(
                doppler_filtered,
                sampling_rate=sampling_rate,
                lowcut=low,
                highcut=high,
                method=filter_type,
                order=filter_order
            )
    else:
        doppler_filtered = signal.copy()

    return doppler_filtered

# ============================================================
# Peaks / troughs
# ============================================================

def detect_doppler_peaks(signal, sampling_rate, method='delineator', prominence=10):
    """
    Detect peaks and troughs in Doppler signal.
    Reuses BP delineation logic as requested.

    Parameters
    ----------
    signal : array
        Filtered Doppler signal
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
        - peaks: Peak indices
        - troughs: Trough indices
        - dicrotic_notches: (kept for compatibility; empty if not applicable)
    """
    if method == 'delineator':
        result = delineate_bp(signal, sampling_rate)
        return {
            'peaks': result.get('peaks', np.array([], dtype=int)),
            'troughs': result.get('onsets', np.array([], dtype=int)),
            # Doppler may not have "dicrotic notch", but we keep key for compatibility
            'dicrotic_notches': result.get('dicrotic_notches', np.array([], dtype=int))
        }

    elif method == 'prominence':
        peaks, _ = find_peaks(signal, prominence=prominence)
        troughs, _ = find_peaks(-signal, prominence=prominence)
        return {
            'peaks': peaks,
            'troughs': troughs,
            'dicrotic_notches': np.array([], dtype=int)
        }

    return {
        'peaks': np.array([], dtype=int),
        'troughs': np.array([], dtype=int),
        'dicrotic_notches': np.array([], dtype=int)
    }


# ============================================================
# 4 Hz aligned metrics (Doppler-friendly names)
# ============================================================

import numpy as np

def calculate_doppler_metrics(signal, peaks, troughs, sampling_rate, target_fs=4.0):
    """
    Create 4 Hz-aligned Doppler metrics by interpolating peak/trough values.

    IMPORTANT:
    - Returns BOTH Doppler-native keys (peak_4hz/trough_4hz/amp_4hz) AND
      BP-compatible alias keys (sbp_4hz/map_4hz/dbp_4hz) so you can reuse
      create_rsp_bp_plot() without changing its BP plotting code.

    Parameters
    ----------
    signal : array-like
        Filtered Doppler waveform
    peaks : array-like (int)
        Indices of systolic-like maxima
    troughs : array-like (int)
        Indices of diastolic-like minima
    sampling_rate : float
        Sampling rate in Hz
    target_fs : float
        Target output sampling rate (default 4 Hz)

    Returns
    -------
    dict with:
      time_4hz
      peak_4hz, trough_4hz, amp_4hz
      mean_peak, mean_trough, mean_amp
      sbp_4hz, dbp_4hz, map_4hz           (BP-compatible curves)
      mean_sbp, mean_dbp, mean_mbp        (BP-compatible summary stats)
    """
    signal = np.asarray(signal) if signal is not None else np.asarray([])
    peaks = np.asarray(peaks, dtype=int) if peaks is not None else np.asarray([], dtype=int)
    troughs = np.asarray(troughs, dtype=int) if troughs is not None else np.asarray([], dtype=int)

    # Handle empty signal
    if signal.size == 0 or sampling_rate is None or sampling_rate <= 0:
        empty = np.array([])
        return {
            'time_4hz': empty,
            'peak_4hz': empty,
            'trough_4hz': empty,
            'amp_4hz': empty,
            'mean_peak': np.nan,
            'mean_trough': np.nan,
            'mean_amp': np.nan,
            # BP-compatible aliases
            'sbp_4hz': empty,
            'dbp_4hz': empty,
            'map_4hz': empty,
            'mean_sbp': np.nan,
            'mean_dbp': np.nan,
            'mean_mbp': np.nan,
        }

    # Time base
    time = np.arange(signal.size) / float(sampling_rate)
    duration = float(time[-1])

    if not np.isfinite(duration) or duration <= 0:
        nan1 = np.array([np.nan])
        t0 = np.array([0.0])
        return {
            'time_4hz': t0,
            'peak_4hz': nan1,
            'trough_4hz': nan1,
            'amp_4hz': nan1,
            'mean_peak': np.nan,
            'mean_trough': np.nan,
            'mean_amp': np.nan,
            'sbp_4hz': nan1,
            'dbp_4hz': nan1,
            'map_4hz': nan1,
            'mean_sbp': np.nan,
            'mean_dbp': np.nan,
            'mean_mbp': np.nan,
        }

    # Match BP behavior: linspace(0, duration, int(duration*target_fs))
    n_4hz = int(duration * float(target_fs))
    n_4hz = max(n_4hz, 1)
    time_4hz = np.linspace(0.0, duration, n_4hz)

    # Guard: keep only valid indices
    peaks = peaks[(peaks >= 0) & (peaks < signal.size)]
    troughs = troughs[(troughs >= 0) & (troughs < signal.size)]

    peak_values = signal[peaks] if peaks.size else np.asarray([])
    trough_values = signal[troughs] if troughs.size else np.asarray([])

    # Interpolate onto 4 Hz grid (NaN-filled if insufficient points)
    if peaks.size >= 2:
        peak_4hz = np.interp(time_4hz, time[peaks], peak_values)
    elif peaks.size == 1:
        peak_4hz = np.full_like(time_4hz, float(peak_values[0]), dtype=float)
    else:
        peak_4hz = np.full_like(time_4hz, np.nan, dtype=float)

    if troughs.size >= 2:
        trough_4hz = np.interp(time_4hz, time[troughs], trough_values)
    elif troughs.size == 1:
        trough_4hz = np.full_like(time_4hz, float(trough_values[0]), dtype=float)
    else:
        trough_4hz = np.full_like(time_4hz, np.nan, dtype=float)

    amp_4hz = peak_4hz - trough_4hz

    # A "mean-like" curve for the middle trace (BP uses DBP + (SBP-DBP)/3)
    map_4hz = trough_4hz + (peak_4hz - trough_4hz) / 3.0

    mean_peak = float(np.nanmean(peak_values)) if peak_values.size else np.nan
    mean_trough = float(np.nanmean(trough_values)) if trough_values.size else np.nan
    mean_amp = float(np.nanmean(amp_4hz)) if amp_4hz.size else np.nan
    mean_map = float(np.nanmean(map_4hz)) if map_4hz.size else np.nan

    return {
        # Native doppler metrics
        'time_4hz': time_4hz,
        'peak_4hz': peak_4hz,
        'trough_4hz': trough_4hz,
        'amp_4hz': amp_4hz,
        'mean_peak': mean_peak,
        'mean_trough': mean_trough,
        'mean_amp': mean_amp,

        # BP-compatible aliases for plotting reuse
        'sbp_4hz': peak_4hz,
        'dbp_4hz': trough_4hz,
        'map_4hz': map_4hz,
        'mean_sbp': mean_peak,
        'mean_dbp': mean_trough,
        'mean_mbp': mean_map,
    }



# ============================================================
# Full pipeline (no calibration parts)
# ============================================================

def process_doppler(signal, sampling_rate, params):
    """
    Complete Doppler processing pipeline WITH Beat Quality Scoring.
    """
    # 1. Filter Signal (Now explicitly unpacking all new Wavelet parameters)
    doppler_filtered = filter_doppler(
        signal, 
        sampling_rate, 
        method=params.get('filter_method', 'sg_wavelet'),
        # Standard filter params
        filter_order=params.get('filter_order', 3),
        cutoff_freq=params.get('cutoff_freq', 25),
        filter_type=params.get('filter_type', 'butterworth'),
        lowcut=params.get('lowcut', 0.5),
        highcut=params.get('highcut', 15.0),
        apply_lowcut=params.get('apply_lowcut', True),
        apply_highcut=params.get('apply_highcut', True),
        # NEW SG+Wavelet params mapped directly from Streamlit UI
        sg_win=params.get('sg_win', 0.1),
        wavelet=params.get('wavelet', 'db6'),
        level=params.get('level', 10),
        alpha=params.get('alpha', 4.0),
        drop_levels=params.get('drop_levels', 1),
        trend_win=params.get('trend_win', 2.0)
    )

    # 2. Detect Peaks (Forcing Delineator as requested)
    peak_result = detect_doppler_peaks(
        doppler_filtered, 
        sampling_rate, 
        method='delineator' 
    )

    peaks = peak_result['peaks']
    troughs = peak_result['troughs']

    if len(peaks) < 2 or len(troughs) < 2:
        return None

    # 3. Calculate Beat Quality Scores (Shift-Invariant Correlation)
    template, scores = extract_template_and_score(doppler_filtered, troughs)
    mean_quality = float(np.mean(scores)) if len(scores) > 0 else np.nan

    # 4. Pack Result
    result = {
        'raw': signal,
        'filtered': doppler_filtered,
        'dicrotic_notches': peak_result['dicrotic_notches'],

        'auto_peaks': peaks.copy(),
        'current_peaks': peaks.copy(),
        'auto_troughs': troughs.copy(),
        'current_troughs': troughs.copy(),

        'peaks_times': peaks / sampling_rate,
        'troughs_times': troughs / sampling_rate,
        'n_peaks': int(len(peaks)),
        'n_troughs': int(len(troughs)),
        
        # Store quality metrics
        'beat_scores': scores, 
        'mean_quality': mean_quality
    }

    # Add the 4Hz aligned metrics
    doppler_metrics = calculate_doppler_metrics(doppler_filtered, peaks, troughs, sampling_rate)
    result.update(doppler_metrics)

    return result
