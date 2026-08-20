"""
Respiration signal processing functions
Pure functions with no classes
"""

import neurokit2 as nk
import numpy as np


def clean_rsp(
    signal,
    sampling_rate,
    method="khodadad2018",
    lowcut=0.05,
    highcut=3.0,
    filter_type="butterworth",
    filter_order=5,
    apply_lowcut=True,
    apply_highcut=True,
):
    """
    Clean respiration signal using NeuroKit2 methods or custom filtering

    Parameters
    ----------
    signal : array
        Raw respiration signal
    sampling_rate : int
        Sampling rate in Hz
    method : str
        Cleaning method ('khodadad2018', 'biosppy', 'hampel', 'custom')
    lowcut : float
        High-pass cutoff frequency (Hz)
    highcut : float
        Low-pass cutoff frequency (Hz)
    filter_type : str
        Filter type for custom filtering
    filter_order : int
        Filter order
    apply_lowcut : bool
        Apply high-pass filter
    apply_highcut : bool
        Apply low-pass filter

    Returns
    -------
    array
        Cleaned respiration signal
    """
    if method == "custom":
        rsp_clean = signal.copy()
        low = lowcut if apply_lowcut else None
        high = highcut if apply_highcut else None
        if (low is not None and low > 0) or (high is not None and high < sampling_rate / 2):
            rsp_clean = nk.signal_filter(
                rsp_clean, sampling_rate=sampling_rate, lowcut=low, highcut=high, method=filter_type, order=filter_order
            )
    else:
        rsp_clean = nk.rsp_clean(signal, sampling_rate=sampling_rate, method=method)

    return rsp_clean


def detect_breath_peaks(signal, sampling_rate, amplitude_method="robust", peak_method="scipy"):
    """
    Detect inhalation peaks and exhalation troughs in respiration signal

    Parameters
    ----------
    signal : array
        Cleaned respiration signal
    sampling_rate : int
        Sampling rate in Hz
    amplitude_method : str
        Method for handling low amplitude signals:
        - 'robust': Use robust normalization (median + MAD)
        - 'standardize': Z-score normalization
        - 'minmax': Min-max normalization to [0, 1]
        - None: No normalization (original behavior)
    peak_method : str
        Peak detection method passed to nk.rsp_peaks():
        - 'scipy': scipy find_peaks (default)
        - 'khodadad2018': Khodadad et al. (2018) parameters
        - 'biosppy': BioSPPy resp() parameters

    Returns
    -------
    dict
        Dictionary containing:
        - peaks: Inhalation peak indices
        - troughs: Exhalation trough indices
    """
    # Apply amplitude normalization for better peak detection in low amplitude signals
    if amplitude_method == "robust":
        # Robust normalization using median and MAD (Median Absolute Deviation)
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        if mad > 1e-10:  # Avoid division by zero
            normalized_signal = (signal - median) / (mad * 1.4826)  # 1.4826 for consistency with std
        else:
            normalized_signal = signal - median
    elif amplitude_method == "standardize":
        # Z-score normalization
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 1e-10:
            normalized_signal = (signal - mean) / std
        else:
            normalized_signal = signal - mean
    elif amplitude_method == "minmax":
        # Min-max normalization to [0, 1]
        min_val = np.min(signal)
        max_val = np.max(signal)
        if max_val - min_val > 1e-10:
            normalized_signal = (signal - min_val) / (max_val - min_val)
        else:
            normalized_signal = signal
    else:
        # No normalization
        normalized_signal = signal

    _, peaks_info = nk.rsp_peaks(normalized_signal, sampling_rate=sampling_rate, method=peak_method)

    peaks = peaks_info["RSP_Peaks"]
    troughs = peaks_info["RSP_Troughs"]

    return {"peaks": peaks, "troughs": troughs}


def calculate_breathing_rate(troughs, sampling_rate, signal_length, rate_method="monotone_cubic"):
    """
    Calculate breathing rate from exhalation troughs

    Parameters
    ----------
    troughs : array
        Trough sample indices
    sampling_rate : int
        Sampling rate in Hz
    signal_length : int
        Length of signal for interpolation
    rate_method : str
        Interpolation method

    Returns
    -------
    dict
        Dictionary containing:
        - br_bpm: Breath-to-breath rate in breaths per minute
        - br_interpolated: Interpolated breathing rate signal
        - mean_br: Mean breathing rate
        - std_br: Standard deviation of breathing rate
    """
    breath_intervals = np.diff(troughs) / sampling_rate
    br_bpm = 60 / breath_intervals

    br_interpolated = nk.signal_rate(
        troughs, sampling_rate=sampling_rate, desired_length=signal_length, interpolation_method=rate_method
    )

    return {
        "br_bpm": br_bpm,
        "br_interpolated": br_interpolated,
        "mean_br": np.nanmean(br_bpm),
        "std_br": np.nanstd(br_bpm),
    }


def compute_rvt(rsp_signal, sampling_rate, method="power2020"):
    """
    Compute Respiratory Volume per Time (RVT) using NeuroKit2.

    Parameters
    ----------
    rsp_signal : array
        Cleaned respiration signal
    sampling_rate : int
        Sampling rate in Hz
    method : str
        RVT method ('power2020', 'harrison2021', 'birn2006')

    Returns
    -------
    array or None
        RVT time series, or None on failure
    """
    try:
        rvt = nk.rsp_rvt(rsp_signal, sampling_rate=sampling_rate, method=method)
        if isinstance(rvt, np.ndarray):
            return rvt
        # nk.rsp_rvt may return a DataFrame/Series
        return np.asarray(rvt).ravel()
    except Exception:
        return None


def compute_rsp_phase(rsp_clean, peaks, troughs):
    """Compute respiratory phase, phase completion, and full cycle completion.

    Uses nk.rsp_phase() for binary phase and within-phase completion,
    then builds full cycle completion (0->1) preserving natural I:E ratio.

    Parameters
    ----------
    rsp_clean : array
        Cleaned respiration signal
    peaks : array
        Inhalation peak indices
    troughs : array
        Exhalation trough indices

    Returns
    -------
    dict or None
        Dictionary with rsp_phase, rsp_phase_completion, rsp_cycle_completion
    """
    n = len(rsp_clean)
    rsp_phase = None
    rsp_phase_completion = None

    # Try NeuroKit2 first
    try:
        phase_df = nk.rsp_phase(
            rsp_clean,
            peaks={"RSP_Peaks": peaks, "RSP_Troughs": troughs},
        )
        rsp_phase = phase_df["RSP_Phase"].values
        rsp_phase_completion = phase_df["RSP_Phase_Completion"].values
    except Exception:
        pass

    # Manual fallback: compute phase directly from peaks and troughs
    if rsp_phase is None:
        rsp_phase = np.zeros(n, dtype=int)
        rsp_phase_completion = np.zeros(n, dtype=float)

        sorted_peaks = np.sort(peaks)
        sorted_troughs = np.sort(troughs)

        events = np.concatenate([sorted_troughs, sorted_peaks])
        # 0 = trough (inhalation starts), 1 = peak (exhalation starts)
        labels = np.concatenate([
            np.zeros(len(sorted_troughs), dtype=int),
            np.ones(len(sorted_peaks), dtype=int),
        ])
        order = np.argsort(events)
        events = events[order]
        labels = labels[order]

        for idx in range(len(events)):
            start = int(events[idx])
            end = int(events[idx + 1]) if idx + 1 < len(events) else n
            segment_len = end - start

            if labels[idx] == 0:  # trough -> inhalation
                rsp_phase[start:end] = 1
            else:  # peak -> exhalation
                rsp_phase[start:end] = 0

            if segment_len > 0:
                rsp_phase_completion[start:end] = np.linspace(
                    0, 1, segment_len, endpoint=False
                )

    cycle_completion = np.full(n, np.nan)

    if len(troughs) < 2:
        return {
            "rsp_phase": rsp_phase,
            "rsp_phase_completion": rsp_phase_completion,
            "rsp_cycle_completion": cycle_completion,
        }

    for i in range(len(troughs) - 1):
        t_start = troughs[i]
        t_end = troughs[i + 1]
        cycle_len = t_end - t_start
        if cycle_len <= 0:
            continue

        # Find inhalation duration within this cycle
        # Inhalation = phase==1 in the range [t_start, t_end)
        cycle_phase = rsp_phase[t_start:t_end]
        inh_samples = np.sum(cycle_phase == 1)
        inh_proportion = inh_samples / cycle_len

        # Map inhalation completion to [0, inh_proportion]
        # Map exhalation completion to [inh_proportion, 1]
        for j in range(t_start, t_end):
            if rsp_phase[j] == 1:  # inhalation
                cycle_completion[j] = rsp_phase_completion[j] * inh_proportion
            else:  # exhalation
                cycle_completion[j] = inh_proportion + rsp_phase_completion[j] * (1.0 - inh_proportion)

    # Handle last incomplete cycle (after last trough)
    if troughs[-1] < n - 1:
        last_start = troughs[-1]
        remaining = n - last_start
        if remaining > 1:
            # Use average cycle duration for proportion estimate
            avg_cycle = np.mean(np.diff(troughs))
            if avg_cycle > 0:
                for j in range(last_start, n):
                    if rsp_phase[j] == 1:
                        inh_samples_last = np.sum(rsp_phase[last_start:n] == 1)
                        est_proportion = inh_samples_last / avg_cycle if avg_cycle > 0 else 0.5
                        est_proportion = min(est_proportion, 1.0)
                        cycle_completion[j] = rsp_phase_completion[j] * est_proportion
                    else:
                        cycle_completion[j] = rsp_phase_completion[j]

    return {
        "rsp_phase": rsp_phase,
        "rsp_phase_completion": rsp_phase_completion,
        "rsp_cycle_completion": cycle_completion,
    }


def process_rsp(signal, sampling_rate, params):
    """
    Complete respiration processing pipeline

    Parameters
    ----------
    signal : array
        Raw respiration signal
    sampling_rate : int
        Sampling rate in Hz
    params : dict
        Processing parameters

    Returns
    -------
    dict or None
        Dictionary containing all processing results, or None if insufficient breaths
        Includes:
        - raw: Raw signal
        - clean: Cleaned signal
        - auto_peaks: Auto-detected inhalation peak indices (immutable)
        - current_peaks: Current inhalation peak indices (mutable for editing)
        - auto_troughs: Auto-detected exhalation trough indices (immutable)
        - current_troughs: Current exhalation trough indices (mutable for editing)
        - peaks_times: Peak times in seconds
        - troughs_times: Trough times in seconds
        - br_bpm: Breath-to-breath breathing rate
        - br_interpolated: Continuous breathing rate signal
        - n_breaths: Number of breaths
        - mean_br, std_br: Statistics
    """
    signal = np.asarray(signal, dtype=float)

    cleaning_method = params.get("cleaning_method", params.get("method", "khodadad2018"))

    rsp_clean = clean_rsp(
        signal,
        sampling_rate,
        method=cleaning_method,
        lowcut=params.get("lowcut", 0.05),
        highcut=params.get("highcut", 3.0),
        filter_type=params.get("filter_type", "butterworth"),
        filter_order=params.get("filter_order", 5),
        apply_lowcut=params.get("apply_lowcut", True),
        apply_highcut=params.get("apply_highcut", True),
    )

    breath_peaks = detect_breath_peaks(
        rsp_clean,
        sampling_rate,
        amplitude_method=params.get("amplitude_method", "robust"),
        peak_method=params.get("peak_method", "scipy"),
    )
    peaks = breath_peaks["peaks"]
    troughs = breath_peaks["troughs"]

    if len(troughs) < 2:
        return None

    br_result = calculate_breathing_rate(
        troughs, sampling_rate, len(rsp_clean), rate_method=params.get("rate_method", "monotone_cubic")
    )

    result = {
        "raw": signal,
        "clean": rsp_clean,
        "raw_signal": signal,
        "cleaned_signal": rsp_clean,
        "auto_peaks": peaks.copy(),
        "current_peaks": peaks.copy(),
        "auto_troughs": troughs.copy(),
        "current_troughs": troughs.copy(),
        "peaks_times": peaks / sampling_rate,
        "troughs_times": troughs / sampling_rate,
        "sampling_rate": sampling_rate,
        "params": params.copy(),
        "n_breaths": len(troughs),
    }

    result.update(br_result)

    rvt_method = params.get("rvt_method", "none")
    if rvt_method and rvt_method != "none":
        rvt = compute_rvt(rsp_clean, sampling_rate, method=rvt_method)
        if rvt is not None:
            result["rvt"] = rvt

    phase_result = compute_rsp_phase(rsp_clean, peaks, troughs)
    if phase_result is not None:
        result.update(phase_result)

    return result
