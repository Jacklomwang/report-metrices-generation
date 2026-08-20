"""
Blood pressure delineator algorithm
Ported from MATLAB implementation
Detects systolic peaks, diastolic troughs, and dicrotic notches
Pure functional implementation with no classes
"""

import numpy as np
from scipy.signal import bessel, filtfilt


def _moving_average(x, win):
    """
    Fast moving average with same length output

    Parameters
    ----------
    x : array
        Input signal
    win : int
        Window size

    Returns
    -------
    array
        Smoothed signal
    """
    x = np.asarray(x, dtype=float)
    win = int(win)
    if win <= 1:
        return x
    win = max(3, win)
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=float) / win
    return np.convolve(x, k, mode="same")


def _smooth_like_matlab(x, fs, win_s=0.02):
    """
    MATLAB smooth() equivalent using small moving-average window

    Parameters
    ----------
    x : array
        Input signal
    fs : float
        Sampling rate in Hz
    win_s : float
        Window size in seconds (default 20ms)

    Returns
    -------
    array
        Smoothed signal
    """
    win = int(round(win_s * fs))
    win = max(5, win)
    return _moving_average(x, win)


def _seek_locales(sig, i1, i2):
    """
    Find minimum and maximum values and their indices in signal segment

    Parameters
    ----------
    sig : array
        Input signal
    i1 : int
        Start index (inclusive)
    i2 : int
        End index (inclusive)

    Returns
    -------
    tuple
        (min_val, min_idx, max_val, max_idx)
    """
    n = len(sig)
    i1 = int(max(0, min(n - 1, i1)))
    i2 = int(max(0, min(n - 1, i2)))
    if i2 < i1:
        i1, i2 = i2, i1
    seg = sig[i1 : i2 + 1]
    if seg.size == 0:
        return float(sig[i1]), i1, float(sig[i1]), i1
    jmin = int(np.argmin(seg))
    jmax = int(np.argmax(seg))
    return float(seg[jmin]), i1 + jmin, float(seg[jmax]), i1 + jmax


def _seek_dicrotic(diff_seg, fs):
    """
    Find dicrotic notch in derivative segment

    Parameters
    ----------
    diff_seg : array
        Derivative signal segment
    fs : float
        Sampling rate in Hz

    Returns
    -------
    int
        Index within diff_seg where dicrotic notch is located (0 if not found)
    """
    tempdiff = np.asarray(diff_seg, dtype=float)
    if tempdiff.size < 10:
        return 0

    tempdiff = _moving_average(tempdiff, win=max(5, int(round(0.02 * fs))))

    izc_min = 0
    tzc_min = []

    itemp = 3
    temp_len = len(tempdiff) - 3

    while itemp <= temp_len:
        if tempdiff[itemp] * tempdiff[itemp + 1] <= 0:
            if tempdiff[itemp - 2] < 0 and tempdiff[itemp + 2] >= 0:
                izc_min += 1
                tzc_min.append(itemp)
        itemp += 1

    if izc_min == 0:
        itemp = 3
        temp_min = tempdiff[itemp]
        itemp_min = itemp
        while itemp < temp_len:
            if tempdiff[itemp] < temp_min:
                temp_min = tempdiff[itemp]
                itemp_min = itemp
            itemp += 1

        itemp = itemp_min + 1
        while itemp < temp_len:
            if tempdiff[itemp + 1] <= tempdiff[itemp - 1]:
                return int(itemp)
            itemp += 1
        return 0

    if izc_min == 1:
        return int(tzc_min[0])

    itemp = tzc_min[0]
    temp_max = tempdiff[itemp]
    itemp_max = itemp

    while itemp < temp_len:
        if tempdiff[itemp] > temp_max:
            temp_max = tempdiff[itemp]
            itemp_max = itemp
        itemp += 1

    for k in range(len(tzc_min) - 1, -1, -1):
        if tzc_min[k] < itemp_max:
            return int(tzc_min[k])

    return 0


def delineate_bp(signal, fs, do_filter=True, filter_cutoff_hz=25.0, smooth_win_s=0.02):
    """
    Delineate blood pressure waveform to find fiducial points

    Algorithm uses derivative-based detection with self-learning threshold.
    Originally ported from MATLAB implementation.

    Parameters
    ----------
    signal : array
        Blood pressure signal
    fs : float
        Sampling rate in Hz
    do_filter : bool
        Apply 3rd-order Bessel lowpass filter at filter_cutoff_hz
    filter_cutoff_hz : float
        Lowpass filter cutoff frequency (default 25 Hz)
    smooth_win_s : float
        Smoothing window in seconds (default 0.02s = 20ms)

    Returns
    -------
    dict
        Dictionary containing:
        - onsets: Diastolic trough indices (beat onsets)
        - peaks: Systolic peak indices
        - dicrotic_notches: Dicrotic notch indices (0 if not found for a beat)
    """
    signal = np.asarray(signal, dtype=float)
    n = signal.size

    if n < 10 or fs <= 0:
        return {
            'onsets': np.array([], dtype=int),
            'peaks': np.array([], dtype=int),
            'dicrotic_notches': np.array([], dtype=int)
        }

    x = signal.copy()

    if do_filter:
        wn = float(filter_cutoff_hz) / (float(fs) / 2.0)
        wn = min(max(wn, 1e-6), 0.999999)
        b, a = bessel(3, wn, btype="low", analog=False, output="ba", norm="phase")
        x = filtfilt(b, a, x)

    x = 10.0 * x
    x = _smooth_like_matlab(x, fs, win_s=smooth_win_s)

    diff1 = np.empty_like(x)
    diff1[1:] = np.diff(x)
    diff1[0] = diff1[1] if n > 1 else 0.0
    diff1 = 100.0 * diff1
    diff1 = _smooth_like_matlab(diff1, fs, win_s=smooth_win_s)

    if n > 12 * fs:
        tk = 10
    elif n > 7 * fs:
        tk = 5
    elif n > 4 * fs:
        tk = 2
    else:
        tk = 1

    close_win = int(np.floor(0.1 * fs))
    step_win = int(2 * fs)

    if tk > 1:
        tatom = int(np.floor(n / (tk + 2)))
        temp_th = []
        for ji in range(1, tk + 1):
            sig_index = ji * tatom
            temp_index = sig_index + int(fs)
            temp_index = min(temp_index, n - 1)
            tmin, _, tmax, _ = _seek_locales(x, sig_index, temp_index)
            temp_th.append(tmax - tmin)
        abp_max_th = float(np.mean(temp_th)) if len(temp_th) else float(np.ptp(x))
    else:
        tmin, _, tmax, _ = _seek_locales(x, close_win, n - 1)
        abp_max_th = float(tmax - tmin)

    abp_max_lt = 0.4 * abp_max_th

    peakp = []
    onsetp = []
    dicron = []

    diff_index = max(close_win, 1)

    while diff_index < n:
        temp_min = x[diff_index]
        temp_max = x[diff_index]
        temp_index = diff_index

        tpeakp = diff_index
        tonsetp = diff_index

        while temp_index < (n - 2):
            if (temp_index - diff_index) > step_win:
                temp_index = diff_index
                abp_max_th = 0.6 * abp_max_th
                if abp_max_th <= abp_max_lt:
                    abp_max_th = 2.5 * abp_max_lt
                break

            if diff1[temp_index - 1] * diff1[temp_index + 1] <= 0:
                jj = max(0, temp_index - 5)
                jk = min(n - 1, temp_index + 5)

                if diff1[jj] < 0 and diff1[jk] > 0:
                    temp_mini, tmin, _, _ = _seek_locales(x, jj, jk)
                    if abs(tmin - temp_index) <= 2:
                        temp_min = temp_mini
                        tonsetp = tmin

                elif diff1[jj] > 0 and diff1[jk] < 0:
                    _, _, temp_maxi, tmax = _seek_locales(x, jj, jk)
                    if abs(tmax - temp_index) <= 2:
                        temp_max = temp_maxi
                        tpeakp = tmax

                amp = (temp_max - temp_min)
                if amp > 0.2 * abp_max_th and amp < 6.0 * abp_max_th and tpeakp > tonsetp:
                    ttemp_min = x[tonsetp]
                    ttonsetp = tonsetp
                    for ttk in range(tpeakp, tonsetp, -1):
                        if x[ttk] < ttemp_min:
                            ttemp_min = x[ttk]
                            ttonsetp = ttk
                    temp_min = ttemp_min
                    tonsetp = ttonsetp

                    if len(peakp) > 0:
                        if (tonsetp - peakp[-1]) < (3 * close_win):
                            temp_index = diff_index
                            abp_max_th = 2.5 * abp_max_lt
                            break

                        if (tpeakp - peakp[-1]) > step_win:
                            peakp.pop()
                            onsetp.pop()
                            if len(dicron) > 0:
                                dicron.pop()

                    peakp.append(int(tpeakp))
                    onsetp.append(int(tonsetp))

                    if len(peakp) >= 2:
                        tf = onsetp[-1] - onsetp[-2]

                        to = int(np.floor(fs / 20.0))
                        tff = int(np.floor(0.1 * tf))
                        if tff < to:
                            to = tff
                        to = peakp[-2] + to

                        te = int(np.floor(fs / 2.0))
                        tff = int(np.floor(0.5 * tf))
                        if tff < te:
                            te = tff
                        te = peakp[-2] + te

                        to = int(np.clip(to, 0, n - 1))
                        te = int(np.clip(te, to + 1, n - 1))

                        local = diff1[to:te]
                        off = _seek_dicrotic(local, fs=fs)
                        if off == 0:
                            off = int((te - peakp[-2]) / 3)
                        dicron.append(int(to + off))
                    else:
                        dicron.append(0)

                    temp_index = temp_index + close_win
                    break

            temp_index += 1

        diff_index = temp_index + 1



    # ==========================================================
    # NEW: BEAT-BOUNDED PEAK REFINEMENT 
    # Fixes the "local maxima" issue by forcing the peak to be
    # the absolute highest point between two consecutive troughs.
    # ==========================================================
    refined_peakp = []
    
    for i in range(len(peakp)):
        start_idx = onsetp[i]
        
        # Define the end of the search window (the next trough)
        if i < len(peakp) - 1:
            end_idx = onsetp[i+1]
        else:
            # For the very last beat, search ahead by an average beat duration 
            # or up to the end of the signal
            avg_beat_len = int(np.median(np.diff(onsetp))) if len(onsetp) > 1 else int(fs)
            end_idx = min(n, start_idx + avg_beat_len)

        # Extract the beat window from the smoothed signal
        beat_window = x[start_idx:end_idx]
        
        if len(beat_window) > 0:
            # Find the absolute maximum within this specific beat
            true_peak_idx = start_idx + int(np.argmax(beat_window))
            refined_peakp.append(true_peak_idx)
        else:
            # Fallback if windowing fails
            refined_peakp.append(peakp[i])
            
    # Overwrite the original delineator peaks
    peakp = refined_peakp

# ==========================================================
    # NEW: MINIMUM DISTANCE ENFORCEMENT (0.4 seconds)
    # Drops beats that occur too close to the previous valid beat.
    # ==========================================================
    # ==========================================================
    # NEW: ROBUST MINIMUM DISTANCE ENFORCEMENT (0.4 seconds)
    # Checks peak-to-peak distance. If too close, keeps the larger peak.
    # ==========================================================
    # FINAL: BEAT REFINEMENT (Amplitude + Distance + Pair Deletion)
    # ==========================================================
    if len(peakp) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    # 1. Calculate amplitudes of every detected pair (including the false ones)
    beat_amps = []
    for i in range(len(peakp)):
        beat_amps.append(x[peakp[i]] - x[onsetp[i]])
    
    # 2. Use 75th percentile to ignore the tiny false bumps when finding the "Average True Size"
    true_size_ref = np.percentile(beat_amps, 75) if len(beat_amps) > 0 else 1.0
    min_amp_thresh = 0.25 * true_size_ref  # Must be >= 25% of a normal true pulse

    min_dist_samples = int(0.35 * fs) 
    
    final_onsetp = []
    final_peakp = []
    final_dicron = []
    
    for i in range(len(peakp)):
        start_idx = onsetp[i]  # <-- This is the Trough
        d_val = dicron[i] if i < len(dicron) else 0
        
        # A. Beat-Bounded Peak Refinement
        if i < len(peakp) - 1:
            end_idx = onsetp[i+1]
        else:
            avg_beat = int(np.median(np.diff(onsetp))) if len(onsetp) > 1 else int(fs)
            end_idx = min(n, start_idx + avg_beat)

        beat_window = x[start_idx:end_idx]
        if len(beat_window) > 0:
            true_peak_idx = start_idx + int(np.argmax(beat_window))
        else:
            true_peak_idx = peakp[i]
            
        # B. AMPLITUDE CHECK (The Trash Can)
        amp = x[true_peak_idx] - x[start_idx]
        if amp < min_amp_thresh:
            # SKIP! Both the false peak AND the false trough are abandoned right here.
            continue 
            
        # C. Distance / Refractory Check
        if len(final_peakp) == 0:
            final_onsetp.append(start_idx)
            final_peakp.append(true_peak_idx)
            final_dicron.append(d_val)
        else:
            peak_dist = true_peak_idx - final_peakp[-1]
            if peak_dist < min_dist_samples:
                # AMPLITUDE CONTEST
                if x[true_peak_idx] > x[final_peakp[-1]]:
                    final_onsetp[-1] = start_idx
                    final_peakp[-1] = true_peak_idx
                    final_dicron[-1] = d_val
            else:
                final_onsetp.append(start_idx)
                final_peakp.append(true_peak_idx)
                final_dicron.append(d_val)
    # ==========================================================

    return {
        'onsets': np.asarray(final_onsetp, dtype=int),
        'peaks': np.asarray(final_peakp, dtype=int),
        'dicrotic_notches': np.asarray(final_dicron, dtype=int)
    }