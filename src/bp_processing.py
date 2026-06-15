
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def lowpass_filter_bp(sig, fs, cutoff=40.0, order=4):
    nyq = fs * 0.5
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, sig)

def detect_calibration_artifacts(bp, fs, thr=0.1, min_duration_sec=0.65, debug=False):
    bp_filt = lowpass_filter_bp(bp, fs)
    dp = np.abs(np.diff(bp_filt, prepend=bp_filt[0]))
    plateau = dp < thr

    starts = np.where(np.diff(np.concatenate([[0], plateau.astype(int)])) == 1)[0]
    ends   = np.where(np.diff(np.concatenate([plateau.astype(int), [0]])) == -1)[0]

    min_len = int(min_duration_sec * fs)
    idx1, idx2 = [], []
    for s, e in zip(starts, ends):
        if (e - s) >= min_len:
            idx1.append(int(s))
            idx2.append(int(e))

    if debug:
        import matplotlib.pyplot as plt
        x = np.arange(len(bp_filt))
        plt.figure(figsize=(14, 6))
        plt.subplot(2, 1, 1)
        plt.plot(x, bp_filt, 'k', linewidth=0.5)
        for s, e in zip(idx1, idx2):
            plt.plot(s, bp_filt[s], 'ro')
            plt.plot(e, bp_filt[e], 'bo')
        plt.title("Calibration artifact detection")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(dp, 'k')
        plt.plot(thr * np.ones_like(dp), '--r')
        plt.title("Derivative threshold")
        plt.grid()
        plt.show()

    return idx1, idx2, bp_filt, dp

def detect_bp_peaks_custom(bp_filt, fs, mask_cal=None, prominence=20.0, min_distance_sec=0.4):
    n = len(bp_filt)
    if mask_cal is None:
        mask_cal = np.zeros(n, dtype=bool)
    else:
        mask_cal = np.asarray(mask_cal, dtype=bool)

    distance = int(min_distance_sec * fs)

    raw_peaks, _ = find_peaks(bp_filt, prominence=prominence, distance=distance)
    good = ~mask_cal[raw_peaks]
    peaks = raw_peaks[good]
    return np.sort(peaks)

def detect_bp_troughs(bp_filt, fs, peaks_idx):
    troughs = []
    if peaks_idx is None or len(peaks_idx) < 2:
        return np.array([], dtype=int)

    for i in range(len(peaks_idx) - 1):
        s = int(peaks_idx[i])
        e = int(peaks_idx[i + 1])
        if e <= s + 1:
            continue
        local = bp_filt[s:e]
        local_min = int(np.argmin(local))
        troughs.append(s + local_min)

    return np.array(troughs, dtype=int)

def compute_bp_derived_from_peaks(sig, t, peaks_idx, troughs_idx=None):
    n = len(sig)
    SBP = np.full(n, np.nan)
    DBP = np.full(n, np.nan)

    if peaks_idx is not None:
        for p in peaks_idx:
            p = int(p)
            if 0 <= p < n:
                SBP[p] = sig[p]

    if troughs_idx is not None:
        for tr in troughs_idx:
            tr = int(tr)
            if 0 <= tr < n:
                DBP[tr] = sig[tr]

    # Fill missing values by interpolation (same style as your snippet)
    for arr in (SBP, DBP):
        mask = np.isnan(arr)
        if (~mask).any():
            arr[mask] = np.interp(
                np.flatnonzero(mask),
                np.flatnonzero(~mask),
                arr[~mask]
            )

    MBP = (SBP + 2 * DBP) / 3.0
    return {"SBP": SBP, "DBP": DBP, "MBP": MBP}

