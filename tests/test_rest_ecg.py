import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rest_ecg import average_ecg_waveform, calculate_interval_metrics


def test_ecg_interval_metrics_use_median_valid_durations():
    fs = 250.0
    waves = {
        "ECG_P_Onsets": [100, 350, 600],
        "ECG_P_Offsets": [128, 378, 628],
        "ECG_R_Onsets": [144, 394, 644],
        "ECG_R_Offsets": [168, 418, 668],
        "ECG_T_Offsets": [242, 492, 742],
    }

    metrics = calculate_interval_metrics(waves, fs)

    assert metrics["ecg_p_duration_ms"] == 112.0
    assert metrics["ecg_qrs_duration_ms"] == 96.0
    assert metrics["ecg_pq_time_ms"] == 176.0
    assert metrics["ecg_qt_time_ms"] == 392.0
    assert metrics["ecg_interval_n_beats"] == 3


def test_average_ecg_waveform_rejects_inverted_outlier():
    fs = 100.0
    time = np.arange(-35, 51) / fs
    beat = np.exp(-((time / 0.035) ** 2))
    signal = np.zeros(600)
    peaks = np.array([100, 200, 300, 400, 500])
    for peak in peaks:
        signal[peak - 35:peak + 51] = beat
    signal[peaks[-1] - 35:peaks[-1] + 51] = -beat

    average = average_ecg_waveform(signal, peaks, fs)

    assert average["n_beats"] == 4
    assert np.max(average["mean"]) > 0.9
