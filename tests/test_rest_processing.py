import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_rest_acq import compute_doppler_noisy_windows, detect_channels


def _channels(*names):
    return [SimpleNamespace(name=name) for name in names]


def test_detect_channels_matches_lcs_rest_layout():
    channels = _channels(
        "Saturation, OXY100C",
        "Pulse, OXY100C",
        "RSP100C",
        "ECG100C",
        "Pulse, OXY100C",
        "Custom, AMI / HLT - A 6",
        "Custom, AMI / HLT - A 7",
        "Custom, AMI / HLT - A 8",
        "Rate, NIBP100D",
        "Custom, AMI / HLT - A10",
        "TSD117 - Medium-flow Pneumotach, DA100C",
    )

    detected = detect_channels(channels)

    assert {name: value[1] for name, value in detected.items()} == {
        "ecg": 3,
        "bp": 9,
        "rsp": 2,
        "spirometer": 10,
        "etco2": 7,
        "doppler": 5,
    }


def test_doppler_noisy_windows_use_quality_threshold():
    fs = 10.0
    troughs = np.arange(0, 201, 10)

    high_windows, high_mask = compute_doppler_noisy_windows(201, fs, troughs, np.full(20, 0.9))
    low_windows, low_mask = compute_doppler_noisy_windows(201, fs, troughs, np.full(20, 0.7))

    assert high_windows == []
    assert not high_mask.any()
    assert low_windows
    assert low_mask.any()
