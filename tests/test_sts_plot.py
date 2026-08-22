import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.run_sts_acq import detect_doppler_channel, mean_beat_quality_in_window


def test_sts_doppler_detection_prefers_a6_channel():
    channels = [
        SimpleNamespace(name="ECG100C"),
        SimpleNamespace(name="Custom, AMI / HLT - A 6"),
        SimpleNamespace(name="Custom, AMI / HLT - A 7"),
    ]

    channel, index = detect_doppler_channel(channels)

    assert index == 1
    assert channel.name.endswith("A 6")


def test_sts_doppler_quality_is_duration_weighted_within_condition():
    troughs = np.array([0, 2, 5, 8])
    scores = np.array([1.0, 0.5, 0.0])

    quality = mean_beat_quality_in_window(troughs, scores, 1.0, 1.0, 6.0)

    assert quality == 0.5


def test_sts_doppler_quality_is_nan_without_beats():
    quality = mean_beat_quality_in_window([], [], 250.0, 0.0, 300.0)

    assert np.isnan(quality)
