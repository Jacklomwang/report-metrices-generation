import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.run_sts_acq import (
    condition_median_bp_panel_allowed,
    detect_doppler_channel,
    mean_beat_quality_in_window,
)


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


def test_sts_bp_panel_ignores_isolated_outlier_when_condition_medians_are_valid():
    time = np.array([60.0, 61.0, 600.0, 601.0, 602.0])
    map_values = np.array([80.0, 90.0, 80.0, 39.0, 100.0])

    allowed, supine_median, standing_median = condition_median_bp_panel_allowed(
        time, map_values, 60.0, 240.0, 600.0, 780.0,
    )

    assert allowed
    assert supine_median == 85.0
    assert standing_median == 80.0


def test_sts_bp_panel_rejects_out_of_range_condition_median():
    time = np.array([60.0, 61.0, 600.0, 601.0, 602.0])
    map_values = np.array([80.0, 90.0, 35.0, 39.0, 100.0])

    allowed, _, standing_median = condition_median_bp_panel_allowed(
        time, map_values, 60.0, 240.0, 600.0, 780.0,
    )

    assert not allowed
    assert standing_median == 39.0


def test_sts_bp_panel_accepts_range_boundaries():
    time = np.array([60.0, 600.0])
    map_values = np.array([40.0, 200.0])

    allowed, supine_median, standing_median = condition_median_bp_panel_allowed(
        time, map_values, 60.0, 240.0, 600.0, 780.0,
    )

    assert allowed
    assert supine_median == 40.0
    assert standing_median == 200.0
