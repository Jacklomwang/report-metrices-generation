import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_valsalva_acq import extract_valsalva_starts_from_event_markers
from src.valsalva_bp import (
    bp_phase_landmarks,
    phase_summary_metrics,
    prepare_hr_traces,
    valsalva_hr_metrics,
)


def test_prepare_hr_traces_suppresses_isolated_peak():
    sampling_rate = 4.0
    hr = np.full(40, 72.0)
    hr[19:22] = 180.0

    time, raw, filtered = prepare_hr_traces(hr, sampling_rate)

    assert len(time) == len(raw) == len(filtered)
    assert np.all(raw[19:22] == 180.0)
    assert filtered[20] == 72.0


def test_valsalva_ratio_uses_filtered_task_and_recovery_windows():
    time = np.arange(0.0, 80.0, 0.25)
    onset = 15.0
    hr = np.full(time.shape, 80.0)
    hr[np.argmin(np.abs(time - (onset + 10.0)))] = 120.0
    hr[np.argmin(np.abs(time - (onset + 25.0)))] = 60.0

    metrics = valsalva_hr_metrics(time, hr, onset)

    assert metrics["max_hr_task"] == 120.0
    assert metrics["min_hr_recovery"] == 60.0
    assert metrics["valsalva_ratio"] == 2.0
    assert metrics["max_hr_time"] == 10.0
    assert metrics["min_hr_time"] == 25.0


def test_bp_phase_landmarks_follow_expected_windows():
    onset = 15.0
    time = np.arange(0.0, 70.0, 0.25)
    relative = time - onset
    map_values = np.interp(
        relative,
        [-15.0, 0.0, 2.0, 6.0, 14.5, 16.5, 21.0, 45.0],
        [100.0, 100.0, 112.0, 82.0, 98.0, 75.0, 120.0, 100.0],
    )
    result = {"time_4hz": time, "map_4hz": map_values, "sbp_4hz": map_values + 25.0}

    landmarks = bp_phase_landmarks(result, onset)
    summary = phase_summary_metrics(landmarks)

    assert set(landmarks["events"]) == {
        "phase1_max", "phase2_nadir", "phase2_late_max", "phase3_nadir", "phase4_max"
    }
    assert summary["map_phase2_drop"] > 0
    assert summary["map_phase4_overshoot"] > 0
    assert summary["baseline_sbp"] > summary["baseline_map"]
    assert summary["sbp_phase1_from_baseline"] > 0
    assert summary["sbp_phase2_early_fall"] < 0
    assert summary["sbp_phase2_late_recovery"] > 0
    assert summary["sbp_phase3_drop"] < 0
    assert summary["sbp_phase4_rise"] > 0
    assert summary["map_phase1_from_baseline"] > 0
    assert summary["map_phase2_early_fall"] < 0
    assert summary["map_phase2_late_recovery"] > 0
    assert summary["map_phase3_drop"] < 0
    assert summary["map_phase4_rise"] > 0


def test_valsalva_starts_use_only_defl_event_markers():
    markers = [
        SimpleNamespace(type_code="apnd", sample_index=0),
        SimpleNamespace(type_code="defl", sample_index=15009),
        SimpleNamespace(type_code="defl", sample_index=39680),
        SimpleNamespace(type_code="defl", sample_index=58073),
    ]

    assert extract_valsalva_starts_from_event_markers(markers).tolist() == [15009, 39680, 58073]
