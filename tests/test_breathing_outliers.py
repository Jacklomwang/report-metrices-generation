import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_breathing_acq import exclude_high_peak_outlier


def test_isolated_high_peak_is_removed_without_backfilling():
    hr = np.array([80., 140., 82., 79., 81., 78.])
    selected = np.arange(5)
    kept, rejected = exclude_high_peak_outlier(hr, selected)
    assert kept.tolist() == [0, 2, 3, 4]
    assert rejected.tolist() == [1]
    assert np.mean(hr[kept]) == 80.5
    assert selected.tolist() == list(range(5))


def test_exactly_25_bpm_is_retained_and_above_25_is_removed():
    hr = np.array([105., 80., 80., 80., 80.])
    kept, rejected = exclude_high_peak_outlier(hr, np.arange(5))
    assert len(kept) == 5 and rejected.size == 0
    hr[0] += 0.01
    kept, rejected = exclude_high_peak_outlier(hr, np.arange(5))
    assert rejected.tolist() == [0]


def test_regular_peaks_and_low_outlier_are_not_rejected():
    for values in ([82., 83., 79., 81., 80.], [40., 80., 80., 80., 80.]):
        kept, rejected = exclude_high_peak_outlier(np.array(values), np.arange(5))
        assert len(kept) == 5 and rejected.size == 0


def test_small_or_nonfinite_groups_are_not_filtered():
    for values in ([], [130.], [80., 130.], [80., np.nan, 130.]):
        kept, rejected = exclude_high_peak_outlier(np.array(values), np.arange(len(values)))
        assert len(kept) == len(values) and rejected.size == 0


def test_rejection_is_not_recursive():
    hr = np.array([160., 110., 80., 80., 80.])
    kept, rejected = exclude_high_peak_outlier(hr, np.arange(5))
    assert rejected.tolist() == [0]
    assert kept.tolist() == [1, 2, 3, 4]


def test_reviewed_window_is_session_specific_and_can_be_overridden():
    from scripts.run_breathing_acq import resolve_breathing_window

    assert resolve_breathing_window("2064", "1") == (3.0, 4.0)
    assert resolve_breathing_window("2019", "1") == (7.0, 8.0)
    assert resolve_breathing_window("2064", "2") == (7.0, 8.0)
    assert resolve_breathing_window("2064", "1", 7, 8) == (7.0, 8.0)
