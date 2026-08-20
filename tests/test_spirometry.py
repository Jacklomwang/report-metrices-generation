import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.merge_subject_all_metrics_only import merge_to_all_metrics
from scripts.run_spirometry_extract import save_missing_spirometry_metrics


def test_missing_spirometry_remains_unavailable_after_merge(tmp_path):
    source_csv = tmp_path / "spiro_data.csv"
    save_missing_spirometry_metrics(
        tmp_path,
        "sub-999999",
        "ses-1",
        source_csv,
        "Spirometry unavailable for test subject",
    )

    merge_to_all_metrics(tmp_path, "999999", "1")

    bundle = loadmat(
        tmp_path / "sub-999999" / "ses-1" / "sub-999999_ses-1_all_metrics.mat",
        simplify_cells=True,
    )
    spirometry = bundle["metrics_by_task"]["spirometry"]
    assert spirometry["present"] == 0
    assert "unavailable" in spirometry["note"].lower()
    assert np.isnan(bundle["whole"]["FEV1"])
