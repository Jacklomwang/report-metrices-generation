import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.render_subject_report_html import _format_scan_time
from src.subject_metadata import _load_redcap_metadata, _resolve_source_path


def test_source_path_falls_back_to_latest_dated_export(tmp_path):
    preferred = tmp_path / "Study_DATA.csv"
    older = tmp_path / "Study_DATA_2026-07-01.csv"
    latest = tmp_path / "Study_DATA_2026-08-26.csv"
    older.write_text("old", encoding="utf-8")
    latest.write_text("new", encoding="utf-8")

    resolved = _resolve_source_path(preferred, "Study_DATA_*.csv")

    assert resolved == latest


def test_redcap_loader_reads_age_height_and_weight(tmp_path):
    export = tmp_path / "study.csv"
    export.write_text(
        "redcap_survey_identifier,age,height_m,weight_kg,asab,gender\n"
        "sub-1001,37,1.72,68.4,1,1\n",
        encoding="utf-8",
    )

    entry = _load_redcap_metadata(export)["sub-1001"]

    assert entry["age"] == 37
    assert entry["height_m"] == 1.72
    assert entry["weight_kg"] == 68.4


def test_scan_time_is_formatted_for_report():
    assert _format_scan_time("2026-07-20T09:00") == "Jul 20, 2026 09:00"
