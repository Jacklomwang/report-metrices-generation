#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import math
import mimetypes
import sys
from datetime import datetime
from pathlib import Path


TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "html_report"
TEMPLATE_PATH = TEMPLATE_DIR / "participant_report_template.html"
REPORT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPORT_ROOT / "src"


def _load_bundle(bundle_path: Path) -> dict:
    with open(bundle_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_subject_metadata(sub_id: str, ses_id: str) -> dict:
    try:
        if str(SRC_ROOT) not in sys.path:
            sys.path.insert(0, str(SRC_ROOT))
        import subject_metadata as local_subject_metadata  # type: ignore

        return local_subject_metadata.build_subject_metadata(
            participant=sub_id,
            session=ses_id,
            task="report",
        )
    except Exception as exc:
        return {"_load_error": str(exc)}


def _escape(text) -> str:
    return html.escape(str(text))


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    return False


def _as_float(value) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _format_number(value, digits: int = 2, signed: bool = False, missing: str = "—") -> str:
    if _is_missing(value):
        return missing
    try:
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, int):
            return f"{value:+d}" if signed and value > 0 else str(value)
        num = float(value)
        if signed:
            return f"{num:+.{digits}f}"
        return f"{num:.{digits}f}"
    except Exception:
        return _escape(value)


def _format_text(value, missing: str = "—") -> str:
    if value is None:
        return missing
    text = str(value).strip()
    return text if text else missing


def _subject_label(sub_id: str) -> str:
    return f"Subject {sub_id.replace('sub-', '')}"


def _session_plain(ses_id: str) -> str:
    return ses_id.replace("ses-", "")


def _task_section(bundle: dict, key: str) -> dict:
    section = bundle.get("metrics_by_task", {}).get(key, {})
    return section if isinstance(section, dict) else {}


def _task_present(bundle: dict, key: str) -> bool:
    section = _task_section(bundle, key)
    return bool(section) and section.get("present", 1) != 0


def _missing_tasks(bundle: dict) -> list[str]:
    labels = {
        "rest": "rest",
        "sts": "sts",
        "valsalva": "valsalva",
        "breathing": "breathing",
        "spirometry": "spirometry",
    }
    return [label for key, label in labels.items() if not _task_present(bundle, key)]


def _metric_card(label: str, value, unit: str = "", note: str = "", digits: int = 2, signed: bool = False) -> str:
    value_str = _format_number(value, digits=digits, signed=signed)
    unit_html = f' <span class="metric-unit">{_escape(unit)}</span>' if unit else ""
    note_html = f'<div class="metric-note">{_escape(note)}</div>' if note else ""
    return (
        '<div class="card metric">'
        f'<span class="metric-label">{_escape(label)}</span>'
        f'<div class="metric-value">{value_str}{unit_html}</div>'
        f'{note_html}'
        '</div>'
    )


def _measure_item(label: str, value, unit: str = "", digits: int = 2, text_value: bool = False) -> str:
    value_str = _format_text(value) if text_value else _format_number(value, digits=digits)
    unit_str = f" {_escape(unit)}" if unit else ""
    return f'<div><span>{_escape(label)}</span><strong>{value_str}{unit_str}</strong></div>'


def _notice(label: str, text: str, kind: str = "danger") -> str:
    extra = " info" if kind == "info" else ""
    return f'<div class="notice{extra}"><strong>{_escape(label)}</strong><span>{_escape(text)}</span></div>'


def _chapter_row(number: str, title: str, subtitle: str, status: str, missing: bool = False) -> str:
    status_class = "status missing" if missing else "status"
    return (
        '<div class="chapter-row">'
        f'<span>{_escape(number)}</span>'
        '<div>'
        f'<strong>{_escape(title)}</strong>'
        f'<small>{_escape(subtitle)}</small>'
        '</div>'
        f'<b class="{status_class}">{_escape(status)}</b>'
        '</div>'
    )


def _page(section_id: str, eyebrow: str, title: str, page_number: str, lead: str, body_html: str, footer_left: str) -> str:
    return (
        f'<section class="report-page" id="{_escape(section_id)}">'
        '<header class="page-header">'
        '<div>'
        f'<span class="eyebrow">{_escape(eyebrow)}</span>'
        f'<h2>{_escape(title)}</h2>'
        '</div>'
        f'<span class="page-number">{_escape(page_number)}</span>'
        '</header>'
        f'<p class="page-lead">{_escape(lead)}</p>'
        f'{body_html}'
        f'<footer class="page-footer"><span>{_escape(footer_left)}</span><span>Research use only</span></footer>'
        '</section>'
    )


def _embed_image(fig_path: str | None) -> str | None:
    if not fig_path:
        return None
    path = Path(fig_path)
    if not path.exists() or not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _figure_card(title: str, subtitle: str, fig_path: str | None, alt: str) -> str:
    src = _embed_image(fig_path)
    if src:
        media = f'<img class="report-image" src="{src}" alt="{_escape(alt)}">'
    else:
        media = f'<div class="figure-empty">Figure not available: {_escape(title)}</div>'
    return (
        '<div class="card chart-card">'
        '<div class="chart-title">'
        f'<strong>{_escape(title)}</strong>'
        f'<span>{_escape(subtitle)}</span>'
        '</div>'
        f'{media}'
        '</div>'
    )


# Predicted-range (LLN/ULN) values come from merge_subject_all_metrics_only.py,
# which looks them up per subject from the shared spirometry predicted-values CSV.
# A row with no LLN/ULN for that subject (e.g. missing demographic data upstream)
# falls back to showing just the value, no band.
_SPIRO_ROWS = [
    # key,     label,         sub label,               unit,   digits, whole LLN key,        whole ULN key
    ("fev1",  "FEV1",       "first-second volume",    "L",   1, "FEV1_LLN",          "FEV1_ULN"),
    ("fvc",   "FVC",        "forced vital capacity",  "L",   1, "FVC_LLN",           "FVC_ULN"),
    ("ratio", "FEV1 / FVC", "calculated ratio",       "",    2, "FEV1_over_FVC_LLN", "FEV1_over_FVC_ULN"),
]

# Band always occupies this fraction span of the track, aligned across every row.
_SPIRO_BAND_START_FRAC = 4 / 6
_SPIRO_BAND_END_FRAC = 5 / 6

_SPIRO_TRACK_X0 = 150.0
_SPIRO_TRACK_X1 = 610.0
_SPIRO_ROW_TOP = 54.0
_SPIRO_ROW_H = 56.0


def _spiro_marker_frac(value: float | None, band_min: float, band_max: float) -> tuple[float, bool]:
    """Return (clamped fraction along the track, whether the raw value was off-scale).

    The band sits at [4/6, 5/6] of the track; to keep that fixed, the track's value
    range extends 4x the band width to the left of the band and 1x to the right.
    """
    if value is None:
        return _SPIRO_BAND_START_FRAC, False
    range_width = band_max - band_min
    if range_width <= 0:
        return 0.5, False
    track_min = band_min - 4 * range_width
    track_max = band_max + range_width
    frac = (value - track_min) / (track_max - track_min)
    clamped = frac < 0.0 or frac > 1.0
    return max(0.0, min(1.0, frac)), clamped


def _spirometry_svg(fev1, fvc, fev1_over_fvc, whole: dict) -> str:
    values = {"fev1": _as_float(fev1), "fvc": _as_float(fvc), "ratio": _as_float(fev1_over_fvc)}

    band_x0 = _SPIRO_TRACK_X0 + _SPIRO_BAND_START_FRAC * (_SPIRO_TRACK_X1 - _SPIRO_TRACK_X0)
    band_x1 = _SPIRO_TRACK_X0 + _SPIRO_BAND_END_FRAC * (_SPIRO_TRACK_X1 - _SPIRO_TRACK_X0)

    rows_svg = []
    for i, (key, label, sub_label, unit, digits, lln_key, uln_key) in enumerate(_SPIRO_ROWS):
        value = values[key]
        y = _SPIRO_ROW_TOP + i * _SPIRO_ROW_H
        band_min = _as_float(whole.get(lln_key))
        band_max = _as_float(whole.get(uln_key))
        has_band = band_min is not None and band_max is not None and band_max > band_min

        value_text = _format_number(value, digits) if value is not None else "—"
        unit_text = f'<tspan class="spiro-value-unit"> {_escape(unit)}</tspan>' if unit else ""

        if has_band:
            frac, clamped = _spiro_marker_frac(value, band_min, band_max)
            x = _SPIRO_TRACK_X0 + frac * (_SPIRO_TRACK_X1 - _SPIRO_TRACK_X0)

            marker = ""
            if value is not None:
                if clamped:
                    direction = -1 if x <= _SPIRO_TRACK_X0 else 1
                    tip_x = x + direction * 9
                    marker = (
                        f'<polygon class="spiro-offscale" points="{x:.1f},{y-13:.1f} {x:.1f},{y+13:.1f} {tip_x:.1f},{y:.1f}"/>'
                    )
                else:
                    marker = f'<line class="spiro-marker" x1="{x:.1f}" y1="{y-13:.1f}" x2="{x:.1f}" y2="{y+13:.1f}"/>'

            rows_svg.append(f'''
      <text class="spiro-row-label" x="20" y="{y-3:.1f}">{_escape(label)}</text>
      <text class="spiro-row-sub" x="20" y="{y+11:.1f}">{_escape(sub_label)}</text>
      <rect class="spiro-track" x="{_SPIRO_TRACK_X0:.1f}" y="{y-4:.1f}" width="{_SPIRO_TRACK_X1 - _SPIRO_TRACK_X0:.1f}" height="8" rx="4"/>
      <rect class="spiro-band" x="{band_x0:.1f}" y="{y-8:.1f}" width="{band_x1 - band_x0:.1f}" height="16" rx="4"/>
      <text class="spiro-edge-label" x="{band_x0:.1f}" y="{y+26:.1f}" text-anchor="middle">{_format_number(band_min, digits)}</text>
      <text class="spiro-edge-label" x="{band_x1:.1f}" y="{y+26:.1f}" text-anchor="middle">{_format_number(band_max, digits)}</text>
      {marker}
      <text class="spiro-value" x="670" y="{y+4:.1f}" text-anchor="end">{value_text}{unit_text}</text>''')
        else:
            # No predicted range for this subject/metric (e.g. missing age/sex/height
            # upstream) — show the value alone with a muted note instead of a track.
            rows_svg.append(f'''
      <text class="spiro-row-label" x="20" y="{y-3:.1f}">{_escape(label)}</text>
      <text class="spiro-row-sub" x="20" y="{y+11:.1f}">{_escape(sub_label)}</text>
      <text class="spiro-row-sub" x="{_SPIRO_TRACK_X0:.1f}" y="{y+4:.1f}">Predicted range unavailable</text>
      <text class="spiro-value" x="670" y="{y+4:.1f}" text-anchor="end">{value_text}{unit_text}</text>''')

    legend_y = _SPIRO_ROW_TOP + len(_SPIRO_ROWS) * _SPIRO_ROW_H + 6
    legend_line_y = legend_y - 16
    height = legend_y + 24

    return f'''<svg class="chart" viewBox="0 0 700 {height:.0f}" role="img" aria-label="Participant spirometry values plotted against predicted values">
      {"".join(rows_svg)}
      <line class="spiro-legend-rule" x1="20" y1="{legend_line_y:.1f}" x2="680" y2="{legend_line_y:.1f}"/>
      <line class="spiro-marker" x1="20" y1="{legend_y-4:.1f}" x2="20" y2="{legend_y+4:.1f}"/>
      <text class="spiro-legend-label" x="30" y="{legend_y+4:.1f}">Participant value</text>
      <rect class="spiro-band" x="170" y="{legend_y-6:.1f}" width="18" height="12" rx="3"/>
      <text class="spiro-legend-label" x="196" y="{legend_y+4:.1f}">Predicted Values</text>
    </svg>'''


def _task_note(bundle: dict, task: str) -> str:
    note = _task_section(bundle, task).get("note")
    if not note:
        return ""
    return f'<div class="inline-note">{_escape(str(note))}</div>'


def _render_overview(bundle: dict, metadata: dict) -> str:
    whole = bundle.get("whole", {})
    missing = _missing_tasks(bundle)
    missing_text = "None noted" if not missing else ", ".join(missing)
    status_text = "Completed" if not missing else "Partial"
    participant = _subject_label(bundle.get("sub_id", "sub-unknown"))
    session = _session_plain(bundle.get("ses_id", "ses-unknown"))
    age_text = _format_number(metadata.get("age") if isinstance(metadata, dict) else None, 0)
    height_text = _format_text(metadata.get("height_cm") if isinstance(metadata, dict) else None)
    if height_text != "—":
        height_text += " cm"
    weight_text = _format_text(metadata.get("weight_kg") if isinstance(metadata, dict) else None)
    if weight_text != "—":
        weight_text += " kg"
    grid = (
        '<div class="participant-grid">'
        f'<div><span>Participant</span><strong>{_escape(participant)}</strong></div>'
        f'<div><span>Session</span><strong>{_escape(session)}</strong></div>'
        f'<div><span>Age</span><strong>{_escape(age_text)} years</strong></div>'
        '<div><span>Scan time</span><strong>—</strong></div>'
        f'<div><span>Testing status</span><strong>{_escape(status_text)}</strong></div>'
        f'<div><span>Height</span><strong>{_escape(height_text)}</strong></div>'
        f'<div><span>Weight</span><strong>{_escape(weight_text)}</strong></div>'
        f'<div><span>Missing tasks</span><strong>{_escape(missing_text)}</strong></div>'
        '</div>'
    )
    notices = [
        _notice(
            "Not for clinical use",
            "This report summarizes research assessments and does not carry diagnostic or prescriptive authority. Measurements can vary with your physiological state on the day of testing.",
        )
    ]
    if missing:
        notices.append(_notice("Incomplete source data", f"The following task outputs are currently missing or incomplete: {missing_text}.", kind="info"))
    neuro_found = bool(isinstance(metadata, dict) and metadata.get("neuropsych", {}).get("found"))
    if not neuro_found:
        notices.append(_notice("Incomplete source data", "No neuropsychological assessment record was found for this subject in the source dataset.", kind="info"))
    if isinstance(metadata, dict) and metadata.get("_load_error"):
        notices.append(_notice("Metadata fallback", str(metadata["_load_error"]), kind="info"))
    report_map = (
        '<div class="chapter-list">'
        + _chapter_row("02", "Cognitive testing", "MoCA score and reaction indices", "Separate page", missing=not neuro_found)
        + _chapter_row("03", "Spirometry", "FVC, FEV1, ratio, and peak flow", "Separate page", missing=not _task_present(bundle, "spirometry"))
        + _chapter_row("04", "Resting cardiovascular", "Heart rate variability and blood pressure", "Figures added", missing=not _task_present(bundle, "rest"))
        + _chapter_row("05", "Autonomic testing", "Supine-to-stand, Valsalva, and deep breathing", "Grouped chapter", missing=not any(_task_present(bundle, k) for k in ["sts", "valsalva", "breathing"]))
        + '</div>'
    )
    body = grid + ''.join(notices) + '<h3 class="section-heading">Report map</h3>' + report_map
    return _page(
        "overview",
        "Physiological & neuropsychological testing",
        "Your study testing report",
        "01 / Overview",
        "A participant-friendly summary of the research assessments completed during this session. Each test family has its own page so findings are easier to review and future sections can be added consistently.",
        body,
        f"LC Study · {participant}",
    )


def _render_cognitive(bundle: dict, metadata: dict) -> str:
    neuro = metadata.get("neuropsych", {}) if isinstance(metadata, dict) else {}
    moca_total = neuro.get("MoCA_Total") if isinstance(neuro, dict) else None
    hero_value = _format_number(moca_total, 0)
    note = "Most recent MoCA value available in the source metadata. A single score is one point-in-time research measurement."
    if not neuro.get("found"):
        note = "This subject has no neuropsychological assessment on record in the source dataset."
    elif _is_missing(moca_total):
        note = "A neuropsych record exists for this subject, but the MoCA total is not filled in yet."
    body = (
        '<div class="score-band">'
        '<div class="card hero-score"><div>'
        f'<strong>{hero_value}<span class="metric-unit">/30</span></strong>'
        '<span>MoCA total</span>'
        '</div></div>'
        '<div class="card interpretation">'
        '<span class="eyebrow">Participant context</span>'
        f'<h3>{_escape(note)}</h3>'
        '<p>A score is one point-in-time research measurement. A healthcare professional can provide appropriate follow-up interpretation if you have concerns.</p>'
        '</div>'
        '</div>'
        '<h3 class="section-heading">Reaction indices <small>Awaiting values from source dataset</small></h3>'
        '<div class="disclosure"><button type="button">About this assessment +</button><div class="disclosure-content">The MoCA is a screening assessment, not a diagnosis. Performance may be influenced by language, education, fatigue, hearing, vision, and the testing environment.</div></div>'
    )
    return _page(
        "cognitive",
        "Section 01",
        "Cognitive testing",
        "02 / Cognitive",
        "The Montreal Cognitive Assessment is a brief screening tool that samples several cognitive functions. The reaction-time placeholders are kept in the layout so we can connect them later without redesigning the report.",
        body,
        "LC Study · Cognitive testing",
    )


def _render_spirometry(bundle: dict) -> str:
    whole = bundle.get("whole", {})
    fev1 = whole.get("FEV1")
    fvc = whole.get("FVC")
    fev1_over_fvc = whole.get("FEV1_over_FVC")
    body = (
        '<div class="grid grid-3">'
        + _metric_card("FEV1", fev1, "L", "First-second volume")
        + _metric_card("FVC", fvc, "L", "Forced vital capacity")
        + _metric_card("FEV1 / FVC", fev1_over_fvc, "", "Calculated from reported values")
        + '</div>'
        '<h3 class="section-heading">Volume comparison</h3>'
        '<div class="card chart-card">'
        '<div class="chart-title"><strong>Participant values and expected reference range</strong><span>Illustrative scale</span></div>'
        f'{_spirometry_svg(fev1, fvc, fev1_over_fvc, whole)}'
        '<p class="clinical-note">The shaded band represents the literature-predicted (Bowerman, 2022) values for the FVC maneuver, '
        'based on demographic information. If it is not displayed, we may be missing some demographic information '
        'from you (age, sex, or height). The vertical mark indicates where your result lies against the predicted '
        'values, though it may be impacted by the quality of the spirometry maneuvers performed.</p>'
        '<p class="clinical-note">This is not clinical advice. If you have any questions or concerns regarding '
        'these results, please talk to a doctor.</p>'
        '</div>'
        + _task_note(bundle, "spirometry") +
        '<div class="disclosure"><button type="button">How to read spirometry values +</button><div class="disclosure-content">FEV1 is the forced expiratory volume in one second and FVC is the forced vital capacity, and FEV1/FVC is a ratio typically used for clinical diagnosis of obstructed lung disorders. Interpretation typically considers age, sex, height, reference equations, and test quality. These results should not be substituted for medical advice; please consult a doctor if you have any concerns.</div></div>'
    )
    return _page(
        "spirometry",
        "Section 02",
        "Spirometry",
        "03 / Respiratory",
        "Spirometry measures how much air you can exhale and how quickly you can exhale it. Results are presented separately from cognitive testing for clearer review.",
        body,
        "LC Study · Spirometry",
    )


def _render_resting(bundle: dict) -> str:
    whole = bundle.get("whole", {})
    body = (
        '<div class="measure-groups">'
        '<div class="card measure-group" style="--group-color:var(--accent-deep)"><h3>1. ECG measures</h3><div class="measure-list">'
        + _measure_item("Mean heart rate", whole.get("mean_HR"), "bpm")
        + _measure_item("Mean RR", whole.get("mean_RR"), "s", digits=4)
        + _measure_item("RMSSD", whole.get("RMSSD"), "ms")
        + _measure_item("LF / HF ratio", whole.get("LF_HF_ratio"))
        + '</div></div>'
        '<div class="card measure-group" style="--group-color:var(--success)"><h3>2. Blood pressure measures</h3><div class="measure-list">'
        + _measure_item("Systolic", whole.get("mean_sysBP"), "mmHg")
        + _measure_item("Mean arterial", whole.get("mean_MAP"), "mmHg")
        + _measure_item("Diastolic", whole.get("mean_diaBP"), "mmHg")
        + _measure_item("Source", "Continuous ABP", text_value=True)
        + '</div></div>'
        '<div class="card measure-group" style="--group-color:var(--respiratory)"><h3>3. Respiratory measures</h3><div class="measure-list">'
        + _measure_item("End-tidal CO2", None, "mmHg")
        + _measure_item("Tidal volume", None, "L")
        + _measure_item("Minute ventilation", None, "L/min")
        + _measure_item("Status", "Awaiting data", text_value=True)
        + '</div></div>'
        '<div class="card measure-group" style="--group-color:var(--doppler)"><h3>4. Doppler measures</h3><div class="measure-list">'
        + _measure_item("Mean flow velocity", None, "cm/s")
        + _measure_item("Pulsatility index", None)
        + _measure_item("Laterality", None, text_value=True)
        + _measure_item("Status", "Awaiting data", text_value=True)
        + '</div></div>'
        '</div>'
        '<div class="legend"><span style="--legend-color:var(--accent-deep)">ECG / heart rate</span><span style="--legend-color:var(--success)">Blood pressure</span><span style="--legend-color:var(--respiratory)">Respiratory</span><span style="--legend-color:var(--doppler)">Doppler</span></div>'
        + _task_note(bundle, "rest") +
        '<div class="disclosure"><button type="button">About HRV measures +</button><div class="disclosure-content">RMSSD is a time-domain heart-rate-variability measure associated primarily with parasympathetic activity. LF/HF is often reported as a frequency-domain index; interpretation remains context dependent.</div></div>'
    )
    return _page(
        "cardiovascular",
        "Section 03",
        "Resting cardiovascular",
        "04 / Resting state",
        "Resting measurements are organized by recording modality so that future respiratory and Doppler results can be inserted without changing the page structure.",
        body,
        "LC Study · Resting cardiovascular",
    )


def _render_autonomic_overview(bundle: dict) -> str:
    body = (
        '<div class="chapter-banner"><div><span class="eyebrow" style="color:oklch(78% .08 245)">One coherent chapter</span><h3 style="margin:8px 0 0;font:650 28px var(--font-display)">Response to posture, strain, and breathing</h3><p>Review the trends first, then the calculated indices and participant-oriented explanation.</p></div><div class="chapter-tests"><div>01 · Supine-to-stand response</div><div>02 · Valsalva maneuver</div><div>03 · Deep breathing response</div></div></div>'
        '<div class="task-grid">'
        '<div class="card task-card"><svg viewBox="0 0 180 118" role="img" aria-label="Person moving from lying on a bed to standing on the floor"><line class="axis" x1="8" y1="96" x2="172" y2="96"/><path class="task-stroke" d="M10 74h60v6H10zM14 80v10M66 80v10M10 74v-7h14"/><circle class="task-stroke" cx="20" cy="66" r="5"/><path class="task-stroke" d="M26 70h38"/><path class="task-secondary" d="M80 46c11-5 22-4 30 4m0 0-9-1m9 1-1 9"/><circle class="task-stroke" cx="140" cy="36" r="7"/><path class="task-stroke" d="M140 43v25m0-19-13 9m13-9 13 9m-13 10-9 19m9-19 9 19"/></svg><h3>Supine to stand</h3><p>Rest quietly while lying down, then stand when instructed while heart rate and blood pressure continue recording.</p></div>'
        '<div class="card task-card"><svg viewBox="0 0 180 118" role="img" aria-label="Person blowing into a syringe"><path class="task-stroke" d="M46 20c-15 0-26 12-26 29 0 11 5 17 5 25 0 8-4 12-4 18 0 4 3 6 8 6h21"/><path class="task-stroke" d="M46 20c13 0 22 10 24 23 1 9-1 15-1 22"/><circle cx="55" cy="52" r="2.4" fill="var(--accent-deep)"/><path class="task-stroke" d="M42 40q7-4 13-1"/><path class="task-stroke" d="M63 67c-4 2-9 3-14 2"/><path class="task-secondary" d="M69 64q7-2 13 0"/><rect class="task-stroke" x="82" y="58" width="46" height="13" rx="1.5"/><path class="task-stroke" d="M94 58v13M106 58v13M118 58v13"/><path class="task-stroke" d="M128 61h9v7h-9M137 64.5h11M148 60v9"/></svg><h3>Valsalva maneuver</h3><p>Blow steadily into a syringe against resistance for the instructed interval, followed by recovery, while synchronized signals are recorded.</p></div>'
        '<div class="card task-card"><svg viewBox="0 0 180 118" role="img" aria-label="Breathing cycle with an inhale circle and an exhale circle"><circle cx="52" cy="59" r="27" fill="none" stroke="var(--success)" stroke-width="5"/><path d="M52 32l6 6-6 6" fill="none" stroke="var(--success)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><text x="52" y="63" text-anchor="middle" fill="var(--success)" style="font:700 12px var(--font-mono)">inhale</text><circle cx="128" cy="59" r="27" fill="none" stroke="var(--danger)" stroke-width="5"/><path d="M128 86l-6-6 6-6" fill="none" stroke="var(--danger)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><text x="128" y="63" text-anchor="middle" fill="var(--danger)" style="font:700 12px var(--font-mono)">exhale</text></svg><h3>Deep breathing</h3><p>Follow paced inhale and exhale cues so breathing-linked changes in heart rate can be assessed.</p></div>'
        '</div>'
    )
    return _page(
        "autonomic",
        "Section 04",
        "Autonomic testing",
        "05 / Chapter overview",
        "The autonomic nervous system helps regulate involuntary functions such as heart rate and blood pressure. This chapter groups the three autonomic assessments while preserving a dedicated page for each.",
        body,
        "LC Study · Autonomic testing",
    )


def _render_sts(bundle: dict) -> str:
    whole = bundle.get("whole", {})
    figures = bundle.get("figures", {})
    body = (
        '<div class="grid grid-4">'
        + _metric_card("Baseline HR", whole.get("baseline_HR"), "bpm")
        + _metric_card("Plateau HR", whole.get("plateau_HR"), "bpm")
        + _metric_card("Delta HR", whole.get("delta_HR"), "bpm", signed=True)
        + _metric_card("Delta BP", whole.get("delta_BP"), "mmHg", signed=True)
        + '</div>'
        '<h3 class="section-heading">Transition trend</h3>'
        + _figure_card("Heart rate and mean blood pressure", "Source-derived figure", figures.get("STS_HR_MAP"), "Supine to stand figure")
        + _task_note(bundle, "sts") +
        '<div class="disclosure"><button type="button">About orthostatic response +</button><div class="disclosure-content">Orthostatic intolerance describes symptoms that occur on standing and improve when lying down. Formal interpretation considers symptoms, timing, heart-rate change, blood-pressure change, medications, and clinical context.</div></div>'
    )
    return _page(
        "sts",
        "Autonomic testing · 01",
        "Supine to stand",
        "06 / STS",
        "This assessment summarizes the change in heart rate and blood pressure from lying down to standing.",
        body,
        "LC Study · Autonomic · Supine to stand",
    )


def _render_valsalva(bundle: dict) -> str:
    whole = bundle.get("whole", {})
    figures = bundle.get("figures", {})
    figure_block = '<div class="figure-grid">'
    figure_block += _figure_card("Valsalva heart-rate response", "Source-derived figure", figures.get("Valsalva_plot"), "Valsalva heart rate figure")
    figure_block += '</div>'
    body = (
        '<div class="grid grid-3">'
        + _metric_card("Valsalva ratio", whole.get("Valsalva_ratio"), "", "Reported in source report")
        + _metric_card("MAP phase II drop", None, "mmHg", "Connect source dataset")
        + _metric_card("MAP phase IV overshoot", None, "mmHg", "Connect source dataset")
        + '</div>'
        '<h3 class="section-heading">Synchronized waveforms</h3>'
        + figure_block
        + _task_note(bundle, "valsalva")
    )
    return _page(
        "valsalva",
        "Autonomic testing · 02",
        "Valsalva maneuver",
        "07 / Valsalva",
        "The Valsalva maneuver records cardiovascular responses during a controlled strain and recovery. The figure below reflects the source-derived best repetition output when available.",
        body,
        "LC Study · Autonomic · Valsalva",
    )


def _render_deep_breathing(bundle: dict) -> str:
    whole = bundle.get("whole", {})
    figures = bundle.get("figures", {})
    body = (
        '<div class="grid grid-3">'
        + _metric_card("E:I ratio", whole.get("E_I_ratio"), "")
        + _metric_card("Delta HR", whole.get("delta_HR_responses"), "bpm")
        + _metric_card("Cycle count", None, "", "Connect source dataset")
        + '</div>'
        '<h3 class="section-heading">Breathing-linked heart-rate response</h3>'
        + _figure_card("Heart rate waveform", "Source-derived figure", figures.get("DeepBreathing_plot"), "Deep breathing figure")
        + _task_note(bundle, "breathing") +
        '<div class="disclosure"><button type="button">About E:I ratio +</button><div class="disclosure-content">The expiratory-to-inspiratory ratio compares the longest RR interval during expiration with the shortest RR interval during inspiration. It is interpreted in relation to age, test conditions, and other autonomic measures.</div></div>'
    )
    return _page(
        "deep-breathing",
        "Autonomic testing · 03",
        "Deep breathing",
        "08 / Deep breathing",
        "The deep-breathing assessment evaluates heart-rate variation across slow breathing cycles, a response associated with cardiovagal function.",
        body,
        "LC Study · Autonomic · Deep breathing",
    )


def _render_glossary(bundle: dict) -> str:
    body = (
        '<h3 class="section-heading">Glossary</h3>'
        '<div class="glossary">'
        '<div><strong>MoCA</strong><span>Montreal Cognitive Assessment.</span></div>'
        '<div><strong>FEV1</strong><span>Forced expiratory volume in one second.</span></div>'
        '<div><strong>FVC</strong><span>Forced vital capacity.</span></div>'
        '<div><strong>PEF</strong><span>Peak expiratory flow.</span></div>'
        '<div><strong>RR interval</strong><span>Time between consecutive R-waves on an ECG.</span></div>'
        '<div><strong>RMSSD</strong><span>Root mean square of successive differences.</span></div>'
        '<div><strong>LF/HF ratio</strong><span>Ratio of low- to high-frequency HRV power.</span></div>'
        '<div><strong>ABP</strong><span>Arterial blood pressure.</span></div>'
        '<div><strong>STS</strong><span>Supine-to-stand test.</span></div>'
        '<div><strong>E:I ratio</strong><span>Expiratory-to-inspiratory ratio.</span></div>'
        '</div>'
        '<h3 class="section-heading">References</h3>'
        '<ol class="references">'
        '<li>Nasreddine ZS, Phillips NA, Bédirian V, et al. The Montreal Cognitive Assessment, MoCA. J Am Geriatr Soc. 2005;53(4):695–699.</li>'
        '<li>Shaffer F, Ginsberg JP. An overview of heart rate variability metrics and norms. Front Public Health. 2017;5:258.</li>'
        '<li>Ewing DJ, Martyn CN, Young RJ, Clarke BF. The value of cardiovascular autonomic function tests. Diabetes Care. 1985;8(5):491–498.</li>'
        '<li>Risk M, Bril V, Broadbridge C, Cohen A. Heart rate variability measurement in diabetic neuropathy. Diabetes Technol Ther. 2001;3(1):63–76.</li>'
        '<li>Bryarly M, Phillips L, Fu Q, Vernino S, Levine BD. Postural orthostatic tachycardia syndrome. J Am Coll Cardiol. 2019;73(10):1207–1228.</li>'
        '<li>Zygmunt A, Stanczyk J. Methods of evaluation of autonomic nervous system function. Arch Med Sci. 2010;6(1):11–18.</li>'
        '</ol>'
        + _notice("Reminder", "This research report is intended to share recorded study measurements with the participant. It is not a clinical diagnosis or treatment recommendation.")
    )
    return _page(
        "glossary",
        "Supporting information",
        "Glossary & references",
        "09 / Reference",
        "Definitions and citations are provided here so the report can stand on its own when it is shared or printed.",
        body,
        "LC Study · Glossary & references",
    )


def build_report_html(bundle: dict, metadata: dict) -> str:
    sub_id = bundle.get("sub_id", "sub-unknown")
    ses_id = bundle.get("ses_id", "ses-unknown")
    title = f"LC Study Participant Testing Report - {sub_id} {ses_id}"

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Missing HTML template: {TEMPLATE_PATH}")

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    body = ''.join([
        _render_overview(bundle, metadata),
        _render_cognitive(bundle, metadata),
        _render_spirometry(bundle),
        _render_resting(bundle),
        _render_autonomic_overview(bundle),
        _render_sts(bundle),
        _render_valsalva(bundle),
        _render_deep_breathing(bundle),
        _render_glossary(bundle),
    ])

    return (
        template
        .replace("__REPORT_TITLE__", _escape(title))
        .replace("__REPORT_BODY__", body)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Render a subject/session HTML report using the standalone HTML template.")
    ap.add_argument("--out_root", default="derived", help="Output root folder (e.g., derived)")
    ap.add_argument("--sub", required=True, help="Subject code like 2062")
    ap.add_argument("--ses", default="1", help="Session number like 1 or 2")
    ap.add_argument("--quarto_bin", default="quarto", help="Deprecated compatibility argument; ignored.")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    sub_id = f"sub-{args.sub}"
    ses_id = f"ses-{args.ses}"
    base_dir = out_root / sub_id / ses_id
    bundle_path = base_dir / f"{sub_id}_{ses_id}_all_metrics.json"
    html_path = base_dir / f"{sub_id}_{ses_id}_report.html"

    if not bundle_path.exists():
        print(f"[ERROR] Missing merged JSON bundle: {bundle_path}")
        return 1

    bundle = _load_bundle(bundle_path)
    metadata = _load_subject_metadata(sub_id, ses_id)
    html_text = build_report_html(bundle, metadata)
    html_path.write_text(html_text, encoding="utf-8")
    print(f"[OK] Saved HTML report: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
