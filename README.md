# LCS Physio Report Generation

This project generates per-task physiological report metrics from Biopac `.acq` files, merges them into one subject/session bundle, and renders a standalone HTML report.

Current status:
- command-line driven
- HTML report is the active final output
- no active Streamlit frontend in this repo
- no active MATLAB report path in this repo

## What the project does

For one subject/session, the pipeline can generate:
- `rest`: resting ECG/HRV and morphology, BP, respiratory, ETCO2, and Doppler metrics plus QC figures
- `sts`: supine-to-stand heart-rate and blood-pressure metrics, onset marker, and an optional mean Doppler velocity panel that requires mean beat quality >= 0.8 in both supine and standing periods
- `valsalva`: Valsalva ratio from artifact-rejected median HR, synchronized HR/BP figure, and SBP/MAP phase summaries
- `breathing`: deep-breathing metrics and a minute 7-8 HR figure
- `spirometry`: FEV1 / FVC / PEF extracted metrics

Then it can:
- merge task outputs into one combined bundle
- render one subject-specific HTML report

## Repository layout

Main source folders:
- `scripts/`: task runners, merge script, and final HTML renderer
- `html_report/`: active HTML report template
- `src/`: shared processing helpers, the local metadata loader, and the vendored Physio-QC resting processors

Generated/local-only folders:
- `derived/`: generated per-subject outputs
- `outputs/`: ad hoc generated outputs
- `logs/`: runtime logs
- `.venv/`: local environment

These generated/local folders should not be committed.

## Input data layout

The scripts expect a physio root like:

```text
/export02/projects/LCS/01_physio/
  sub-2062/
    ses-1/
      sub-2062_ses-1_task-rest_physio.acq
      sub-2062_ses-1_task-STS_physio.acq
      sub-2062_ses-1_task-valsalva_physio.acq
      sub-2062_ses-1_task-breath_physio.acq
```

Some scripts also read spirometry data from the shared spirometry CSV configured in the code.

## Metadata sources

The final HTML renderer is now self-contained with respect to metadata loading. It does **not** import code from the neighboring `physio-qc` repository.

Instead, this repo reads metadata directly from the shared phenotype/source files under:

```text
/export02/projects/LCS/05_phenotype/redcap_exports
```

This is used to populate fields such as:
- age
- BMI
- sex assigned at birth
- gender
- recording date
- researchers
- neuropsych summary including `MoCA_Total`

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you already have a working environment from `physio-qc`, you can also use that interpreter directly.

## Main workflow

Run the complete pipeline for one subject/session with one command:

```bash
python scripts/run_subject_all.py \
  --root /export02/projects/LCS/01_physio \
  --sub 2062 \
  --ses 1 \
  --one_based \
  --out_root derived
```

This runs the task scripts and writes per-task outputs under:

```text
derived/sub-2062/ses-1/
  rest/
  sts/
  valsalva/
  breathing/
  spirometry/
```

It then creates the merged MAT and JSON bundles and renders:

```text
derived/sub-2062/ses-1/sub-2062_ses-1_report.html
```

## Re-merge or re-render only

If task outputs already exist, rebuild the MAT and JSON bundle without repeating physiological processing:

```bash
python scripts/merge_subject_all_metrics_only.py \
  --out_root derived \
  --sub 2062 \
  --ses 1
```

This writes:

```text
derived/sub-2062/ses-1/sub-2062_ses-1_all_metrics.mat
derived/sub-2062/ses-1/sub-2062_ses-1_all_metrics.json
```

The merged bundle contains:
- `metrics_by_task`
- `whole`
- `figures`
- `sub_id`
- `ses_id`

Render the HTML again from an existing merged JSON bundle:

```bash
python scripts/render_subject_report_html.py \
  --out_root derived \
  --sub 2062 \
  --ses 1
```

This writes:

```text
derived/sub-2062/ses-1/sub-2062_ses-1_report.html
```

The active template is:
- `html_report/participant_report_template.html`

## Individual task scripts

If you want to run tasks separately instead of using `run_subject_all.py`, the main scripts are:
- `scripts/run_rest_acq.py`
- `scripts/run_sts_acq.py`
- `scripts/run_valsalva_acq.py`
- `scripts/run_breathing_acq.py`
- `scripts/run_spirometry_extract.py`

Example:

```bash
python scripts/run_rest_acq.py \
  --root /export02/projects/LCS/01_physio \
  --sub 2062 \
  --ses 1 \
  --save \
  --out_root derived
```

The resting runner auto-detects ECG, continuous blood pressure, respiratory
belt, pneumotach, CO2, and Doppler channels by name. `--ecg_ch` and `--bp_ch`
remain available only as backward-compatible fallbacks when channel names are
unusual.

## Opening the HTML report

One reliable way to view the output is to serve the report directory:

```bash
cd "derived/sub-2062/ses-1"
python3 -m http.server 8000
```

Then open:

```text
http://<server-name>:8000/sub-2062_ses-1_report.html
```

## Channel notes

The resting script uses Physio-QC-compatible name detection and does not require
channel numbers for normal LCS recordings. Its detected channel names are saved
in `rest_metrics.mat` for traceability.

STS, Valsalva, and breathing still use explicit ECG / BP / PPG channel numbers.
If their defaults are wrong for a subject, override them when calling the script.

`--one_based` means:
- channel 1 = first channel in the `.acq` file

Use `--one_based` with `run_subject_all.py` for the current LCS channel defaults.

## Resting processing

The report repository contains a focused copy of the required Physio-QC
processors under `src/physio_qc/`, so it does not require a neighboring
`physio-qc` checkout. The resting pipeline uses:

- NeuroKit cleaning and fixed NeuroKit R-peak detection for ECG/HRV
- NeuroKit DWT delineation for the average ECG waveform and median P/QRS/PQ/QT intervals
- the Physio-QC BP filter, delineator, and calibration-artifact exclusion
- Physio-QC respiratory-belt and BreathMetrics pneumotach processing
- the Physio-QC ETCO2 envelope extractor after A8 voltage conversion
- the Physio-QC Doppler wavelet filter, delineator, and beat-quality score

Valsalva repetition starts are read from Biopac `defl` event markers. An
explicit `--trig_ch` remains available as an override; automatic fallback only
uses channels whose names contain trigger/event/sync patterns and never scans
SpO2 or other physiological waveforms for trigger-like values.

Doppler noise is classified in 10-second windows stepped every 5 seconds. A
window is excluded when its time-weighted mean beat quality is below `0.8`.
The Doppler summary values are calculated only from retained peaks/troughs and
the excluded percentage is stored as `doppler_noisy_percent`.

## Troubleshooting

### Missing Python dependency
Example:
```text
ModuleNotFoundError: No module named 'bioread'
```
Use the correct virtual environment and install requirements.

### Final HTML report not created
Make sure you ran all three steps:
1. `run_subject_all.py`
2. `merge_subject_all_metrics_only.py`
3. `render_subject_report_html.py`

For the renderer itself, no neighboring `physio-qc` checkout is required anymore.

### Some tasks fail for a subject
Common causes:
- wrong ECG/BP channel assumptions
- missing task `.acq` file
- subject-specific signal quality problems

The merge/render steps can still be used on partial outputs if enough task files exist.

## Current active files for report output

Active:
- `scripts/run_subject_all.py`
- `scripts/merge_subject_all_metrics_only.py`
- `scripts/render_subject_report_html.py`
- `html_report/participant_report_template.html`
- `src/subject_metadata.py`

Legacy/removed from active path:
- Streamlit report frontend
- MATLAB report generation
- Quarto-based report rendering
- dependency on `physio-qc` metadata-loading code

## License

Internal research code. Add a public license before external distribution.
