# LCS Physio Report Generation

This project generates per-task physiological report metrics from Biopac `.acq` files, merges them into one subject/session bundle, and renders a standalone HTML report.

Current status:
- command-line driven
- HTML report is the active final output
- no active Streamlit frontend in this repo
- no active MATLAB report path in this repo

## What the project does

For one subject/session, the pipeline can generate:
- `rest`: resting HR/HRV and BP summary metrics plus figures
- `sts`: supine-to-stand metrics plus figure
- `valsalva`: Valsalva ratio plus figure
- `breathing`: deep-breathing metrics plus figure
- `spirometry`: FEV1 / FVC / PEF extracted metrics

Then it can:
- merge task outputs into one combined bundle
- render one subject-specific HTML report

## Repository layout

Main source folders:
- `scripts/`: task runners, merge script, and final HTML renderer
- `html_report/`: active HTML report template
- `src/`: shared processing helpers, including the local metadata loader used by the report renderer

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

The current workflow is three steps.

### 1. Run per-task processing for one subject/session

```bash
python scripts/run_subject_all.py \
  --root /export02/projects/LCS/01_physio \
  --sub 2062 \
  --ses 1 \
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

Important:
- `run_subject_all.py` currently stops after task execution plus a combined `.mat` bundle
- it does **not** by itself produce the merged JSON or final HTML report

### 2. Build the merged MAT + JSON bundle

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

### 3. Render the final HTML report

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
  --one_based \
  --ecg_ch 6 \
  --bp_ch 10 \
  --save \
  --out_root derived
```

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

The task scripts use explicit channel arguments for ECG / BP / PPG where needed.

Many runs depend on correct channel numbers. If defaults are wrong for a subject, override them when calling the script.

`--one_based` means:
- channel 1 = first channel in the `.acq` file

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
