# LCS Physio Report Metrics Generation

This project generates per-task physiological metrics (and key figures) from Biopac `.acq` recordings, and optionally merges them into a single subject/session bundle for downstream MATLAB/Word report generation. A Streamlit UI is provided to run tasks interactively and preview outputs.

---

## What this project does

For a given subject/session (e.g., `sub-2062/ses-1`), the pipeline can produce:

* REST: baseline HR/HRV and BP summary metrics + REST figures
* STS: supine vs standing HR/MAP plateau metrics + STS figure
* Valsalva: Valsalva ratio + best repetition HR figure (+ optional debug figure)
* Breathing: deep breathing response metrics + breathing HR figure
* Spirometry: FEV1/FVC/PEF extracted metrics (if available)

Then it can MERGE all task `.mat` files into a single:

* `derived/sub-XXXX/ses-Y/sub-XXXX_ses-Y_all_metrics.mat`

This merged bundle contains:

* `metrics_by_task`: nested task metrics (rest/sts/valsalva/breathing/spirometry)
* `whole`: flattened fields used by your MATLAB report code
* `figures`: paths to key figures for report insertion

---

## Repository layout

Typical structure:

project_root/
streamlit_app/
app.py
scripts/
run_rest_acq.py
run_sts_acq.py
run_valsalva_acq.py
run_breathing_acq.py
run_spirometry_extract.py
merge_subject_all_metrics_only.py
src/
bp_processing.py
derived/
sub-2062/
ses-1/
rest/
rest_metrics.mat
resting_hr.png
resting_BP.png
sts/
sts_metrics.mat
STS_HR_MAP.png
valsalva/
valsalva_metrics.mat
valsalva_best_rep_hr.png
valsalva_debug_full_hr.png
breathing/
breathing_metrics.mat
breathing_hr_8to9min.png
spirometry/
spirometry_metrics.mat
sub-2062_ses-1_all_metrics.mat

Note: output filenames may differ slightly depending on script versions. The Streamlit app expects the defaults above.

---

## Input data layout

The scripts expect a BIDS-like folder structure under a physio root (default shown):

/export02/projects/LCS/01_physio/
sub-2062/
ses-1/
sub-2062_ses-1_task-rest_physio.acq
sub-2062_ses-1_task-sts_physio.acq
sub-2062_ses-1_task-valsalva_physio.acq
sub-2062_ses-1_task-breathing_physio.acq
...

Most scripts have fallback glob searches if names differ.

---

## Installation

Create a virtual environment and install dependencies:

python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install streamlit numpy scipy pandas matplotlib bioread neurokit2

If your repo uses additional packages in `src/`, install those as needed.

---

## Run the Streamlit UI

From project root:

streamlit run streamlit_app/app.py

In the UI:

1. Set Physio root, Output root, Subject, and Session
2. Set channel numbers (ECG/BP/PPG) as needed
3. Run tasks (REST / STS / Valsalva / Breathing / Spirometry)
4. Preview figures + `.mat` metrics per task
5. Click merge_tasks_results to build the combined `*_all_metrics.mat`

---

## Run scripts directly (CLI)

REST:
python scripts/run_rest_acq.py --root /export02/projects/LCS/01_physio --sub 2062 --ses 1 --one_based --ecg_ch 6 --bp_ch 10 --save --out_root derived

Outputs:

* derived/sub-2062/ses-1/rest/rest_metrics.mat
* derived/sub-2062/ses-1/rest/resting_hr.png
* derived/sub-2062/ses-1/rest/resting_BP.png

STS:
python scripts/run_sts_acq.py --root /export02/projects/LCS/01_physio --sub 2062 --ses 1 --one_based --ecg_ch 4 --bp_ch 10 --save --out_root derived

Valsalva:
python scripts/run_valsalva_acq.py --root /export02/projects/LCS/01_physio --sub 2062 --ses 1 --one_based --ecg_ch 4 --save --out_root derived

Optional PPG:
python scripts/run_valsalva_acq.py ... --force_ppg --ppg_ch 5

Breathing:
python scripts/run_breathing_acq.py --root /export02/projects/LCS/01_physio --sub 2062 --ses 1 --one_based --ecg_ch 4 --win_start_min 8 --win_end_min 9 --save --out_root derived

Spirometry:
python scripts/run_spirometry_extract.py --sub 2062 --ses 1 --out_root derived

---

## Merge all task outputs (no re-processing)

After running tasks, generate the combined bundle:

python scripts/merge_subject_all_metrics_only.py --out_root derived --sub 2062 --ses 1

Optional: auto-create a Valsalva placeholder when it is missing AND no `.acq` exists:

python scripts/merge_subject_all_metrics_only.py --out_root derived --sub 2062 --ses 1 --root /export02/projects/LCS/01_physio --make_valsalva_placeholder_if_missing

Output:

* derived/sub-2062/ses-1/sub-2062_ses-1_all_metrics.mat

---

## Notes on channels (1-based vs 0-based)

Most Biopac channel selections in the UI/scripts use 1-based indexing by default:

* Channel 1 = first channel in the `.acq` file

If you want 0-based, disable the checkbox in Streamlit or remove `--one_based`.

---

## Troubleshooting

1. “missing ScriptRunContext” warnings:
   This happens when running a Streamlit script with python. Always run:
   streamlit run streamlit_app/app.py

2. “unrecognized arguments”:
   Only pass flags that the underlying script supports. Keep Streamlit debug flags task-specific.

3. `.mat` metrics not displaying:
   The app uses scipy.io.loadmat. Ensure SciPy is installed:
   pip install scipy

4. Output file not found after merge:
   Check that the merge script writes the same filename that the app expects (paths["all_mat"]) and that out_root/sub/ses match.

---

## Extending

* Add new task scripts under scripts/
* Make them write: {task}/{task}_metrics.mat and any figures you want to preview
* Update merge_subject_all_metrics_only.py to include the new task in metrics_by_task / whole mapping
* Update streamlit_app/app.py to add a task section (button + preview)

---

## License

Internal research code. Add a license if you intend to distribute publicly.
