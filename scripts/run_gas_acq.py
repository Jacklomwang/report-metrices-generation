#!/usr/bin/env python3
"""
Gas-manipulation task page inputs.

Renders the four gas-task signals for the participant report, all preprocessed
with physio-qc's DEFAULT parameters (no bespoke tuning):

  Panel 1  CO2 (mmHg)          raw capnogram + highlighted END-TIDAL CO2 trace (peaks)
  Panel 2  O2 (mmHg)           raw O2 + highlighted END-TIDAL O2 trace (troughs)
  Panel 3  SpO2 (%)            pulse-oximetry saturation channel (Session-A physio), cleaned
  Panel 4  Doppler MV (cm/s)   transcranial Doppler + mean-velocity (TCD MV) trace

Reuse map (physio-qc @ $PHYSIOQC_DIR):
  utils.file_io.load_acq_file  -> df, sampling_rate, signal_mappings (channel auto-ID +
                                  Pct/Volts gas conversion to mmHg)
  metrics.etco2.process_etco2  (config.DEFAULT_ETCO2_PARAMS) -> etco2_envelope, auto_peaks
  metrics.eto2.process_eto2    (config.DEFAULT_ETO2_PARAMS)  -> eto2_envelope, auto_troughs
  metrics.spo2.process_spo2    (config.DEFAULT_SPO2_PARAMS)  -> cleaned_signal
  metrics.doppler.process_doppler (config.DEFAULT_BP_PARAMS) -> filtered, map_4hz (mean vel)

Outputs (when --save):
  {out_root}/sub-<ID>/ses-<S>/gas/gas_signals.png
  {out_root}/sub-<ID>/ses-<S>/gas/gas_metrics.mat

Usage: python scripts/run_gas_acq.py --root /export02/projects/LCS/01_physio --sub 1053 --ses 1 --save
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHYSIOQC_DIR = os.environ.get("PHYSIOQC_DIR", "/export02/users/sloparco/physio-qc")
if PHYSIOQC_DIR not in sys.path:
    sys.path.insert(0, PHYSIOQC_DIR)
import config as pqc_config                    # noqa: E402
from utils.file_io import load_acq_file        # noqa: E402
from metrics import etco2, eto2, spo2, doppler  # noqa: E402

# highlight palette (raw traces recede to light grey; end-tidal / mean traces carry colour)
C_RAW = "#b6bcc4"
C_CO2 = "#d9631d"   # warm — CO2 / hypercapnia
C_O2 = "#2f6fb0"    # blue — O2
C_SPO2 = "#2f8f57"  # green — saturation
C_DOP = "#7c3aed"   # violet — Doppler


def build_gas_acq_path(root: Path, sub_code: str, ses_num: str) -> Path:
    sub_id, ses_id = f"sub-{sub_code}", f"ses-{ses_num}"
    cand = [
        root / sub_id / ses_id / f"{sub_id}_{ses_id}_task-gas_physio.acq",
        root / sub_id / ses_id / f"{sub_id}_{ses_id}_task-gas*physio*.acq",
    ]
    for c in cand:
        hits = sorted(c.parent.glob(c.name)) if "*" in c.name else ([c] if c.exists() else [])
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No gas .acq found for {sub_id} {ses_id} under {root}")


def _baseline(x: np.ndarray, t: np.ndarray, sec: float = 60.0) -> float:
    m = np.isfinite(x) & (t <= sec)
    return float(np.nanmedian(x[m])) if m.any() else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="Gas manipulation: CO2/O2 end-tidal, SpO2, Doppler (physio-qc defaults)")
    ap.add_argument("--root", default="/export02/projects/LCS/01_physio")
    ap.add_argument("--sub", required=True)
    ap.add_argument("--ses", default="1")
    ap.add_argument("--save", action="store_true", help="Save gas_signals.png and gas_metrics.mat")
    ap.add_argument("--out_root", default="derived")
    # accepted for orchestrator compatibility (unused here — channels are auto-mapped):
    ap.add_argument("--one_based", action="store_true", help=argparse.SUPPRESS)
    args, _ = ap.parse_known_args()

    sub_id, ses_id = f"sub-{args.sub}", f"ses-{args.ses}"
    acq_path = build_gas_acq_path(Path(args.root), args.sub, args.ses)
    print(f"[INFO] Gas ACQ: {acq_path}")

    d = load_acq_file(str(acq_path))
    df, fs, mp = d["df"], float(d["sampling_rate"]), d["signal_mappings"]
    n = len(df)
    t = np.arange(n) / fs
    print(f"[INFO] fs={fs:.0f} Hz  n={n}  duration={n/fs:.0f}s  channels mapped: {sorted(mp)}")

    def col(key):
        c = mp.get(key)
        return df[c].to_numpy(dtype=float) if (c is not None and c in df.columns) else None

    co2_raw, o2_raw, spo2_raw, dop_raw = col("etco2"), col("eto2"), col("spo2"), col("doppler")

    co2 = etco2.process_etco2(co2_raw, fs, dict(pqc_config.DEFAULT_ETCO2_PARAMS)) if co2_raw is not None else None
    o2 = eto2.process_eto2(o2_raw, fs, dict(pqc_config.DEFAULT_ETO2_PARAMS)) if o2_raw is not None else None
    sat = spo2.process_spo2(spo2_raw, fs, dict(pqc_config.DEFAULT_SPO2_PARAMS)) if spo2_raw is not None else None
    dop = doppler.process_doppler(dop_raw, fs, dict(pqc_config.DEFAULT_BP_PARAMS)) if dop_raw is not None else None

    # ---- gas-paradigm event windows: onset timing FROM physio-qc ------------
    # Resolve the gas variant via physio-qc's bids_summary map, then read its onset
    # CSV (onsets_gas-{short,long}.csv). Windows are the delivered stimulus blocks
    # (Hypercapnia/Hypoxia); the end-tidal response lags ~10 s into each block.
    sub_num = str(args.sub).replace("sub-", "")
    variant = pqc_config.GAS_VARIANT_MAP.get(sub_num)
    variant_source = "bids_summary"
    if variant is None:
        # the long paradigm's last block runs to ~960 s, so a too-short recording
        # must be the short variant.
        variant = "gas-long" if (n / fs) >= 900.0 else "gas-short"
        variant_source = "duration-fallback"
    acq_start = float(d.get("acquisition_start") or 0.0)   # seconds from recording start
    gas_events = pqc_config.load_onset_events(variant)     # (onset, duration, trial_type, color)
    print(f"[INFO] gas variant={variant} ({variant_source}); acq_start={acq_start:.1f}s; "
          f"{len(gas_events)} onset event(s) from physio-qc")

    # ---- 4-panel figure ---------------------------------------------------
    LBL_FS, TICK_FS, LEG_FS, SPAN_FS = 13, 11.5, 10.5, 12
    fig, axes = plt.subplots(4, 1, figsize=(12.5, 11), sharex=True)

    ax = axes[0]
    if co2 is not None:
        ax.plot(t, co2_raw, color=C_RAW, lw=0.6, label="capnogram")
        ax.plot(t, co2["etco2_envelope"], color=C_CO2, lw=2.2, label="end-tidal CO₂")
        pk = np.asarray(co2.get("auto_peaks", []), dtype=int)
        if pk.size:
            ax.plot(t[pk], co2_raw[pk], "o", ms=3.5, color=C_CO2, mec="white", mew=0.5)
    ax.set_ylabel("CO₂ (mmHg)", fontsize=LBL_FS)

    ax = axes[1]
    if o2 is not None:
        ax.plot(t, o2_raw, color=C_RAW, lw=0.6, label="O₂ trace")
        ax.plot(t, o2["eto2_envelope"], color=C_O2, lw=2.2, label="end-tidal O₂")
        tr = np.asarray(o2.get("auto_troughs", []), dtype=int)
        if tr.size:
            ax.plot(t[tr], o2_raw[tr], "o", ms=3.5, color=C_O2, mec="white", mew=0.5)
    ax.set_ylabel("O₂ (mmHg)", fontsize=LBL_FS)

    ax = axes[2]
    if sat is not None:
        ax.plot(t, sat["cleaned_signal"], color=C_SPO2, lw=1.6, label="SpO₂ (pulse-ox)")
    ax.set_ylabel("SpO₂ (%)", fontsize=LBL_FS)

    ax = axes[3]
    if dop is not None:
        ax.plot(t, dop["filtered"], color=C_RAW, lw=0.5, label="Doppler velocity")
        if dop.get("time_4hz") is not None and dop.get("map_4hz") is not None:
            ax.plot(dop["time_4hz"], dop["map_4hz"], color=C_DOP, lw=2.2, label="mean velocity")
    ax.set_ylabel("Doppler MV (cm/s)", fontsize=LBL_FS)
    ax.set_xlabel("Time (s)", fontsize=LBL_FS)

    # shade each gas-paradigm block across every panel (physio-qc onset windows +
    # colours); label once above the top panel
    for a in axes:
        for onset, dur, ttype, color in gas_events:
            a.axvspan(acq_start + onset, acq_start + onset + dur, color=color, alpha=0.16, lw=0, zorder=0)
    for onset, dur, ttype, color in gas_events:
        axes[0].text(acq_start + onset + dur / 2.0, 1.04, ttype, transform=axes[0].get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=SPAN_FS, fontweight="bold", color=color, clip_on=False)

    for a in axes:
        a.grid(True, alpha=0.15)
        a.tick_params(labelsize=TICK_FS)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        # legend OUTSIDE the axes (to the right) so it never overlaps the traces
        if a.get_legend_handles_labels()[0]:
            a.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=LEG_FS,
                     frameon=False, borderaxespad=0.0)
    axes[0].set_xlim(t[0], t[-1])
    # No figure title — the page header already reads "Gas manipulation".
    fig.subplots_adjust(left=0.085, right=0.80, top=0.95, bottom=0.075, hspace=0.32)

    # ---- summary metrics --------------------------------------------------
    def env(x):
        return np.asarray(x, dtype=float) if x is not None else np.array([np.nan])
    co2_env = env(co2["etco2_envelope"]) if co2 is not None else env(None)
    o2_env = env(o2["eto2_envelope"]) if o2 is not None else env(None)
    spo2_c = env(sat["cleaned_signal"]) if sat is not None else env(None)
    dop_mv = env(dop["map_4hz"]) if (dop is not None and dop.get("map_4hz") is not None) else env(None)
    dop_mv_t = env(dop["time_4hz"]) if (dop is not None and dop.get("time_4hz") is not None) else t

    def _win_median(x, tx, lo, hi):
        mm = np.isfinite(x) & (tx >= lo) & (tx <= hi)
        return float(np.nanmedian(x[mm])) if mm.any() else float("nan")

    # Cerebrovascular reactivity (CVR): %ΔCBF-velocity per mmHg ΔET-CO2 during the
    # hypercapnia block (steady-state plateau vs pre-block baseline).
    cvr = float("nan")
    hc = [ev for ev in gas_events if ev[2] == "Hypercapnia"]
    if hc and co2 is not None and dop is not None:
        on, dur = acq_start + hc[0][0], hc[0][1]
        etco2_b = _win_median(co2_env, t, max(0.0, on - 40), on - 5)
        etco2_h = _win_median(co2_env, t, on + dur - 45, on + dur - 5)
        mv_b = _win_median(dop_mv, dop_mv_t, max(0.0, on - 40), on - 5)
        mv_h = _win_median(dop_mv, dop_mv_t, on + dur - 45, on + dur - 5)
        dco2 = etco2_h - etco2_b
        if np.isfinite(dco2) and abs(dco2) > 1.0 and np.isfinite(mv_b) and mv_b > 0:
            cvr = ((mv_h - mv_b) / mv_b * 100.0) / dco2   # %/mmHg

    # Minimum SpO2 during hypoxia (nadir often lags the block, so include ~90 s after).
    spo2_min_hypoxia = float("nan")
    hx = [ev for ev in gas_events if ev[2] == "Hypoxia"]
    if hx and sat is not None:
        on, dur = acq_start + hx[0][0], hx[0][1]
        mm = np.isfinite(spo2_c) & (t >= on) & (t <= on + dur + 90)
        if mm.any():
            spo2_min_hypoxia = float(np.nanmin(spo2_c[mm]))

    metrics = {
        "present": 1,
        "duration_s": float(n / fs),
        "sampling_rate": float(fs),
        "etco2_baseline_mmHg": _baseline(co2_env, t),
        "etco2_max_mmHg": float(np.nanmax(co2_env)),
        "eto2_baseline_mmHg": _baseline(o2_env, t),
        "eto2_min_mmHg": float(np.nanmin(o2_env)),
        "spo2_baseline_pct": _baseline(spo2_c, t),
        "spo2_min_pct": float(np.nanmin(spo2_c)),
        "doppler_mv_baseline_cms": _baseline(dop_mv, dop_mv_t),
        "doppler_mv_max_cms": float(np.nanmax(dop_mv)),
        "cvr_pct_per_mmHg": cvr,
        "spo2_min_hypoxia_pct": spo2_min_hypoxia,
        "gas_variant": variant,
        "gas_variant_source": variant_source,
        "note": "SpO2 measured from the finger pulse-oximeter; CVR = %ΔDoppler velocity per mmHg "
                "ΔET-CO2 during hypercapnia; minimum SpO2 taken during the hypoxia block.",
    }
    print("\n===== GAS RESULTS =====")
    for k in ("etco2_max_mmHg", "eto2_min_mmHg", "spo2_min_hypoxia_pct", "cvr_pct_per_mmHg"):
        print(f"  {k:24s} = {metrics[k]:.3f}")

    if not args.save:
        print("\n[INFO] --save not set: not writing outputs.")
        plt.close(fig)
        return 0

    task_out = Path(args.out_root) / sub_id / ses_id / "gas"
    task_out.mkdir(parents=True, exist_ok=True)
    fig_path = task_out / "gas_signals.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"[OK] Saved gas figure: {fig_path}")

    from scipy.io import savemat
    savemat(str(task_out / "gas_metrics.mat"), metrics, do_compression=True)
    print(f"[OK] Saved gas metrics bundle: {task_out / 'gas_metrics.mat'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
