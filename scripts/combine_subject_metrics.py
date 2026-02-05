#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.io import loadmat, savemat


def _to_py(obj: Any) -> Any:
    """Convert numpy/matlab-ish objects into JSON-serializable Python types."""
    # matlab structs sometimes come as objects with __dict__ (mat_struct), but scipy may already simplify
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [_to_py(x) for x in obj.tolist()]
        # numeric array
        if obj.ndim == 0:
            return _to_py(obj.item())
        return obj.tolist()

    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return str(obj)

    if isinstance(obj, str):
        return obj

    if isinstance(obj, dict):
        return {str(k): _to_py(v) for k, v in obj.items()}

    # scipy can return mat_struct-like objects
    if hasattr(obj, "_fieldnames"):
        out = {}
        for f in obj._fieldnames:
            out[f] = _to_py(getattr(obj, f))
        return out

    # fallback
    return obj


def load_metrics_mat(mat_path: Path) -> Dict[str, Any]:
    """
    Load .mat metrics file and return a clean dict (no __header__/__globals__/__version__).
    """
    data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    data = {k: v for k, v in data.items() if not k.startswith("__")}
    return _to_py(data)


def main():
    ap = argparse.ArgumentParser(description="Combine task-level metrics into one subject bundle.")
    ap.add_argument("--out_root", default="derived", help="Root output folder (e.g., derived)")
    ap.add_argument("--sub", required=True, help="Subject code like 2062 (will be sub-2062)")
    ap.add_argument("--ses", default="1", help="Session number like 1/2 (will be ses-1)")
    ap.add_argument("--pattern", default="*metrics.mat", help="Glob pattern for metrics files (default *metrics.mat)")
    ap.add_argument("--also_json", action="store_true", help="Also write a JSON bundle next to the MAT.")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    sub_id = f"sub-{args.sub}"
    ses_id = f"ses-{args.ses}"
    base = out_root / sub_id / ses_id

    if not base.exists():
        raise FileNotFoundError(f"Cannot find session folder: {base}")

    # find all metrics .mat under derived/sub-XXXX/ses-Y/**
    mats = sorted(base.rglob(args.pattern))
    if not mats:
        raise FileNotFoundError(f"No metrics files found under {base} with pattern {args.pattern}")

    tasks: Dict[str, Any] = {}
    for mp in mats:
        task_name = mp.parent.name  # e.g., rest, sts, valsalva, breathing
        try:
            tasks[task_name] = load_metrics_mat(mp)
            print(f"[OK] Loaded: {mp}  -> task='{task_name}'")
        except Exception as e:
            print(f"[WARN] Failed loading {mp}: {e}")

    bundle = {
        "sub_id": sub_id,
        "ses_id": ses_id,
        "metrics_by_task": tasks,
    }

    out_mat = base / f"{sub_id}_{ses_id}_all_metrics.mat"
    savemat(str(out_mat), bundle, do_compression=True)
    print(f"\n[OK] Wrote combined MAT: {out_mat}")

    if args.also_json:
        out_json = base / f"{sub_id}_{ses_id}_all_metrics.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
        print(f"[OK] Wrote combined JSON: {out_json}")


if __name__ == "__main__":
    main()
