"""Compare standalone inference output with the frozen E1 reference output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--scale", choices=("normalized", "physical"), required=True)
    args = parser.parse_args()

    reference = np.load(args.reference, allow_pickle=False)
    candidate = np.load(args.candidate, allow_pickle=False)
    shape_match = reference.shape == candidate.shape
    if not shape_match:
        raise ValueError(f"Shape mismatch: {reference.shape} != {candidate.shape}")
    reference_nan = int(np.isnan(reference).sum())
    candidate_nan = int(np.isnan(candidate).sum())
    reference_inf = int(np.isinf(reference).sum())
    candidate_inf = int(np.isinf(candidate).sum())
    if reference_nan + candidate_nan + reference_inf + candidate_inf:
        raise ValueError("NaN or Inf detected")

    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    report = {
        "status": "PASS",
        "scale": args.scale,
        "shape_match": shape_match,
        "shape": list(reference.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "dtype_match": reference.dtype == candidate.dtype,
        "reference_nan_count": reference_nan,
        "candidate_nan_count": candidate_nan,
        "reference_inf_count": reference_inf,
        "candidate_inf_count": candidate_inf,
        "max_absolute_difference": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "mean_absolute_difference": float(np.mean(np.abs(difference))) if difference.size else 0.0,
        "rmse_difference": float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0,
        "tolerance_enforced": False,
        "note": "E1 records host self-differences; Jetson acceptance tolerance is not fixed here.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

