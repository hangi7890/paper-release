#!/usr/bin/env python3
"""Independently compare saved Jetson outputs with frozen RTX references."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
BUNDLE = (
    PROJECT
    / "edge_deployment_final"
    / "e2_transfer_freeze"
    / "edge_deployment_jetson_fp32"
)

CRITERIA = {
    "shape_exact": True,
    "nan_count": 0,
    "inf_count": 0,
    "normalized": {
        "max_absolute_difference_lte": 1e-4,
        "mean_absolute_difference_lte": 2e-5,
    },
    "physical": {
        "max_absolute_difference_lte": 0.05,
        "mean_absolute_difference_lte": 0.01,
    },
}


def load(name: str) -> np.ndarray:
    return np.load(name, allow_pickle=False)


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    shape_match = reference.shape == candidate.shape
    if not shape_match:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(difference)
    max_flat = int(np.argmax(absolute)) if absolute.size else 0
    max_index = tuple(int(value) for value in np.unravel_index(max_flat, absolute.shape))
    return {
        "shape_match": True,
        "shape": list(reference.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "reference_nan_count": int(np.isnan(reference).sum()),
        "candidate_nan_count": int(np.isnan(candidate).sum()),
        "reference_inf_count": int(np.isinf(reference).sum()),
        "candidate_inf_count": int(np.isinf(candidate).sum()),
        "max_absolute_difference": float(absolute[max_index]) if absolute.size else 0.0,
        "mean_absolute_difference": float(np.mean(absolute)) if absolute.size else 0.0,
        "rmse_difference": float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0,
        "max_error_index": list(max_index),
        "max_error_reference": float(reference[max_index]) if reference.size else None,
        "max_error_jetson": float(candidate[max_index]) if candidate.size else None,
        "max_error_signed_difference": float(difference[max_index]) if difference.size else None,
    }


def main() -> None:
    benchmark_ref_n = load(BUNDLE / "validation" / "benchmark_reference_output_normalized.npy")
    benchmark_ref_p = load(BUNDLE / "validation" / "benchmark_reference_output_physical.npy")
    benchmark_jetson_n = load(HERE / "jetson_benchmark_output_normalized.npy")
    benchmark_jetson_p = load(HERE / "jetson_benchmark_output_physical.npy")

    benchmark_n = metrics(benchmark_ref_n, benchmark_jetson_n)
    benchmark_p = metrics(benchmark_ref_p, benchmark_jetson_p)
    if not benchmark_n.get("shape_match") or benchmark_ref_n.shape != (1, 24):
        raise RuntimeError("Unexpected normalized benchmark shape")
    if not benchmark_p.get("shape_match") or benchmark_ref_p.shape != (1, 24):
        raise RuntimeError("Unexpected physical benchmark shape")

    comparison_path = HERE / "benchmark_vector_comparison.csv"
    if comparison_path.exists():
        raise FileExistsError(comparison_path)
    with comparison_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "vector_id",
                "horizon_index",
                "normalized_reference",
                "normalized_jetson",
                "normalized_abs_diff",
                "physical_reference",
                "physical_jetson",
                "physical_abs_diff",
            ],
        )
        writer.writeheader()
        for horizon in range(24):
            writer.writerow(
                {
                    "vector_id": 0,
                    "horizon_index": horizon,
                    "normalized_reference": repr(float(benchmark_ref_n[0, horizon])),
                    "normalized_jetson": repr(float(benchmark_jetson_n[0, horizon])),
                    "normalized_abs_diff": repr(
                        abs(float(benchmark_jetson_n[0, horizon]) - float(benchmark_ref_n[0, horizon]))
                    ),
                    "physical_reference": repr(float(benchmark_ref_p[0, horizon])),
                    "physical_jetson": repr(float(benchmark_jetson_p[0, horizon])),
                    "physical_abs_diff": repr(
                        abs(float(benchmark_jetson_p[0, horizon]) - float(benchmark_ref_p[0, horizon]))
                    ),
                }
            )

    reference_n = load(BUNDLE / "validation" / "validation_reference_output_normalized.npy")
    reference_p = load(BUNDLE / "validation" / "validation_reference_output_physical.npy")
    jetson_n = load(HERE / "jetson_validation_output_normalized.npy")
    jetson_p = load(HERE / "jetson_validation_output_physical.npy")
    normalized = metrics(reference_n, jetson_n)
    physical = metrics(reference_p, jetson_p)

    if reference_n.shape != (64, 24) or reference_p.shape != (64, 24):
        raise RuntimeError("Frozen reference shape changed")
    if not normalized.get("shape_match") or not physical.get("shape_match"):
        raise RuntimeError("Jetson output shape mismatch")

    finite_ok = (
        normalized["reference_nan_count"] == 0
        and normalized["candidate_nan_count"] == 0
        and normalized["reference_inf_count"] == 0
        and normalized["candidate_inf_count"] == 0
        and physical["reference_nan_count"] == 0
        and physical["candidate_nan_count"] == 0
        and physical["reference_inf_count"] == 0
        and physical["candidate_inf_count"] == 0
    )
    checks = {
        "shape_exact": normalized["shape_match"] and physical["shape_match"],
        "nan_inf_zero": finite_ok,
        "normalized_max": normalized["max_absolute_difference"]
        <= CRITERIA["normalized"]["max_absolute_difference_lte"],
        "normalized_mean": normalized["mean_absolute_difference"]
        <= CRITERIA["normalized"]["mean_absolute_difference_lte"],
        "physical_max": physical["max_absolute_difference"]
        <= CRITERIA["physical"]["max_absolute_difference_lte"],
        "physical_mean": physical["mean_absolute_difference"]
        <= CRITERIA["physical"]["mean_absolute_difference_lte"],
    }
    summary = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "independent_local_recalculation": True,
        "output_elements": int(reference_n.size),
        "criteria": CRITERIA,
        "checks": checks,
        "benchmark_vector": {
            "vector_id": 0,
            "normalized": benchmark_n,
            "physical": benchmark_p,
        },
        "validation_64": {
            "normalized": normalized,
            "physical": physical,
        },
        "performance_benchmark_performed": False,
    }
    summary_path = HERE / "validation_64_comparison_summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    maximum = {
        "index_convention": "zero-based [vector_id, horizon_index]",
        "normalized": {
            "vector_id": normalized["max_error_index"][0],
            "horizon_index": normalized["max_error_index"][1],
            "reference": normalized["max_error_reference"],
            "jetson": normalized["max_error_jetson"],
            "absolute_difference": normalized["max_absolute_difference"],
            "signed_difference": normalized["max_error_signed_difference"],
        },
        "physical": {
            "vector_id": physical["max_error_index"][0],
            "horizon_index": physical["max_error_index"][1],
            "reference": physical["max_error_reference"],
            "jetson": physical["max_error_jetson"],
            "absolute_difference": physical["max_absolute_difference"],
            "signed_difference": physical["max_error_signed_difference"],
        },
    }
    maximum_path = HERE / "max_error_case.json"
    if maximum_path.exists():
        raise FileExistsError(maximum_path)
    maximum_path.write_text(json.dumps(maximum, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
