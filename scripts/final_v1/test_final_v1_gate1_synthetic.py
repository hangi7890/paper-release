"""Synthetic tests for the frozen final_v1 Gate 1 temporal rule."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_final_v1_gate1 import (  # noqa: E402
    frozen_boundaries,
    longest_complete_run,
    validate_building_structure,
)


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def main() -> None:
    # A clean 100-day hourly series must produce the frozen six-budget design.
    index = pd.date_range("2020-01-01", periods=100 * 24, freq="h")
    series = pd.Series(np.arange(len(index), dtype=float), index=index)
    rows, summary, reasons = validate_building_structure("SyntheticClean", series)
    check("clean series passes", summary["structure_status"] == "PASS" and not reasons)
    check("six budgets emitted", [r["adaptation_days"] for r in rows] == [0, 1, 3, 7, 14, 28])
    check(
        "expected adaptation sample counts",
        [r["adaptation_train_sample_count"] for r in rows] == [0, 1, 49, 145, 313, 649],
    )
    check("28 daily windows", all(r["daily_test_sample_count"] == 28 for r in rows))
    check("672 shared target hours", all(r["test_target_hour_count"] == 672 for r in rows))
    check("zero adaptation/test overlap", all(r["adaptation_test_target_overlap_count"] == 0 for r in rows))
    check("X/y continuity", all(r["all_test_windows_X24_y24_contiguous"] for r in rows))

    bounds = frozen_boundaries(index[-1])
    check(
        "frozen range is 57 days",
        len(pd.date_range(bounds["common_start"], bounds["test_end"], freq="h")) == 57 * 24,
    )
    check(
        "test starts after 24h history and 28d adaptation",
        bounds["test_start"] == bounds["common_start"] + pd.Timedelta(hours=24) + pd.Timedelta(days=28),
    )

    # Missing data must split runs. With only 80 days total, both parts are below 90 days.
    short_index = pd.date_range("2021-01-01", periods=80 * 24, freq="h")
    short_values = np.ones(len(short_index), dtype=float)
    short_values[40 * 24] = np.nan
    short_series = pd.Series(short_values, index=short_index)
    _, short_summary, short_reasons = validate_building_structure("SyntheticShort", short_series)
    check("under-90d run fails", short_summary["structure_status"] == "data_integrity_failure")
    check("under-90d reason recorded", "actual_longest_complete_run_lt_90d" in short_reasons)

    # Equal length runs: exact historical rule retains the earliest one.
    tie_index = pd.date_range("2022-01-01", periods=49, freq="h")
    tie_values = np.ones(49, dtype=float)
    tie_values[24] = np.nan
    tie_start, tie_end, tie_len = longest_complete_run(tie_index, tie_values)
    check("earliest run wins ties", tie_start == tie_index[0] and tie_end == tie_index[23] and tie_len == 24)

    print("SUMMARY 12/12 PASS")


if __name__ == "__main__":
    main()
