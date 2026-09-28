"""Synthetic unit tests for final_v1 Gate 2 windowing and serialization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_final_v1_gate1 import validate_building_structure  # noqa: E402
from run_final_v1_gate2 import (  # noqa: E402
    EXPECTED_SAMPLES,
    adaptation_windows,
    daily_test_windows,
    prediction_frame,
    weekly_prediction,
)


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def main() -> None:
    timestamps = pd.date_range("2020-01-01", periods=100 * 24, freq="h")
    values = np.arange(len(timestamps), dtype=float)
    raw = pd.DataFrame({"timestamp": timestamps, "Synthetic": values})
    series = pd.Series(values, index=timestamps)
    rows, summary, reasons = validate_building_structure("Synthetic", series)
    check("Gate1 synthetic source passes", summary["structure_status"] == "PASS" and not reasons)
    temporal = pd.DataFrame(rows)

    test_x, test_y, meta = daily_test_windows(raw, temporal, "Synthetic")
    check("daily X/y shapes", test_x.shape == (28, 24) and test_y.shape == (28, 24))
    check("previous-day alignment", np.array_equal(test_x[:, 0] + 24, test_y[:, 0]))
    weekly = weekly_prediction(raw, "Synthetic", meta)
    check("weekly shape", weekly.shape == (28, 24))
    check("weekly alignment", np.array_equal(weekly + 168, test_y))

    for days in (1, 3, 7, 14, 28):
        x, y, adapt_meta = adaptation_windows(raw, temporal, "Synthetic", days)
        check(
            f"{days}d adaptation shape",
            x.shape == (EXPECTED_SAMPLES[days], 24)
            and y.shape == (EXPECTED_SAMPLES[days], 24)
            and len(adapt_meta) == EXPECTED_SAMPLES[days],
        )

    frame = prediction_frame(
        meta, test_y, test_x, "SyntheticSite", "previous_day", None, None
    )
    check("prediction serialization rows", len(frame) == 672)
    check("prediction target uniqueness", frame.target_timestamp.nunique() == 672)
    check("forecast origins", frame.forecast_origin.nunique() == 28)
    check("horizons", set(frame.horizon_hour) == set(range(1, 25)))
    check("all predictions available", frame.prediction_available.all())
    print("SUMMARY 15/15 PASS")


if __name__ == "__main__":
    main()
