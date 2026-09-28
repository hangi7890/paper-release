"""Synthetic checks for frozen final_v1 Gate 3 statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze_final_v1_gate3 import (  # noqa: E402
    BUDGETS,
    SEEDS,
    bootstrap_ci,
    build_curves,
    build_efficiency,
    build_failures,
    build_hypotheses,
    lag_repeatability,
    numeric_summary,
)


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def synthetic_metrics():
    rows = []
    cohort_rows = []
    for bindex, (building, site) in enumerate((("B1", "S1"), ("B2", "S2"))):
        cohort_rows.append({"building_id": building, "site_id": site, "primaryspaceusage": "office"})
        rows.extend(
            [
                {"building_id": building, "site_id": site, "model": "previous_day", "seed_key": -1, "days_key": -1, "mae": 8.0, "rmse": 10.0},
                {"building_id": building, "site_id": site, "model": "weekly_same_hour", "seed_key": -1, "days_key": -1, "mae": 7.0, "rmse": 9.0},
            ]
        )
        for seed in SEEDS:
            for days in BUDGETS:
                source = 10.0 + bindex
                delta = 0.0 if days == 0 else (1.0 if days == 1 else -days / 28.0)
                rows.append(
                    {
                        "building_id": building,
                        "site_id": site,
                        "model": "c0_source_only_lstm" if days == 0 else "c0_head_only_adaptation_lstm",
                        "seed_key": seed,
                        "days_key": days,
                        "mae": source + delta,
                        "rmse": source + 2 + delta / 2,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(cohort_rows)


def main() -> None:
    metrics, cohort = synthetic_metrics()
    efficiency = build_efficiency(metrics, cohort)
    check("efficiency grid", len(efficiency) == 2 * 3 * 6)
    check("0d delta zero", efficiency.loc[efficiency.adaptation_days.eq(0), "delta_mae_source"].eq(0).all())
    check("1d worsens", efficiency.loc[efficiency.adaptation_days.eq(1), "delta_mae_source"].gt(0).all())
    check("28d improves", efficiency.loc[efficiency.adaptation_days.eq(28), "delta_mae_source"].lt(0).all())
    events, summary = build_failures(efficiency)
    check("F1 detected at 1d", events.loc[events.adaptation_days.eq(1), "f1"].all())
    check("F1 absent at 28d", ~events.loc[events.adaptation_days.eq(28), "f1"].any())
    check("failure summary has all budgets", len(summary) == 6)
    hypothesis, h1, h2, h3, contingency = build_hypotheses(efficiency, events)
    check("H1 effect negative", hypothesis["h1_effect"] < 0)
    check("H1 small-site status", hypothesis["h1_status"] == "inconclusive_fewer_than_5_sites")
    check("H2 effect negative", hypothesis["h2_effect"] < 0)
    check("H3 table complete", int(contingency["count"].sum()) == 2 * 3)
    curve, site_curve = build_curves(efficiency)
    check("full six-budget curve", list(curve.adaptation_days) == list(BUDGETS))
    ci1 = bootstrap_ci(pd.Series([-2.0, -1.0, -3.0]))
    ci2 = bootstrap_ci(pd.Series([-2.0, -1.0, -3.0]))
    check("bootstrap deterministic", ci1 == ci2 and ci1[1] < 0)
    stats = numeric_summary(np.array([0.0, 1.0, 2.0, 3.0]))
    check("zero ratio", stats["zero_ratio"] == 0.25)
    n, corr, status = lag_repeatability(np.arange(48, dtype=float), 24, 24)
    check("daily repeatability", n == 24 and np.isclose(corr, 1.0) and status == "ok")
    print("SUMMARY 15/15 PASS")


if __name__ == "__main__":
    main()
