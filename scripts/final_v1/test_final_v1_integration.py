"""Integration checks for the completed final_v1 one-shot final evaluation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(HERE.parents[1]))).resolve()
REPORTS = ROOT / "reports"
SUPPORT = REPORTS / "final_v1"
MODELS = ROOT / "models" / "final_v1"
sys.path.insert(0, str(HERE))

from analyze_final_v1_gate3 import bootstrap_ci  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def newline_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            count += block.count(b"\n")
    return count


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def main() -> None:
    required = [
        REPORTS / "final_v1_cohort_manifest.csv",
        REPORTS / "final_v1_temporal_manifest.csv",
        REPORTS / "final_v1_protocol_manifest.md",
        REPORTS / "final_v1_structure_validation.md",
        REPORTS / "final_v1_raw_predictions.csv",
        REPORTS / "final_v1_metrics_by_building_seed_budget.csv",
        REPORTS / "final_v1_data_efficiency.csv",
        REPORTS / "final_v1_failure_events.csv",
        REPORTS / "final_v1_failure_summary.csv",
        REPORTS / "final_v1_hypothesis_results.md",
        REPORTS / "final_v1_building_characteristics_exploratory.md",
        REPORTS / "final_v1_final_test_report.md",
        REPORTS / "final_v1_software_addendum_001.md",
    ]
    check("required pre-manifest artifacts exist", all(path.exists() and path.stat().st_size > 0 for path in required))

    cohort = pd.read_csv(REPORTS / "final_v1_cohort_manifest.csv")
    selected = cohort.loc[cohort.selection_status.eq("selected_for_final_v1")]
    excluded = cohort.loc[cohort.selection_status.ne("selected_for_final_v1")]
    check("Gate0 counts", len(cohort) == 510 and len(selected) == 453 and selected.site_id.nunique() == 7)
    check("exact Gate0 exclusions", len(excluded) == 57 and excluded.exclusion_reason.eq("continuous_zero_run_ge_30d").all())
    check("selection independent of performance", ~cohort.selection_used_model_performance.astype(bool).any())

    temporal = pd.read_csv(REPORTS / "final_v1_temporal_manifest.csv")
    check("Gate1 grid and status", len(temporal) == 453 * 6 and temporal.structure_status.eq("PASS").all())
    check("Gate1 overlap zero", temporal.adaptation_test_target_overlap_count.eq(0).all())
    check("Gate1 test shape", temporal.daily_test_sample_count.eq(28).all() and temporal.test_target_hour_count.eq(672).all())
    check("Gate1 full baseline availability", temporal.previous_day_available_target_count.eq(672).all() and temporal.weekly_same_hour_available_target_count.eq(672).all())

    gate2 = json.loads((SUPPORT / "gate2_run_config.json").read_text(encoding="utf-8"))
    predictions = REPORTS / "final_v1_raw_predictions.csv"
    check("Gate2 complete and zero failures", gate2["status"] == "complete" and gate2["failure_count"] == 0)
    check("raw prediction row count", newline_count(predictions) - 1 == 6_088_320)
    check("raw prediction frozen hash", sha256(predictions) == gate2["outputs"][str(predictions)])
    check("frozen seeds and budgets", tuple(gate2["seeds"]) == (20260924, 20260925, 20260926) and tuple(gate2["budgets"]) == (0, 1, 3, 7, 14, 28))
    check("source normalization frozen", gate2["source_normalization"] == {"mean": 167.0651976323119, "std_population": 354.328669571624})

    metrics = pd.read_csv(REPORTS / "final_v1_metrics_by_building_seed_budget.csv")
    check("metric group count", len(metrics) == 9060 and metrics.n_values.eq(672).all())
    check("common truth per building", metrics.groupby("building_id").truth_timestamp_digest.nunique().eq(1).all())
    comparison = pd.read_csv(SUPPORT / "gate3_metric_recalculation_comparison.csv")
    maximum = max(comparison.mae_abs_difference.max(), comparison.rmse_abs_difference.max())
    check("independent metric tolerance", maximum <= 1e-10)

    efficiency = pd.read_csv(REPORTS / "final_v1_data_efficiency.csv")
    check("complete neural grid", len(efficiency) == 453 * 3 * 6 and set(efficiency.adaptation_days) == {0, 1, 3, 7, 14, 28})
    check("three seeds per building budget", efficiency.groupby(["building_id", "adaptation_days"]).seed.nunique().eq(3).all())
    check("0d is source-only", efficiency.loc[efficiency.adaptation_days.eq(0), "adaptation_samples"].eq(0).all() and efficiency.loc[efficiency.adaptation_days.eq(0), "delta_mae_source"].abs().le(1e-12).all())

    events = pd.read_csv(REPORTS / "final_v1_failure_events.csv")
    summary = pd.read_csv(REPORTS / "final_v1_failure_summary.csv")
    check("failure event count", len(events) == 453 * 3 * 5)
    check("F1-F3 definitions", (events.f1 == events.delta_mae_source.gt(1e-12)).all() and (events.f2 == events.delta_rmse_source.gt(1e-12)).all() and (events.f3_any == (events.f3_previous_day | events.f3_weekly_same_hour)).all())
    check("F4 is building-budget event", events.groupby(["building_id", "adaptation_days"]).f4.nunique().eq(1).all())
    total = summary.loc[summary.adaptation_days.eq("ALL_NONZERO")].iloc[0]
    check("failure totals consistent", int(total.f1_count) == int(events.f1.sum()) and int(total.f2_count) == int(events.f2.sum()) and int(total.f4_count) == int(events.drop_duplicates(["building_id", "adaptation_days"]).f4.sum()))

    h1 = pd.read_csv(SUPPORT / "final_v1_h1_site_effects.csv")
    effect = float(h1.delta_mae_source.mean())
    low, high = bootstrap_ci(h1.set_index("site_id").delta_mae_source)
    hypothesis_text = (REPORTS / "final_v1_hypothesis_results.md").read_text(encoding="utf-8")
    check("H1 numerical result", np.isclose(effect, -16.482888133, atol=1e-9) and np.isclose(low, -30.907569879, atol=1e-9) and np.isclose(high, -6.56297796855, atol=1e-9))
    check("H1 supported rule", effect < 0 and high < 0 and "`supported`" in hypothesis_text)

    characteristics = pd.read_csv(SUPPORT / "final_v1_building_characteristics.csv")
    check("characteristic grid", len(characteristics) == 453 * 6)
    check("characteristics pretest only", ~characteristics.test_target_values_used_for_characteristics.astype(bool).any() and (pd.to_datetime(characteristics.feature_information_cutoff) < pd.to_datetime(characteristics.test_target_start)).all())
    check("0d long-history features absent", characteristics.loc[characteristics.adaptation_days.eq(0), "available_pretest_hours"].eq(0).all() and characteristics.loc[characteristics.adaptation_days.eq(0), "available_mean"].isna().all())
    association = pd.read_csv(SUPPORT / "final_v1_characteristic_associations.csv")
    retained = association.loc[(association.response == "seed_mean_mae_delta") & (association.feature == "context_24h_absolute_mean_gap_in_source_config_std")]
    check("missing usage retained after addendum", len(retained) == 5 and retained.n_complete.eq(453).all())
    groups = pd.read_csv(SUPPORT / "final_v1_characteristic_group_summary.csv")
    check("explicit missing usage category", "<missing>" in set(groups.group_value.astype(str)))

    bundles = sorted(MODELS.glob("final_v1_head_only_seed_*d.pt"))
    check("15 adaptation checkpoint bundles", len(bundles) == 15)
    for bundle in bundles:
        payload = torch.load(bundle, map_location="cpu", weights_only=True)
        if len(payload["building_head_state_dict"]) != 453:
            raise AssertionError(f"Incomplete checkpoint bundle: {bundle}")
    check("all checkpoint bundles contain 453 heads", True)

    figures = sorted((SUPPORT / "figures").glob("*.png"))
    check("required graphs created", len(figures) >= 6)
    for figure in figures:
        with Image.open(figure) as image:
            image.verify()
    check("graphs open successfully", True)
    check("software addendum preserves first artifacts", (SUPPORT / "addendum_pre_characteristic_fix").exists() and (ROOT / "scripts" / "final_v1" / "addendum_pre_characteristic_fix").exists())
    print("SUMMARY 35/35 PASS")


if __name__ == "__main__":
    main()
