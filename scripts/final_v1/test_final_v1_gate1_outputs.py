"""Post-run checks for immutable Gate 0 and Gate 1 artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
REPORTS = ROOT / "reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def main() -> None:
    cohort = pd.read_csv(REPORTS / "final_v1_cohort_manifest.csv")
    temporal = pd.read_csv(REPORTS / "final_v1_temporal_manifest.csv")
    failures = pd.read_csv(REPORTS / "final_v1" / "gate1_data_integrity_failures.csv")
    selected = cohort.loc[cohort.selection_status.eq("selected_for_final_v1")]
    check("Gate0 cohort frozen hash", sha256(REPORTS / "final_v1_cohort_manifest.csv") == "8216847545f7814c12f3e2edb31523ac11da0ae19cc142985f736bbd7185d3ec")
    check("Gate1 temporal frozen hash", sha256(REPORTS / "final_v1_temporal_manifest.csv") == "1cb76d4cc7cef687fc747b2e0015112d85aec4e0f9e76271b28284f54421bc08")
    check("453 selected buildings", len(selected) == 453 and selected.building_id.nunique() == 453)
    check("7 selected sites", selected.site_id.nunique() == 7)
    check("six rows per building", len(temporal) == 453 * 6 and temporal.building_id.nunique() == 453)
    check("all structures pass", temporal.structure_status.eq("PASS").all())
    check("all test windows fixed", temporal.daily_test_sample_count.eq(28).all() and temporal.test_target_hour_count.eq(672).all())
    check("expected adaptation budgets", set(temporal.adaptation_days) == {0, 1, 3, 7, 14, 28})
    expected = {0: 0, 1: 1, 3: 49, 7: 145, 14: 313, 28: 649}
    check("expected adaptation samples", all(frame.adaptation_train_sample_count.eq(expected[int(days)]).all() for days, frame in temporal.groupby("adaptation_days")))
    check("zero target overlap", temporal.adaptation_test_target_overlap_count.eq(0).all())
    check("baselines fully available", temporal.previous_day_available_target_count.eq(672).all() and temporal.weekly_same_hour_available_target_count.eq(672).all())
    check("no data-integrity failures", failures.empty)
    check("Gate0/Gate1 IDs identical", set(selected.building_id) == set(temporal.building_id))
    print("SUMMARY 13/13 PASS")


if __name__ == "__main__":
    main()
