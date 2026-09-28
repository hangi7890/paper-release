"""Gate 0 tests that do not read final-test power values."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
REPORTS = ROOT / "reports"
COHORT = REPORTS / "final_v1_cohort_manifest.csv"
PROTOCOL = REPORTS / "final_v1_protocol_manifest.md"
HASHES = REPORTS / "final_v1" / "gate0_frozen_hashes.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FinalV1Gate0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cohort = pd.read_csv(COHORT)
        cls.hashes = pd.read_csv(HASHES)

    def test_exact_proposal_and_selected_counts(self) -> None:
        self.assertEqual(len(self.cohort), 510)
        self.assertEqual(self.cohort.building_id.nunique(), 510)
        self.assertEqual(self.cohort.site_id.nunique(), 9)
        selected = self.cohort.selection_status.eq("selected_for_final_v1")
        self.assertEqual(int(selected.sum()), 453)
        self.assertEqual(self.cohort.loc[selected, "site_id"].nunique(), 7)
        self.assertEqual(int((~selected).sum()), 57)

    def test_only_final_proposal_sites(self) -> None:
        self.assertTrue(self.cohort.alternative_group.eq("final_test_proposal").all())
        self.assertEqual(
            set(self.cohort.site_id),
            {"Bobcat", "Bull", "Eagle", "Gator", "Hog", "Moose", "Mouse", "Shrew", "Swan"},
        )

    def test_frozen_quality_rule(self) -> None:
        selected = self.cohort.loc[self.cohort.selection_status.eq("selected_for_final_v1")]
        self.assertTrue(selected.meets_preliminary_candidate_condition.all())
        self.assertTrue(selected.meets_90d_complete_run.all())
        self.assertFalse(selected.no_numeric_values.any())
        self.assertFalse(selected.all_numeric_values_zero.any())
        self.assertFalse(selected.zero_ge_30_flag.any())
        self.assertTrue(selected.eligible_by_frozen_quality_rule.all())

    def test_exclusions_are_explicit_and_pre_access(self) -> None:
        excluded = self.cohort.loc[self.cohort.selection_status.eq("excluded_pre_access")]
        self.assertEqual(set(excluded.exclusion_reason), {"continuous_zero_run_ge_30d"})
        self.assertTrue(excluded.zero_ge_30_flag.all())
        by_site = excluded.groupby("site_id").size().to_dict()
        self.assertEqual(by_site["Mouse"], 7)
        self.assertEqual(by_site["Shrew"], 9)

    def test_no_performance_or_final_power_used_at_gate0(self) -> None:
        self.assertFalse(self.cohort.selection_used_model_performance.any())
        self.assertFalse(self.cohort.final_power_read_at_gate0.any())

    def test_frozen_hash_table_matches_bytes(self) -> None:
        for row in self.hashes.itertuples(index=False):
            path = ROOT / row.path
            self.assertTrue(path.exists(), path)
            self.assertEqual(sha256(path), row.sha256, path)
        cohort_row = self.hashes.loc[self.hashes.role.eq("frozen_final_cohort")].iloc[0]
        self.assertEqual(cohort_row.sha256, sha256(COHORT))

    def test_protocol_manifest_records_fixed_rules(self) -> None:
        text = PROTOCOL.read_text(encoding="utf-8")
        for phrase in (
            "Gate 0: **PASS**",
            "10,000회",
            "p-value/Holm 검정 없음",
            "EPS=1e-12",
            "final_v1_cohort_manifest.csv",
        ):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
