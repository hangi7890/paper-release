from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
SUPP = ROOT / "supplementary_final"
DATA = SUPP / "data"
TABLES = SUPP / "tables"
FIGURES = SUPP / "figures"
QA = SUPP / "qa"

REPORTS = ROOT / "reports"
FINAL_SUPPORT = REPORTS / "final_v1"
EDGE = ROOT / "edge_deployment_final"
E4 = EDGE / "e4_cross_device_validation"
E5 = EDGE / "e5_performance_benchmark"

SEEDS = [20260924, 20260925, 20260926]
BUDGETS = [0, 1, 3, 7, 14, 28]
NONZERO_BUDGETS = [1, 3, 7, 14, 28]
EXPECTED_SAMPLES = {0: 0, 1: 1, 3: 49, 7: 145, 14: 313, 28: 649}
EFFECTIVE_BATCH = {0: 0, 1: 1, 3: 49, 7: 64, 14: 64, 28: 64}
BATCHES_PER_EPOCH = {0: 0, 1: 1, 3: 1, 7: 3, 14: 5, 28: 11}
SOURCE_MEAN = 167.0651976323119
SOURCE_STD = 354.328669571624
EPS = 1e-12


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def copy_csv(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.22,
            "savefig.transparent": False,
        }
    )


def prepare_full_data() -> dict[str, pd.DataFrame]:
    cohort = pd.read_csv(REPORTS / "final_v1_cohort_manifest.csv")
    temporal = pd.read_csv(REPORTS / "final_v1_temporal_manifest.csv")
    metrics = pd.read_csv(REPORTS / "final_v1_metrics_by_building_seed_budget.csv")
    efficiency = pd.read_csv(REPORTS / "final_v1_data_efficiency.csv")
    failures = pd.read_csv(REPORTS / "final_v1_failure_events.csv")
    failure_summary = pd.read_csv(REPORTS / "final_v1_failure_summary.csv")
    associations = pd.read_csv(FINAL_SUPPORT / "final_v1_characteristic_associations.csv")
    group_summary = pd.read_csv(FINAL_SUPPORT / "final_v1_characteristic_group_summary.csv")
    characteristics = pd.read_csv(FINAL_SUPPORT / "final_v1_building_characteristics.csv")
    metric_checks = pd.read_csv(FINAL_SUPPORT / "gate3_metric_recalculation_comparison.csv")
    figure8_predictions = pd.read_csv(ROOT / "figure8_extraction" / "case_predictions.csv")
    figure8_metrics = pd.read_csv(ROOT / "figure8_extraction" / "case_metrics.csv")

    copy_csv(REPORTS / "final_v1_cohort_manifest.csv", DATA / "Data_S1_final_cohort_manifest.csv")
    copy_csv(REPORTS / "final_v1_temporal_manifest.csv", DATA / "Data_S2_temporal_manifest.csv")
    copy_csv(REPORTS / "final_v1_metrics_by_building_seed_budget.csv", DATA / "Data_S3_all_model_metrics.csv")
    copy_csv(REPORTS / "final_v1_data_efficiency.csv", DATA / "Data_S4_neural_data_efficiency.csv")

    site_budget_seed = (
        efficiency.groupby(["site_id", "adaptation_days", "seed"], as_index=False)
        .agg(
            buildings=("building_id", "nunique"),
            adaptation_samples=("adaptation_samples", "first"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            source_only_mae=("source_only_mae", "mean"),
            source_only_rmse=("source_only_rmse", "mean"),
            delta_mae_source=("delta_mae_source", "mean"),
            delta_rmse_source=("delta_rmse_source", "mean"),
            a_source_improvement_rate=("a_source_improvement", "mean"),
            b_beats_both_baselines_rate=("b_beats_both_baselines", "mean"),
        )
        .sort_values(["site_id", "adaptation_days", "seed"])
    )
    save_csv(site_budget_seed, DATA / "Data_S5_site_budget_seed_summary.csv")
    copy_csv(REPORTS / "final_v1_failure_events.csv", DATA / "Data_S6_failure_events.csv")

    building_budget_failures = (
        failures.groupby(
            ["building_id", "site_id", "primaryspaceusage", "adaptation_days"], as_index=False, dropna=False
        )
        .agg(
            seed_count=("seed", "nunique"),
            f1_seed_count=("f1", "sum"),
            f2_seed_count=("f2", "sum"),
            f3_previous_day_seed_count=("f3_previous_day", "sum"),
            f3_weekly_same_hour_seed_count=("f3_weekly_same_hour", "sum"),
            f3_any_seed_count=("f3_any", "sum"),
            f3_both_seed_count=("f3_both", "sum"),
            mae_improved_rmse_worsened_seed_count=("mae_improved_rmse_worsened", "sum"),
            f4=("f4", "first"),
            delta_mae_min=("delta_mae_source", "min"),
            delta_mae_max=("delta_mae_source", "max"),
            delta_mae_mean=("delta_mae_source", "mean"),
            delta_rmse_mean=("delta_rmse_source", "mean"),
        )
        .sort_values(["site_id", "building_id", "adaptation_days"])
    )
    save_csv(building_budget_failures, DATA / "Data_S7_failure_by_building_budget.csv")
    copy_csv(FINAL_SUPPORT / "final_v1_characteristic_associations.csv", DATA / "Data_S8_exploratory_associations.csv")
    copy_csv(FINAL_SUPPORT / "final_v1_characteristic_group_summary.csv", DATA / "Data_S9_exploratory_group_summary.csv")
    copy_csv(FINAL_SUPPORT / "final_v1_building_characteristics.csv", DATA / "Data_S10_building_characteristics.csv")

    latency_parts = []
    for device_key, label in [("jetson", "Jetson Orin Nano"), ("rtx3080", "RTX 3080")]:
        for mode, filename in [
            ("model_only", "latency_model_only_raw.csv"),
            ("end_to_end", "latency_end_to_end_raw.csv"),
        ]:
            part = pd.read_csv(E5 / device_key / filename)
            part.insert(0, "mode", mode)
            part.insert(0, "device", label)
            latency_parts.append(part)
    latency = pd.concat(latency_parts, ignore_index=True)
    save_csv(latency, DATA / "Data_S11_edge_latency_observations.csv")

    telemetry_rows = []
    for device_key in ("jetson", "rtx3080"):
        summary = load_json(E5 / device_key / "power_thermal_summary.json")
        for phase, values in summary["phases"].items():
            power_key = "vdd_in_w" if device_key == "jetson" else "power_draw_w"
            power = values[power_key]
            temp = values["gpu_temperature_c"]
            telemetry_rows.append(
                {
                    "device": summary["device"],
                    "phase": phase,
                    "measurement_source": summary["measurement_source"],
                    "measurement_domain": summary["measurement_domain"],
                    "sample_count": values["sample_count"],
                    "power_mean_w": power["mean"],
                    "power_median_w": power["median"],
                    "power_min_w": power["minimum"],
                    "power_max_w": power["maximum"],
                    "gpu_temperature_mean_c": temp["mean"],
                    "gpu_temperature_median_c": temp["median"],
                    "gpu_temperature_min_c": temp["minimum"],
                    "gpu_temperature_max_c": temp["maximum"],
                }
            )
    telemetry = pd.DataFrame(telemetry_rows)
    save_csv(telemetry, DATA / "Data_S12_edge_telemetry_summary.csv")
    validation_reference_dir = EDGE / "e2_transfer_freeze" / "edge_deployment_jetson_fp32" / "validation"
    reference_normalized = np.load(validation_reference_dir / "validation_reference_output_normalized.npy")
    reference_physical = np.load(validation_reference_dir / "validation_reference_output_physical.npy")
    jetson_normalized = np.load(E4 / "jetson_validation_output_normalized.npy")
    jetson_physical = np.load(E4 / "jetson_validation_output_physical.npy")
    if not (
        reference_normalized.shape
        == reference_physical.shape
        == jetson_normalized.shape
        == jetson_physical.shape
        == (64, 24)
    ):
        raise AssertionError("Unexpected E4 validation tensor shape")
    e4_full = pd.DataFrame(
        {
            "vector_id": np.repeat(np.arange(64), 24),
            "horizon_index": np.tile(np.arange(24), 64),
            "normalized_reference": reference_normalized.reshape(-1),
            "normalized_jetson": jetson_normalized.reshape(-1),
            "normalized_abs_diff": np.abs(jetson_normalized - reference_normalized).reshape(-1),
            "physical_reference": reference_physical.reshape(-1),
            "physical_jetson": jetson_physical.reshape(-1),
            "physical_abs_diff": np.abs(jetson_physical - reference_physical).reshape(-1),
        }
    )
    save_csv(e4_full, DATA / "Data_S13_e4_validation_64_comparison.csv")
    copy_csv(ROOT / "figure8_extraction" / "case_predictions.csv", DATA / "Data_S14_figure8_case_predictions.csv")
    copy_csv(ROOT / "figure8_extraction" / "case_metrics.csv", DATA / "Data_S15_figure8_case_metrics.csv")
    copy_csv(FINAL_SUPPORT / "gate3_metric_recalculation_comparison.csv", DATA / "Data_S16_metric_recalculation_checks.csv")

    audit = load_json(FINAL_SUPPORT / "final_stage6_training_config_audit.json")
    checkpoint_rows = []
    for item in audit["audited_execution_files"]:
        if "bdg2_stage6_c0_lstm_seed_" in item["path"]:
            seed = int(item["path"].split("seed_")[1].split("_")[0])
            checkpoint_rows.append({"seed": seed, "path": item["path"], "sha256": item["sha256"]})
    checkpoint_frame = pd.DataFrame(checkpoint_rows).sort_values("seed")
    save_csv(checkpoint_frame, DATA / "Data_S17_source_checkpoint_hashes.csv")

    return {
        "cohort": cohort,
        "temporal": temporal,
        "metrics": metrics,
        "efficiency": efficiency,
        "failures": failures,
        "failure_summary": failure_summary,
        "associations": associations,
        "group_summary": group_summary,
        "characteristics": characteristics,
        "metric_checks": metric_checks,
        "figure8_predictions": figure8_predictions,
        "figure8_metrics": figure8_metrics,
        "site_budget_seed": site_budget_seed,
        "building_budget_failures": building_budget_failures,
        "latency": latency,
        "telemetry": telemetry,
        "checkpoints": checkpoint_frame,
    }


def build_tables(d: dict[str, pd.DataFrame]) -> None:
    cohort = d["cohort"]
    selected = cohort.loc[cohort["selection_status"].eq("selected_for_final_v1")]
    development = pd.read_csv(REPORTS / "bdg2_multibuilding_cohort_manifest_v2.csv")
    development = development.loc[development["selected_for_multibuilding_v2"].astype(str).str.lower().eq("true")]
    cohort_summary = pd.DataFrame(
        [
            {"cohort": "Source", "purpose": "Source pretraining/model selection", "buildings": int((development.selection_role == "source_pretraining").sum()), "sites": int(development.loc[development.selection_role == "source_pretraining", "site_id"].nunique())},
            {"cohort": "Development", "purpose": "Development adaptation and evaluation", "buildings": int((development.selection_role == "development_adaptation_evaluation").sum()), "sites": int(development.loc[development.selection_role == "development_adaptation_evaluation", "site_id"].nunique())},
            {"cohort": "Final proposed", "purpose": "Frozen final-test proposal before quality exclusion", "buildings": len(cohort), "sites": cohort.site_id.nunique()},
            {"cohort": "Final retained", "purpose": "Independent final evaluation", "buildings": len(selected), "sites": selected.site_id.nunique()},
        ]
    )
    save_csv(cohort_summary, TABLES / "Table_S1_A_cohort_summary.csv")
    site_counts = (
        cohort.groupby("site_id", as_index=False)
        .agg(
            proposed_buildings=("building_id", "size"),
            retained_buildings=("selection_status", lambda x: int((x == "selected_for_final_v1").sum())),
            excluded_buildings=("selection_status", lambda x: int((x != "selected_for_final_v1").sum())),
        )
        .sort_values("site_id")
    )
    site_counts["retained_site"] = site_counts.retained_buildings.gt(0)
    save_csv(site_counts, TABLES / "Table_S1_B_final_site_counts.csv")

    audit = load_json(FINAL_SUPPORT / "final_stage6_training_config_audit.json")
    config_rows = [
        ("Input", "Sequence length", "24", "hours"),
        ("Input", "Features per step", "6", "normalized past electricity plus five target-calendar features"),
        ("Output", "Forecast horizon", "24", "direct multi-output hours"),
        ("Architecture", "LSTM hidden units/layers", "32 / 1", "single-layer LSTM"),
        ("Architecture", "Trainable parameters", "5912", "FP32 deployment model"),
        ("Normalization", "Source mean", repr(SOURCE_MEAN), "source-only training data"),
        ("Normalization", "Source population standard deviation", repr(SOURCE_STD), "source-only training data"),
        ("Source training", "Loss", audit["source_training_loss"]["result"], "mean reduction"),
        ("Source training", "Optimizer and learning rate", "Adam; 0.001", "all model parameters"),
        ("Source training", "Batch rule", "256; OOM-only fallbacks 128, 64", "completed run used 256"),
        ("Source training", "Samples", "218206 train; 12980 validation", "fixed source buildings"),
        ("Source training", "Epoch rule", "maximum 100; patience 10; min_delta 1e-5", "validation MSE selection"),
        ("Source training", "Shuffle / drop_last", "True / False", "partial source batch retained"),
        ("Adaptation", "Loss", audit["target_adaptation_loss"]["result"], "same loss as source training"),
        ("Adaptation", "Optimizer and learning rate", "Adam; 0.0005", "output head only"),
        ("Adaptation", "Epochs", "50", "no early stopping"),
        ("Adaptation", "Nominal candidate batch", "64", "fallback 32 then 16 only on CUDA OOM"),
        ("Adaptation", "Actual loader rule", "min(candidate batch, number of samples)", "drop_last=False"),
        ("Adaptation", "Shuffle / drop_last", "True / False", "seeded generator"),
        ("Adaptation", "Frozen parameters", "all LSTM parameters", "optimizer receives model.head.parameters() only"),
    ]
    config = pd.DataFrame(config_rows, columns=["section", "setting", "value", "interpretation"])
    save_csv(config, TABLES / "Table_S2_training_and_adaptation_configuration.csv")
    batch_table = pd.DataFrame(
        [
            {
                "adaptation_days": day,
                "supervised_windows": EXPECTED_SAMPLES[day],
                "effective_loader_batch_size": EFFECTIVE_BATCH[day],
                "batches_per_epoch": BATCHES_PER_EPOCH[day],
                "batch_sizes_per_epoch": {
                    0: "none", 1: "1", 3: "49", 7: "64,64,17", 14: "64,64,64,64,57", 28: "64x10,9"
                }[day],
            }
            for day in BUDGETS
        ]
    )
    save_csv(batch_table, TABLES / "Table_S2_B_budget_batch_behavior.csv")

    curve = pd.read_csv(FINAL_SUPPORT / "final_v1_site_equal_adaptation_curve.csv")
    metrics = d["metrics"]
    baseline_rows = []
    for model in ["previous_day", "weekly_same_hour"]:
        part = metrics.loc[metrics.model.eq(model)]
        site = part.groupby("site_id", as_index=False).agg(mae=("mae", "mean"), rmse=("rmse", "mean"))
        baseline_rows.append(
            {
                "condition": model,
                "adaptation_days": np.nan,
                "sites": site.site_id.nunique(),
                "mae_site_equal": site.mae.mean(),
                "rmse_site_equal": site.rmse.mean(),
                "delta_mae_source_site_equal": np.nan,
                "delta_rmse_source_site_equal": np.nan,
                "a_rate_site_equal": np.nan,
                "b_rate_site_equal": np.nan,
            }
        )
    performance = curve.copy()
    performance.insert(0, "condition", performance.adaptation_days.map(lambda x: f"C0_{int(x)}d"))
    performance = pd.concat([performance, pd.DataFrame(baseline_rows)], ignore_index=True)
    save_csv(performance, TABLES / "Table_S3_site_equal_performance.csv")

    hypothesis_global = pd.DataFrame(
        [
            {"hypothesis": "H1", "estimand": "site-equal MAE(28d) - MAE(source-only)", "estimate": -16.482888133, "ci95_lower": -30.907569879, "ci95_upper": -6.56297796855, "status": "supported by predefined criterion"},
            {"hypothesis": "H2", "estimand": "site-equal F1 rate(28d) - F1 rate(1d)", "estimate": -0.551526864698, "ci95_lower": -0.679617292789, "ci95_upper": -0.455237391104, "status": "secondary effect-size analysis"},
            {"hypothesis": "H3", "estimand": "site-equal A rate(28d) - B rate(28d)", "estimate": 0.679884877048, "ci95_lower": 0.583579951647, "ci95_upper": 0.783540480704, "status": "secondary effect-size analysis"},
        ]
    )
    save_csv(hypothesis_global, TABLES / "Table_S4_A_hypothesis_effects.csv")
    site_effect_parts = []
    for label, path in [
        ("H1", FINAL_SUPPORT / "final_v1_h1_site_effects.csv"),
        ("H2", FINAL_SUPPORT / "final_v1_h2_site_effects.csv"),
        ("H3", FINAL_SUPPORT / "final_v1_h3_site_effects.csv"),
    ]:
        part = pd.read_csv(path)
        part.insert(0, "hypothesis", label)
        site_effect_parts.append(part)
    site_effects = pd.concat(site_effect_parts, ignore_index=True, sort=False)
    save_csv(site_effects, TABLES / "Table_S4_B_site_effects.csv")
    save_csv(d["failure_summary"], TABLES / "Table_S5_failure_counts_and_rates.csv")

    associations = d["associations"]
    association_summary = (
        associations.groupby(["feature", "response"], as_index=False)
        .agg(
            budgets=("adaptation_days", "nunique"),
            valid_correlations=("spearman_rho", lambda x: int(x.notna().sum())),
            n_complete_min=("n_complete", "min"),
            n_complete_max=("n_complete", "max"),
            spearman_rho_min=("spearman_rho", "min"),
            spearman_rho_max=("spearman_rho", "max"),
        )
        .sort_values(["response", "feature"])
    )
    save_csv(association_summary, TABLES / "Table_S6_A_exploratory_association_ranges.csv")
    missing_use = d["group_summary"].loc[
        d["group_summary"].group_field.eq("primaryspaceusage")
        & d["group_summary"].group_value.eq("<missing>")
    ].copy()
    save_csv(missing_use, TABLES / "Table_S6_B_swan_missing_use_category.csv")

    e4 = load_json(E4 / "validation_64_comparison_summary.json")
    e4_rows = []
    for scale in ("normalized", "physical"):
        vals = e4["validation_64"][scale]
        crit = e4["criteria"][scale]
        e4_rows.append(
            {
                "scale": scale,
                "output_elements": e4["output_elements"],
                "max_abs_difference": vals["max_absolute_difference"],
                "max_abs_tolerance": crit["max_absolute_difference_lte"],
                "mean_abs_difference": vals["mean_absolute_difference"],
                "mean_abs_tolerance": crit["mean_absolute_difference_lte"],
                "rmse_difference": vals["rmse_difference"],
                "nan_count": vals["candidate_nan_count"],
                "inf_count": vals["candidate_inf_count"],
                "status": "PASS" if all(e4["checks"].values()) else "FAIL",
            }
        )
    save_csv(pd.DataFrame(e4_rows), TABLES / "Table_S7_cross_device_fp32_consistency.csv")

    latency_rows = []
    memory_rows = []
    for key, label in [("jetson", "Jetson Orin Nano"), ("rtx3080", "RTX 3080")]:
        latency_json = load_json(E5 / key / "latency_summary.json")
        for mode in ("model_only", "end_to_end"):
            vals = latency_json[mode]["overall"]
            latency_rows.append({"device": label, "mode": mode, **vals})
        memory = load_json(E5 / key / "memory_summary.json")
        memory_rows.append(
            {
                "device": label,
                **memory,
                "cuda_peak_allocated_mib": memory["cuda_max_memory_allocated_bytes"] / 2**20,
                "cuda_peak_reserved_mib": memory["cuda_max_memory_reserved_bytes"] / 2**20,
                "rss_status": "unavailable" if memory["rss_current_bytes"] is None else "measured",
            }
        )
    save_csv(pd.DataFrame(latency_rows), TABLES / "Table_S8_A_edge_latency.csv")
    save_csv(pd.DataFrame(memory_rows), TABLES / "Table_S8_B_edge_memory.csv")
    save_csv(d["telemetry"], TABLES / "Table_S8_C_edge_telemetry.csv")

    raw_manifest = pd.read_csv(REPORTS / "final_v1_artifact_manifest.csv")
    raw_row = raw_manifest.loc[raw_manifest.relative_path.eq("reports/final_v1_raw_predictions.csv")].iloc[0]
    reproducibility_rows = [
        {"artifact": "final raw predictions", "path": raw_row.relative_path, "size_bytes": int(raw_row.size_bytes), "sha256": raw_row.sha256, "verification": "frozen manifest and Figure 8 search audit"},
        {"artifact": "final cohort manifest", "path": "reports/final_v1_cohort_manifest.csv", "size_bytes": (REPORTS / "final_v1_cohort_manifest.csv").stat().st_size, "sha256": sha256(REPORTS / "final_v1_cohort_manifest.csv"), "verification": "Gate 0 frozen input"},
        {"artifact": "final temporal manifest", "path": "reports/final_v1_temporal_manifest.csv", "size_bytes": (REPORTS / "final_v1_temporal_manifest.csv").stat().st_size, "sha256": sha256(REPORTS / "final_v1_temporal_manifest.csv"), "verification": "Gate 1 frozen input"},
        {"artifact": "final metric table", "path": "reports/final_v1_metrics_by_building_seed_budget.csv", "size_bytes": (REPORTS / "final_v1_metrics_by_building_seed_budget.csv").stat().st_size, "sha256": sha256(REPORTS / "final_v1_metrics_by_building_seed_budget.csv"), "verification": "independent metric output"},
        {"artifact": "Stage-6 training code", "path": "scripts/run_bdg2_stage6_c0_lstm_v3.py", "size_bytes": (ROOT / "scripts" / "run_bdg2_stage6_c0_lstm_v3.py").stat().st_size, "sha256": sha256(ROOT / "scripts" / "run_bdg2_stage6_c0_lstm_v3.py"), "verification": "read-only training audit"},
        {"artifact": "C0 feature code", "path": "scripts/bdg2_stage6_c0_common_v3.py", "size_bytes": (ROOT / "scripts" / "bdg2_stage6_c0_common_v3.py").stat().st_size, "sha256": sha256(ROOT / "scripts" / "bdg2_stage6_c0_common_v3.py"), "verification": "read-only input audit"},
    ]
    for _, row in d["checkpoints"].iterrows():
        path = ROOT / row.path
        external = json.loads((Path(__file__).resolve().parents[2] / "configs" / "external_artifacts.json").read_text(encoding="utf-8"))
        recorded_size = next(item["size_bytes"] for item in external if item["path"] == row.path)
        reproducibility_rows.append({"artifact": f"source checkpoint seed {int(row.seed)}", "path": row.path, "size_bytes": path.stat().st_size if path.exists() else recorded_size, "sha256": row.sha256, "verification": "Stage-6/final Gate 2 frozen hash"})
    save_csv(pd.DataFrame(reproducibility_rows), TABLES / "Table_S9_reproducibility_hashes.csv")

    cases = pd.read_csv(FINAL_SUPPORT / "final_v1_representative_failure_cases.csv")
    figure8_metrics = d["figure8_metrics"].merge(
        cases[["case_type", "building_id", "adaptation_days", "selection_rule", "selection_value"]],
        on=["building_id", "adaptation_days"], how="left", validate="many_to_one"
    )
    save_csv(figure8_metrics, TABLES / "Table_S10_figure8_case_metrics.csv")


def figure_s1() -> None:
    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(BUDGETS))
    samples = [EXPECTED_SAMPLES[d] for d in BUDGETS]
    batches = [EFFECTIVE_BATCH[d] for d in BUDGETS]
    bars = ax1.bar(x, samples, color="#356a8a", width=0.66, label="Supervised windows")
    ax1.set_xticks(x, [str(d) for d in BUDGETS])
    ax1.set_xlabel("Target-adaptation budget (days)")
    ax1.set_ylabel("Supervised 24-to-24 windows")
    ax1.grid(axis="y")
    for bar, value in zip(bars, samples):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 10, str(value), ha="center", va="bottom", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, batches, marker="o", color="#b24a3a", linewidth=2, label="Effective loader batch")
    ax2.set_ylabel("Effective loader batch size")
    ax2.set_ylim(0, 72)
    for xi, day, value in zip(x, BUDGETS, batches):
        if day > 0:
            ax2.annotate(f"{BATCHES_PER_EPOCH[day]} batch{'es' if BATCHES_PER_EPOCH[day] != 1 else ''}/epoch", (xi, value), xytext=(0, 11), textcoords="offset points", ha="center", va="bottom", fontsize=7, color="#7f3028")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", frameon=False)
    ax1.set_title("Supplementary Figure S1. Adaptation windows and effective batching")
    fig.tight_layout()
    save_figure(fig, "Figure_S1_adaptation_windows_and_batching")


def figure_s2(site_budget_seed: pd.DataFrame) -> None:
    site_curve = site_budget_seed.groupby(["site_id", "adaptation_days"], as_index=False).delta_mae_source.mean()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    colors = plt.cm.tab10(np.linspace(0, 0.7, site_curve.site_id.nunique()))
    for color, (site, part) in zip(colors, site_curve.groupby("site_id", sort=True)):
        part = part.sort_values("adaptation_days")
        ax.plot(part.adaptation_days, part.delta_mae_source, marker="o", linewidth=1.8, label=f"{site} (n={int(site_budget_seed.loc[site_budget_seed.site_id.eq(site), 'buildings'].max())})", color=color)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(BUDGETS)
    ax.set_xlabel("Target-adaptation budget (days)")
    ax.set_ylabel("Mean ΔMAE relative to source-only (original meter units)")
    ax.set_title("Supplementary Figure S2. Site-level adaptation effects")
    ax.grid(True)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    save_figure(fig, "Figure_S2_site_level_delta_mae")


def figure_s3(summary: pd.DataFrame) -> None:
    part = summary.loc[summary.adaptation_days.astype(str).isin([str(d) for d in NONZERO_BUDGETS])].copy()
    part["adaptation_days"] = part.adaptation_days.astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for column, label, color in [
        ("f1_rate", "F1 MAE worse than source", "#b64949"),
        ("f2_rate", "F2 RMSE worse than source", "#6c3f7d"),
        ("f3_any_rate", "F3-any worse than either baseline", "#d17a22"),
        ("f3_both_rate", "F3-both worse than both baselines", "#8d6e63"),
    ]:
        axes[0].plot(part.adaptation_days, 100 * part[column], marker="o", label=label, color=color)
    axes[0].set_xticks(NONZERO_BUDGETS)
    axes[0].set_xlabel("Adaptation budget (days)")
    axes[0].set_ylabel("Pooled building × seed events (%)")
    axes[0].set_title("(a) F1-F3, denominator 1,359 per budget")
    axes[0].grid(True)
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].plot(part.adaptation_days, 100 * part.f4_rate, marker="o", color="#287c7c", label="F4 mixed seed direction")
    axes[1].plot(part.adaptation_days, 100 * part.mae_improved_rmse_worsened_rate, marker="s", color="#355c9a", label="MAE improved / RMSE worsened")
    axes[1].set_xticks(NONZERO_BUDGETS)
    axes[1].set_xlabel("Adaptation budget (days)")
    axes[1].set_ylabel("Events (%)")
    axes[1].set_title("(b) F4 denominator 453; discordance denominator 1,359")
    axes[1].grid(True)
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Supplementary Figure S3. Complete adaptation outcome event rates")
    fig.tight_layout()
    save_figure(fig, "Figure_S3_failure_event_rates")


def figure_s4(associations: pd.DataFrame) -> None:
    responses = [
        "seed_mean_mae_delta",
        "seed_mean_rmse_delta",
        "f1_seed_rate",
        "f2_seed_rate",
        "f3_any_seed_rate",
    ]
    features = associations.feature.drop_duplicates().tolist()
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 16.5), constrained_layout=True)
    axes = axes.ravel()
    image = None
    for ax, response in zip(axes, responses):
        pivot = associations.loc[associations.response.eq(response)].pivot(index="feature", columns="adaptation_days", values="spearman_rho").reindex(index=features, columns=NONZERO_BUDGETS)
        image = ax.imshow(pivot.to_numpy(float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(NONZERO_BUDGETS)), NONZERO_BUDGETS)
        ax.set_yticks(range(len(features)), [f.replace("_", " ") for f in features], fontsize=7)
        ax.set_xlabel("Adaptation budget (days)")
        ax.set_title(response.replace("_", " "))
        for i in range(len(features)):
            for j in range(len(NONZERO_BUDGETS)):
                value = pivot.iloc[i, j]
                text = "N/A" if pd.isna(value) else f"{value:.2f}"
                color = "white" if pd.notna(value) and abs(value) > 0.55 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=6, color=color)
    axes[-1].axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes[:5].tolist(), shrink=0.7, label="Spearman ρ")
    fig.suptitle("Supplementary Figure S4. Full exploratory association matrix", fontsize=13)
    save_figure(fig, "Figure_S4_full_exploratory_associations")


def figure_s5(latency: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), sharex=True)
    groups = [
        ("Jetson Orin Nano", "model_only"),
        ("Jetson Orin Nano", "end_to_end"),
        ("RTX 3080", "model_only"),
        ("RTX 3080", "end_to_end"),
    ]
    for ax, (device, mode) in zip(axes.ravel(), groups):
        part = latency.loc[latency.device.eq(device) & latency["mode"].eq(mode)]
        arrays = [part.loc[part.run_id.eq(run), "latency_ms"].to_numpy(float) for run in range(1, 6)]
        ax.boxplot(arrays, showfliers=True, widths=0.6, patch_artist=True, boxprops={"facecolor": "#d8e7ef", "edgecolor": "#356a8a"}, medianprops={"color": "#b24a3a", "linewidth": 1.5}, flierprops={"markersize": 2, "alpha": 0.45})
        ax.set_yscale("log")
        ax.set_xticks(range(1, 6), [str(i) for i in range(1, 6)])
        ax.set_title(f"{device}: {mode.replace('_', ' ')}")
        ax.set_xlabel("Run")
        ax.set_ylabel("Latency (ms, log scale)")
        ax.grid(axis="y", which="both")
    fig.suptitle("Supplementary Figure S5. Run-level latency distributions with all outliers retained")
    fig.tight_layout()
    save_figure(fig, "Figure_S5_edge_latency_by_run")


def figure_s6(telemetry: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    phase_order = ["idle", "active", "cooldown"]
    for row, device in enumerate(["Jetson Orin Nano", "NVIDIA GeForce RTX 3080"]):
        part = telemetry.loc[telemetry.device.eq(device)].set_index("phase").reindex(phase_order)
        x = np.arange(3)
        axes[row, 0].errorbar(
            x,
            part.power_mean_w,
            yerr=[part.power_mean_w - part.power_min_w, part.power_max_w - part.power_mean_w],
            marker="o",
            capsize=3,
            color="#356a8a",
        )
        axes[row, 0].set_xticks(x, phase_order)
        axes[row, 0].set_ylabel("Reported power (W)")
        axes[row, 0].set_title(f"{device}: device-specific power boundary")
        axes[row, 0].grid(axis="y")
        axes[row, 1].errorbar(
            x,
            part.gpu_temperature_mean_c,
            yerr=[part.gpu_temperature_mean_c - part.gpu_temperature_min_c, part.gpu_temperature_max_c - part.gpu_temperature_mean_c],
            marker="o",
            capsize=3,
            color="#b24a3a",
        )
        axes[row, 1].set_xticks(x, phase_order)
        axes[row, 1].set_ylabel("GPU temperature (°C)")
        axes[row, 1].set_title(f"{device}: GPU temperature")
        axes[row, 1].grid(axis="y")
    fig.suptitle("Supplementary Figure S6. Device-specific power and thermal phases")
    fig.tight_layout()
    save_figure(fig, "Figure_S6_edge_power_thermal_phases")


def figure_s7(efficiency: pd.DataFrame) -> None:
    buildings = ["Hog_office_Roger", "Eagle_education_Brooke", "Hog_office_Gustavo"]
    labels = ["Repeated F1", "F4 seed-direction mixture", "MAE improved / RMSE worsened"]
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.5), constrained_layout=True)
    filtered = efficiency.loc[efficiency.building_id.isin(buildings) & efficiency.adaptation_days.isin(NONZERO_BUDGETS)]
    max_mae = float(np.nanmax(np.abs(filtered.delta_mae_source)))
    max_rmse = float(np.nanmax(np.abs(filtered.delta_rmse_source)))
    for i, (building, label) in enumerate(zip(buildings, labels)):
        part = filtered.loc[filtered.building_id.eq(building)]
        for j, (metric, bound, title) in enumerate([
            ("delta_mae_source", max_mae, "ΔMAE"),
            ("delta_rmse_source", max_rmse, "ΔRMSE"),
        ]):
            pivot = part.pivot(index="seed", columns="adaptation_days", values=metric).reindex(index=SEEDS, columns=NONZERO_BUDGETS)
            ax = axes[i, j]
            im = ax.imshow(pivot.to_numpy(float), cmap="coolwarm", vmin=-bound, vmax=bound, aspect="auto")
            ax.set_xticks(range(5), NONZERO_BUDGETS)
            ax.set_yticks(range(3), [str(s) for s in SEEDS])
            ax.set_xlabel("Adaptation days")
            ax.set_ylabel("Seed")
            ax.set_title(f"{building}: {title}\n{label}")
            for r in range(3):
                for c in range(5):
                    value = pivot.iloc[r, c]
                    ax.text(c, r, f"{value:.1f}", ha="center", va="center", fontsize=7, color="white" if abs(value) > 0.55 * bound else "black")
            fig.colorbar(im, ax=ax, shrink=0.75, label="original meter units")
    fig.suptitle("Supplementary Figure S7. Full budget-by-seed patterns for frozen representative cases")
    save_figure(fig, "Figure_S7_representative_case_budget_seed_matrix")


def validate_outputs(d: dict[str, pd.DataFrame]) -> dict:
    failures = d["failures"]
    checks = {
        "cohort_rows_510": len(d["cohort"]) == 510,
        "retained_buildings_453": d["cohort"].selection_status.eq("selected_for_final_v1").sum() == 453,
        "retained_sites_7": d["cohort"].loc[d["cohort"].selection_status.eq("selected_for_final_v1"), "site_id"].nunique() == 7,
        "temporal_rows_2718": len(d["temporal"]) == 453 * 6,
        "temporal_unique_key": not d["temporal"].duplicated(["building_id", "adaptation_days"]).any(),
        "all_temporal_structure_pass": d["temporal"].structure_status.eq("PASS").all(),
        "all_test_hours_672": d["temporal"].test_target_hour_count.eq(672).all(),
        "metric_rows_9060": len(d["metrics"]) == 9060,
        "metric_unique_key": not d["metrics"].duplicated(["building_id", "model", "seed", "adaptation_days"]).any(),
        "neural_rows_8154": len(d["efficiency"]) == 453 * 3 * 6,
        "neural_unique_key": not d["efficiency"].duplicated(["building_id", "seed", "adaptation_days"]).any(),
        "site_budget_seed_rows_126": len(d["site_budget_seed"]) == 7 * 6 * 3,
        "failure_rows_6795": len(failures) == 453 * 5 * 3,
        "building_budget_rows_2265": len(d["building_budget_failures"]) == 453 * 5,
        "association_rows_350": len(d["associations"]) == 350,
        "swan_missing_category_18": int(d["group_summary"].loc[(d["group_summary"].group_field == "primaryspaceusage") & (d["group_summary"].group_value == "<missing>"), "buildings"].max()) == 18,
        "edge_latency_rows_20000": len(d["latency"]) == 2 * 2 * 5000,
        "e4_vector_rows_1536": len(pd.read_csv(DATA / "Data_S13_e4_validation_64_comparison.csv")) == 64 * 24,
        "figure8_rows_6720": len(d["figure8_predictions"]) == 6720,
        "figure8_combinations_10": d["figure8_predictions"].groupby(["building_id", "seed", "adaptation_days"]).ngroups == 10,
        "figure8_all_metrics_pass": d["figure8_metrics"].validation_status.eq("PASS").all(),
        "independent_metric_max_diff_lte_1e10": max(d["metric_checks"].mae_abs_difference.max(), d["metric_checks"].rmse_abs_difference.max()) <= 1e-10,
        "f1_total_1508": int(failures.f1.sum()) == 1508,
        "f2_total_1399": int(failures.f2.sum()) == 1399,
        "f3_any_total_5828": int(failures.f3_any.sum()) == 5828,
        "f4_total_448": int(d["building_budget_failures"].f4.sum()) == 448,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "retained_buildings": 453,
            "retained_sites": 7,
            "metric_rows_all_models": len(d["metrics"]),
            "neural_building_seed_budget_rows": len(d["efficiency"]),
            "latency_observations": len(d["latency"]),
            "figure8_rows": len(d["figure8_predictions"]),
        },
    }
    (QA / "supplementary_data_qa.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["status"] != "PASS":
        raise AssertionError(result)
    return result


def main() -> None:
    for directory in (DATA, TABLES, FIGURES, QA):
        directory.mkdir(parents=True, exist_ok=True)
    setup_style()
    data = prepare_full_data()
    build_tables(data)
    figure_s1()
    figure_s2(data["site_budget_seed"])
    figure_s3(data["failure_summary"])
    figure_s4(data["associations"])
    figure_s5(data["latency"])
    figure_s6(data["telemetry"])
    figure_s7(data["efficiency"])
    qa = validate_outputs(data)
    print(json.dumps(qa, indent=2))


def resume_figures_s5_to_s7_and_qa() -> None:
    """Resume after the completed data/tables/S1-S4 phase without regenerating it."""
    setup_style()
    latency = pd.read_csv(DATA / "Data_S11_edge_latency_observations.csv")
    telemetry = pd.read_csv(DATA / "Data_S12_edge_telemetry_summary.csv")
    efficiency = pd.read_csv(DATA / "Data_S4_neural_data_efficiency.csv")
    figure_s5(latency)
    figure_s6(telemetry)
    figure_s7(efficiency)
    validation_data = {
        "cohort": pd.read_csv(DATA / "Data_S1_final_cohort_manifest.csv"),
        "temporal": pd.read_csv(DATA / "Data_S2_temporal_manifest.csv"),
        "metrics": pd.read_csv(DATA / "Data_S3_all_model_metrics.csv"),
        "efficiency": efficiency,
        "failures": pd.read_csv(DATA / "Data_S6_failure_events.csv"),
        "failure_summary": pd.read_csv(TABLES / "Table_S5_failure_counts_and_rates.csv"),
        "associations": pd.read_csv(DATA / "Data_S8_exploratory_associations.csv"),
        "group_summary": pd.read_csv(DATA / "Data_S9_exploratory_group_summary.csv"),
        "characteristics": pd.read_csv(DATA / "Data_S10_building_characteristics.csv"),
        "metric_checks": pd.read_csv(DATA / "Data_S16_metric_recalculation_checks.csv"),
        "figure8_predictions": pd.read_csv(DATA / "Data_S14_figure8_case_predictions.csv"),
        "figure8_metrics": pd.read_csv(DATA / "Data_S15_figure8_case_metrics.csv"),
        "site_budget_seed": pd.read_csv(DATA / "Data_S5_site_budget_seed_summary.csv"),
        "building_budget_failures": pd.read_csv(DATA / "Data_S7_failure_by_building_budget.csv"),
        "latency": latency,
        "telemetry": telemetry,
    }
    qa = validate_outputs(validation_data)
    print(json.dumps(qa, indent=2))


def resume_qa_only() -> None:
    validation_data = {
        "cohort": pd.read_csv(DATA / "Data_S1_final_cohort_manifest.csv"),
        "temporal": pd.read_csv(DATA / "Data_S2_temporal_manifest.csv"),
        "metrics": pd.read_csv(DATA / "Data_S3_all_model_metrics.csv"),
        "efficiency": pd.read_csv(DATA / "Data_S4_neural_data_efficiency.csv"),
        "failures": pd.read_csv(DATA / "Data_S6_failure_events.csv"),
        "failure_summary": pd.read_csv(TABLES / "Table_S5_failure_counts_and_rates.csv"),
        "associations": pd.read_csv(DATA / "Data_S8_exploratory_associations.csv"),
        "group_summary": pd.read_csv(DATA / "Data_S9_exploratory_group_summary.csv"),
        "characteristics": pd.read_csv(DATA / "Data_S10_building_characteristics.csv"),
        "metric_checks": pd.read_csv(DATA / "Data_S16_metric_recalculation_checks.csv"),
        "figure8_predictions": pd.read_csv(DATA / "Data_S14_figure8_case_predictions.csv"),
        "figure8_metrics": pd.read_csv(DATA / "Data_S15_figure8_case_metrics.csv"),
        "site_budget_seed": pd.read_csv(DATA / "Data_S5_site_budget_seed_summary.csv"),
        "building_budget_failures": pd.read_csv(DATA / "Data_S7_failure_by_building_budget.csv"),
        "latency": pd.read_csv(DATA / "Data_S11_edge_latency_observations.csv"),
        "telemetry": pd.read_csv(DATA / "Data_S12_edge_telemetry_summary.csv"),
    }
    qa = validate_outputs(validation_data)
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
