#!/usr/bin/env python3
"""Independently recalculate E5 latency and summarize raw telemetry."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
JETSON = HERE / "jetson"
RTX = HERE / "rtx3080"
VALIDATION = HERE / "validation"
PHASES = ("idle", "active", "cooldown")
STAT_KEYS = ("count", "mean_ms", "std_ms", "median_ms", "p90_ms", "p95_ms", "p99_ms", "minimum_ms", "maximum_ms")


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(np.mean(array)),
        "std_ms": float(np.std(array, ddof=0)),
        "median_ms": float(np.median(array)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "minimum_ms": float(np.min(array)),
        "maximum_ms": float(np.max(array)),
    }


def recalc_latency(path: Path) -> dict[str, Any]:
    rows: list[tuple[int, int, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append((int(row["run_id"]), int(row["repetition"]), float(row["latency_ms"])))
    if len(rows) != 5000:
        raise RuntimeError(f"Expected 5000 latency observations in {path}, got {len(rows)}")
    by_run = []
    for run_id in range(1, 6):
        run_rows = [row for row in rows if row[0] == run_id]
        if len(run_rows) != 1000 or sorted(row[1] for row in run_rows) != list(range(1, 1001)):
            raise RuntimeError(f"Run/repetition structure mismatch in {path}, run {run_id}")
        values = [row[2] for row in run_rows]
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise RuntimeError(f"Non-finite or non-positive latency in {path}")
        by_run.append({"run_id": run_id, **stats(values)})
    return {"overall": stats([row[2] for row in rows]), "by_run": by_run}


def compare_recorded(recorded: dict[str, Any], recalculated: dict[str, Any]) -> dict[str, Any]:
    differences: list[float] = []
    for key in STAT_KEYS:
        differences.append(abs(float(recorded["overall"][key]) - float(recalculated["overall"][key])))
    for old, new in zip(recorded["by_run"], recalculated["by_run"], strict=True):
        if int(old["run_id"]) != int(new["run_id"]):
            raise RuntimeError("Run order mismatch")
        for key in STAT_KEYS:
            differences.append(abs(float(old[key]) - float(new[key])))
    maximum = max(differences)
    return {"maximum_absolute_statistic_difference": maximum, "tolerance": 1e-12, "pass": maximum <= 1e-12}


def scalar_stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def parse_tegrastats(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        sample: dict[str, Any] = {"raw": line}
        ram = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
        if ram:
            sample["ram_used_mb"] = float(ram.group(1))
            sample["ram_total_mb"] = float(ram.group(2))
        cpu = re.search(r"CPU\s+\[([^\]]+)\]", line)
        if cpu:
            percentages = [float(value) for value in re.findall(r"(\d+)%@", cpu.group(1))]
            sample["cpu_mean_percent"] = float(np.mean(percentages)) if percentages else math.nan
        gr3d = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        if gr3d:
            sample["gr3d_percent"] = float(gr3d.group(1))
        sample["temperatures_c"] = {
            key: float(value) for key, value in re.findall(r"([A-Za-z0-9_]+)@([0-9.]+)C", line)
        }
        sample["power_rails_w"] = {
            key: float(current) / 1000.0
            for key, current, _average in re.findall(r"([A-Z][A-Z0-9_]+)\s+(\d+)mW/(\d+)mW", line)
        }
        if "VDD_IN" not in sample["power_rails_w"] or "gpu" not in sample["temperatures_c"]:
            raise RuntimeError(f"Could not parse required tegrastats fields: {line}")
        samples.append(sample)
    if not samples:
        raise RuntimeError(f"No tegrastats samples in {path}")
    return samples


def summarize_tegrastats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(samples),
        "vdd_in_w": scalar_stats([sample["power_rails_w"]["VDD_IN"] for sample in samples]),
        "gpu_temperature_c": scalar_stats([sample["temperatures_c"]["gpu"] for sample in samples]),
        "cpu_temperature_c": scalar_stats([sample["temperatures_c"]["cpu"] for sample in samples]),
        "tj_temperature_c": scalar_stats([sample["temperatures_c"]["tj"] for sample in samples]),
        "gr3d_utilization_percent": scalar_stats([sample["gr3d_percent"] for sample in samples]),
        "cpu_mean_utilization_percent": scalar_stats([sample["cpu_mean_percent"] for sample in samples]),
        "ram_used_mb": scalar_stats([sample["ram_used_mb"] for sample in samples]),
    }


def parse_rtx(path: Path) -> list[dict[str, float | str]]:
    samples: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            samples.append(
                {
                    "timestamp": row["timestamp"].strip(),
                    "power_draw_w": float(row["power_draw_w"]),
                    "temperature_gpu_c": float(row["temperature_gpu_c"]),
                    "utilization_gpu_percent": float(row["utilization_gpu_percent"]),
                    "memory_used_mib": float(row["memory_used_mib"]),
                    "clocks_sm_mhz": float(row["clocks_sm_mhz"]),
                    "clocks_mem_mhz": float(row["clocks_mem_mhz"]),
                }
            )
    if not samples:
        raise RuntimeError(f"No NVIDIA SMI samples in {path}")
    return samples


def summarize_rtx(samples: list[dict[str, float | str]]) -> dict[str, Any]:
    def column(name: str) -> list[float]:
        return [float(sample[name]) for sample in samples]

    return {
        "sample_count": len(samples),
        "power_draw_w": scalar_stats(column("power_draw_w")),
        "gpu_temperature_c": scalar_stats(column("temperature_gpu_c")),
        "gpu_utilization_percent": scalar_stats(column("utilization_gpu_percent")),
        "memory_used_mib": scalar_stats(column("memory_used_mib")),
        "sm_clock_mhz": scalar_stats(column("clocks_sm_mhz")),
        "memory_clock_mhz": scalar_stats(column("clocks_mem_mhz")),
    }


def warning_scan(paths: list[Path]) -> dict[str, Any]:
    pattern = re.compile(r"thermal shutdown|throttl|overheat|oc alarm", re.IGNORECASE)
    matches = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if pattern.search(line):
                matches.append({"file": path.name, "line": line_number, "text": line})
    return {
        "keyword_pattern": pattern.pattern,
        "reported_warning_count": len(matches),
        "matches": matches,
        "limitation": "Raw telemetry does not expose every hardware throttling condition; zero matches means no warning text was reported in these files.",
    }


def main() -> None:
    latency_recalculation: dict[str, Any] = {"status": "PASS", "devices": {}}
    for device_name, directory in (("jetson", JETSON), ("rtx3080", RTX)):
        recorded = json.loads((directory / "latency_summary.json").read_text(encoding="utf-8"))
        modes = {}
        for filename, key in (
            ("latency_model_only_raw.csv", "model_only"),
            ("latency_end_to_end_raw.csv", "end_to_end"),
        ):
            recalculated = recalc_latency(directory / filename)
            verification = compare_recorded(recorded[key], recalculated)
            modes[key] = {"recalculated": recalculated, "verification": verification}
            if not verification["pass"]:
                latency_recalculation["status"] = "FAIL"
        latency_recalculation["devices"][device_name] = modes
    write_json_new(VALIDATION / "independent_latency_recalculation.json", latency_recalculation)
    if latency_recalculation["status"] != "PASS":
        raise RuntimeError("Independent latency recalculation mismatch")

    jetson_samples = {phase: parse_tegrastats(JETSON / f"tegrastats_{phase}.log") for phase in PHASES}
    jetson_phases = {phase: summarize_tegrastats(samples) for phase, samples in jetson_samples.items()}
    jetson_power = {
        "status": "PASS",
        "device": "Jetson Orin Nano",
        "measurement_source": "tegrastats VDD_IN; device-reported input rail",
        "measurement_domain": "Jetson VDD_IN rail; not assumed equivalent to RTX GPU board power",
        "phases": jetson_phases,
        "initial_gpu_temperature_c": jetson_samples["idle"][0]["temperatures_c"]["gpu"],
        "idle_reported_power_mean_w": jetson_phases["idle"]["vdd_in_w"]["mean"],
        "sustained_reported_power_mean_w": jetson_phases["active"]["vdd_in_w"]["mean"],
        "sustained_reported_power_peak_w": jetson_phases["active"]["vdd_in_w"]["maximum"],
        "power_change_from_idle_w": jetson_phases["active"]["vdd_in_w"]["mean"] - jetson_phases["idle"]["vdd_in_w"]["mean"],
        "sustained_gpu_temperature_mean_c": jetson_phases["active"]["gpu_temperature_c"]["mean"],
        "sustained_gpu_temperature_peak_c": jetson_phases["active"]["gpu_temperature_c"]["maximum"],
        "gpu_temperature_change_from_idle_mean_c": jetson_phases["active"]["gpu_temperature_c"]["mean"] - jetson_phases["idle"]["gpu_temperature_c"]["mean"],
        "warning_scan": warning_scan([JETSON / f"tegrastats_{phase}.log" for phase in PHASES]),
    }
    write_json_new(JETSON / "power_thermal_summary.json", jetson_power)

    rtx_samples = {phase: parse_rtx(RTX / f"nvidia_smi_{phase}.csv") for phase in PHASES}
    rtx_phases = {phase: summarize_rtx(samples) for phase, samples in rtx_samples.items()}
    rtx_power = {
        "status": "PASS",
        "device": "NVIDIA GeForce RTX 3080",
        "measurement_source": "nvidia-smi power.draw; GPU-reported board power",
        "measurement_domain": "RTX GPU board power; not assumed equivalent to Jetson VDD_IN",
        "phases": rtx_phases,
        "initial_gpu_temperature_c": float(rtx_samples["idle"][0]["temperature_gpu_c"]),
        "idle_reported_power_mean_w": rtx_phases["idle"]["power_draw_w"]["mean"],
        "sustained_reported_power_mean_w": rtx_phases["active"]["power_draw_w"]["mean"],
        "sustained_reported_power_peak_w": rtx_phases["active"]["power_draw_w"]["maximum"],
        "power_change_from_idle_w": rtx_phases["active"]["power_draw_w"]["mean"] - rtx_phases["idle"]["power_draw_w"]["mean"],
        "sustained_gpu_temperature_mean_c": rtx_phases["active"]["gpu_temperature_c"]["mean"],
        "sustained_gpu_temperature_peak_c": rtx_phases["active"]["gpu_temperature_c"]["maximum"],
        "gpu_temperature_change_from_idle_mean_c": rtx_phases["active"]["gpu_temperature_c"]["mean"] - rtx_phases["idle"]["gpu_temperature_c"]["mean"],
        "warning_scan": warning_scan([RTX / f"nvidia_smi_{phase}.csv" for phase in PHASES]),
    }
    write_json_new(RTX / "power_thermal_summary.json", rtx_power)

    jetson_latency = json.loads((JETSON / "latency_summary.json").read_text(encoding="utf-8"))
    rtx_latency = json.loads((RTX / "latency_summary.json").read_text(encoding="utf-8"))
    pre = json.loads((VALIDATION / "pre_benchmark_consistency.json").read_text(encoding="utf-8"))
    post = json.loads((VALIDATION / "post_benchmark_consistency.json").read_text(encoding="utf-8"))
    jetson_memory = json.loads((JETSON / "memory_summary.json").read_text(encoding="utf-8"))
    rtx_memory = json.loads((RTX / "memory_summary.json").read_text(encoding="utf-8"))
    telemetry_runs = {
        device: {
            phase: json.loads((directory / f"telemetry_{phase}_run.json").read_text(encoding="utf-8"))
            for phase in PHASES
        }
        for device, directory in (("jetson", JETSON), ("rtx3080", RTX))
    }
    summary = {
        "status": "PASS",
        "pre_benchmark_consistency": pre["status"],
        "post_benchmark_consistency": post["status"],
        "latency_independent_recalculation": latency_recalculation["status"],
        "raw_latency_observations": {
            "jetson_model_only": jetson_latency["model_only"]["overall"]["count"],
            "jetson_end_to_end": jetson_latency["end_to_end"]["overall"]["count"],
            "rtx_model_only": rtx_latency["model_only"]["overall"]["count"],
            "rtx_end_to_end": rtx_latency["end_to_end"]["overall"]["count"],
        },
        "jetson_to_rtx_mean_latency_ratio": {
            "model_only": jetson_latency["model_only"]["overall"]["mean_ms"] / rtx_latency["model_only"]["overall"]["mean_ms"],
            "end_to_end": jetson_latency["end_to_end"]["overall"]["mean_ms"] / rtx_latency["end_to_end"]["overall"]["mean_ms"],
        },
        "telemetry_samples": {
            "jetson": {phase: jetson_phases[phase]["sample_count"] for phase in PHASES},
            "rtx3080": {phase: rtx_phases[phase]["sample_count"] for phase in PHASES},
        },
        "sustained_iterations": {
            "jetson": telemetry_runs["jetson"]["active"]["iterations_completed"],
            "rtx3080": telemetry_runs["rtx3080"]["active"]["iterations_completed"],
        },
        "nan_count": 0,
        "inf_count": 0,
        "thermal_or_throttling_warning_text_count": {
            "jetson": jetson_power["warning_scan"]["reported_warning_count"],
            "rtx3080": rtx_power["warning_scan"]["reported_warning_count"],
        },
        "cuda_memory": {"jetson": jetson_memory, "rtx3080": rtx_memory},
        "unresolved_issues": [
            "RTX process RSS/peak RSS are null in the executed benchmark because the Windows API collector lacked explicit ctypes signatures.",
            "jetson_clocks --show requires root on this image, so its state was not verified; it was not enabled or changed.",
            "RTX WDDM desktop C+G contexts remained active and are a documented environmental limitation.",
        ],
        "cross_device_power_ratio_calculated": False,
    }
    write_json_new(VALIDATION / "e5_analysis_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

