#!/usr/bin/env python3
"""E5 FP32 benchmark for the frozen C0 LSTM bundle.

This script does not train, adapt, compile, quantize, or modify the model.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPECTED_CHECKPOINT_SHA256 = "88eb3173435a5ea728e9490ad7139485162ad9c6ff4bce9019b1eb4121b4908b"
EXPECTED_PARAMETER_COUNT = 5912
EXPECTED_INPUT_SHAPE = (1, 24, 6)
EXPECTED_OUTPUT_SHAPE = (1, 24)
WARMUP = 200
RUNS = 5
REPETITIONS = 1000
STABILIZATION_SECONDS = 1.0
CRITERIA = {
    "normalized_max_abs_diff_lte": 1e-4,
    "normalized_mean_abs_diff_lte": 2e-5,
    "physical_max_abs_diff_lte": 0.05,
    "physical_mean_abs_diff_lte": 0.01,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text_new(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def run_capture(args: list[str], timeout: int = 20) -> str:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return (
            f"$ {' '.join(args)}\nreturncode={completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout}--- stderr ---\n{completed.stderr}"
        )
    except Exception as exc:  # environment evidence should survive unsupported queries
        return f"$ {' '.join(args)}\nERROR: {type(exc).__name__}: {exc}\n"


def environment_report(kind: str) -> str:
    lines = [
        f"platform_kind={kind}",
        f"hostname={platform.node()}",
        f"platform={platform.platform()}",
        f"python={sys.version}",
        f"torch={torch.__version__}",
        f"torch_cuda={torch.version.cuda}",
        f"numpy={np.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"device_count={torch.cuda.device_count()}",
    ]
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        lines.extend(
            [
                f"device={torch.cuda.get_device_name(0)}",
                f"capability={torch.cuda.get_device_capability(0)}",
                f"gpu_memory_total_bytes={props.total_memory}",
                f"tf32_matmul_allow={torch.backends.cuda.matmul.allow_tf32}",
                f"tf32_cudnn_allow={torch.backends.cudnn.allow_tf32}",
            ]
        )

    if kind == "jetson":
        release = Path("/etc/nv_tegra_release")
        if release.exists():
            lines.append("\n## /etc/nv_tegra_release\n" + release.read_text(encoding="utf-8", errors="replace"))
        lines.append("\n## nvpmodel\n" + run_capture(["/usr/sbin/nvpmodel", "-q"]))
        lines.append("\n## jetson_clocks\n" + run_capture(["/usr/bin/jetson_clocks", "--show"]))
        lines.append("\n## free\n" + run_capture(["free", "-b"]))
        lines.append("\n## initial tegrastats\n" + run_capture(["timeout", "3", "tegrastats", "--interval", "1000"], timeout=8))
    elif kind == "rtx":
        nvidia_smi = shutil.which("nvidia-smi") or "nvidia-smi"
        lines.append(
            "\n## nvidia-smi gpu\n"
            + run_capture(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version,memory.total,temperature.gpu,power.draw,utilization.gpu",
                    "--format=csv,noheader",
                ]
            )
        )
        lines.append(
            "\n## nvidia-smi compute contexts\n"
            + run_capture(
                [
                    nvidia_smi,
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ]
            )
        )
        lines.append("\n## nvidia-smi full\n" + run_capture([nvidia_smi]))
    else:
        raise ValueError(kind)
    return "\n".join(lines).rstrip() + "\n"


def load_bundle(bundle_root: Path, device: torch.device) -> tuple[Any, dict[str, Any], Path, str, dict[str, Any]]:
    scripts = bundle_root / "scripts"
    sys.path.insert(0, str(scripts))
    from model_definition import C0LSTM, ModelConfig  # type: ignore

    raw = json.loads((bundle_root / "config" / "model_config.json").read_text(encoding="utf-8"))
    config = ModelConfig(
        input_sequence_length=int(raw["input_sequence_length"]),
        input_dimension=int(raw["input_dimension"]),
        hidden_dimension=int(raw["hidden_dimension"]),
        num_layers=int(raw["num_layers"]),
        output_horizon=int(raw["output_horizon"]),
        dropout=float(raw["dropout"]),
    )
    checkpoint_path = bundle_root / "model" / raw["checkpoint_filename"]
    observed_hash = sha256(checkpoint_path)
    if observed_hash != EXPECTED_CHECKPOINT_SHA256 or observed_hash != raw["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {observed_hash}")

    model = C0LSTM(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing}, unexpected={unexpected}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Parameter count mismatch: {parameter_count}")
    if parameter_dtypes != ["torch.float32"]:
        raise RuntimeError(f"Model is not FP32: {parameter_dtypes}")
    model.eval()
    audit = {
        "checkpoint_sha256": observed_hash,
        "strict_state_dict_load": True,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "parameter_count": parameter_count,
        "parameter_dtypes": parameter_dtypes,
        "input_dimension": config.input_dimension,
        "sequence_length": config.input_sequence_length,
        "output_horizon": config.output_horizon,
        "model_training_mode": bool(model.training),
        "model_modified": False,
    }
    return model, raw, checkpoint_path, observed_hash, audit


def load_benchmark_arrays(bundle_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validation = bundle_root / "validation"
    values = np.load(validation / "benchmark_input.npy", allow_pickle=False)
    reference_normalized = np.load(validation / "benchmark_reference_output_normalized.npy", allow_pickle=False)
    reference_physical = np.load(validation / "benchmark_reference_output_physical.npy", allow_pickle=False)
    if values.shape != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(f"Benchmark input shape mismatch: {values.shape}")
    if reference_normalized.shape != EXPECTED_OUTPUT_SHAPE or reference_physical.shape != EXPECTED_OUTPUT_SHAPE:
        raise RuntimeError("Reference output shape mismatch")
    return values, reference_normalized, reference_physical


def comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    shape_match = reference.shape == candidate.shape
    if not shape_match:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    diff = candidate.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(diff)
    return {
        "shape_match": True,
        "shape": list(candidate.shape),
        "candidate_dtype": str(candidate.dtype),
        "candidate_nan_count": int(np.isnan(candidate).sum()),
        "candidate_inf_count": int(np.isinf(candidate).sum()),
        "max_absolute_difference": float(np.max(absolute)),
        "mean_absolute_difference": float(np.mean(absolute)),
        "rmse_difference": float(np.sqrt(np.mean(diff**2))),
    }


def run_consistency(bundle_root: Path, output: Path, device: torch.device) -> None:
    model, raw, checkpoint_path, checkpoint_hash, audit = load_bundle(bundle_root, device)
    values, reference_n, reference_p = load_benchmark_arrays(bundle_root)
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)
    with torch.no_grad():
        grad_enabled_inside = torch.is_grad_enabled()
        candidate_n = model(tensor).detach().cpu().numpy().astype(np.float32, copy=False)
    preprocessing = json.loads((bundle_root / "config" / "preprocessing.json").read_text(encoding="utf-8"))
    mean = float(preprocessing["electricity_normalization"]["mean"])
    std = float(preprocessing["electricity_normalization"]["population_std"])
    candidate_p = candidate_n.astype(np.float64) * std + mean
    normalized = comparison(reference_n, candidate_n)
    physical = comparison(reference_p, candidate_p)
    checks = {
        "shape_exact": normalized.get("shape_match") is True and physical.get("shape_match") is True,
        "nan_zero": normalized.get("candidate_nan_count") == 0 and physical.get("candidate_nan_count") == 0,
        "inf_zero": normalized.get("candidate_inf_count") == 0 and physical.get("candidate_inf_count") == 0,
        "normalized_max": normalized.get("max_absolute_difference", math.inf) <= CRITERIA["normalized_max_abs_diff_lte"],
        "normalized_mean": normalized.get("mean_absolute_difference", math.inf) <= CRITERIA["normalized_mean_abs_diff_lte"],
        "physical_max": physical.get("max_absolute_difference", math.inf) <= CRITERIA["physical_max_abs_diff_lte"],
        "physical_mean": physical.get("mean_absolute_difference", math.inf) <= CRITERIA["physical_mean_abs_diff_lte"],
        "model_eval": audit["model_training_mode"] is False,
        "no_grad": grad_enabled_inside is False,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "precision": "FP32",
        "batch_size": 1,
        "input_shape": list(values.shape),
        "output_shape": list(candidate_n.shape),
        "model_audit": audit,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "normalized": normalized,
        "physical": physical,
        "criteria": CRITERIA,
        "checks": checks,
        "autocast_used": False,
        "fp16_used": False,
        "onnx_used": False,
        "tensorrt_used": False,
        "torch_compile_used": False,
    }
    write_json_new(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}))
    if payload["status"] != "PASS":
        raise SystemExit(2)


def latency_statistics(values: list[float]) -> dict[str, float | int]:
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


def write_latency_csv(path: Path, rows: list[tuple[int, int, float]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run_id", "repetition", "latency_ms"])
        writer.writerows(rows)


def process_memory() -> dict[str, int | None]:
    if os.name == "nt":
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            return {"rss_current_bytes": None, "rss_peak_bytes": None}
        return {
            "rss_current_bytes": int(counters.WorkingSetSize),
            "rss_peak_bytes": int(counters.PeakWorkingSetSize),
        }

    values: dict[str, int | None] = {"rss_current_bytes": None, "rss_peak_bytes": None}
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                values["rss_current_bytes"] = int(line.split()[1]) * 1024
            elif line.startswith("VmHWM:"):
                values["rss_peak_bytes"] = int(line.split()[1]) * 1024
    return values


def run_latency(bundle_root: Path, output_dir: Path, device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("E5 latency requires CUDA")
    output_dir.mkdir(parents=True, exist_ok=True)
    model, raw, checkpoint_path, checkpoint_hash, audit = load_bundle(bundle_root, device)
    values, _, _ = load_benchmark_arrays(bundle_root)
    input_cpu = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
    input_cuda = input_cpu.to(device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    model_rows: list[tuple[int, int, float]] = []
    e2e_rows: list[tuple[int, int, float]] = []
    model_by_run: list[dict[str, float | int]] = []
    e2e_by_run: list[dict[str, float | int]] = []
    last_output_cpu: torch.Tensor | None = None

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(input_cuda)
        torch.cuda.synchronize(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for run_id in range(1, RUNS + 1):
            run_values: list[float] = []
            for repetition in range(1, REPETITIONS + 1):
                start_event.record()
                output_cuda = model(input_cuda)
                end_event.record()
                end_event.synchronize()
                latency_ms = float(start_event.elapsed_time(end_event))
                run_values.append(latency_ms)
                model_rows.append((run_id, repetition, latency_ms))
            model_by_run.append({"run_id": run_id, **latency_statistics(run_values)})
            if run_id < RUNS:
                time.sleep(STABILIZATION_SECONDS)

        for _ in range(WARMUP):
            warm_cuda = input_cpu.to(device)
            warm_output = model(warm_cuda)
            _ = warm_output.to("cpu")
        torch.cuda.synchronize(device)

        for run_id in range(1, RUNS + 1):
            run_values = []
            for repetition in range(1, REPETITIONS + 1):
                torch.cuda.synchronize(device)
                start_ns = time.perf_counter_ns()
                current_input = input_cpu.to(device)
                current_output = model(current_input)
                last_output_cpu = current_output.to("cpu")
                torch.cuda.synchronize(device)
                latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
                run_values.append(latency_ms)
                e2e_rows.append((run_id, repetition, latency_ms))
            e2e_by_run.append({"run_id": run_id, **latency_statistics(run_values)})
            if run_id < RUNS:
                time.sleep(STABILIZATION_SECONDS)

    if len(model_rows) != RUNS * REPETITIONS or len(e2e_rows) != RUNS * REPETITIONS:
        raise RuntimeError("Frozen repetition count was not completed")
    if last_output_cpu is None:
        raise RuntimeError("No end-to-end output was produced")
    last_array = last_output_cpu.numpy()
    if tuple(last_array.shape) != EXPECTED_OUTPUT_SHAPE or not np.isfinite(last_array).all():
        raise RuntimeError("Latency output failed shape/finite validation")

    model_path = output_dir / "latency_model_only_raw.csv"
    e2e_path = output_dir / "latency_end_to_end_raw.csv"
    write_latency_csv(model_path, model_rows)
    write_latency_csv(e2e_path, e2e_rows)
    model_values = [row[2] for row in model_rows]
    e2e_values = [row[2] for row in e2e_rows]
    latency_summary = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "runtime": f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}",
        "precision": "FP32",
        "batch_size": 1,
        "parameters": EXPECTED_PARAMETER_COUNT,
        "input_shape": list(EXPECTED_INPUT_SHAPE),
        "output_shape": list(last_array.shape),
        "warmup_per_mode": WARMUP,
        "runs": RUNS,
        "inferences_per_run": REPETITIONS,
        "stabilization_seconds_between_runs": STABILIZATION_SECONDS,
        "outliers_removed": False,
        "model_only_timing": "CUDA events; input already on CUDA; output retained on CUDA",
        "end_to_end_timing": "perf_counter_ns; CPU RAM to CUDA, forward, output to CPU, CUDA synchronization",
        "model_only": {
            "overall": latency_statistics(model_values),
            "by_run": model_by_run,
        },
        "end_to_end": {
            "overall": latency_statistics(e2e_values),
            "by_run": e2e_by_run,
        },
        "checkpoint_sha256": checkpoint_hash,
        "model_audit": audit,
        "nan_count": int(np.isnan(last_array).sum()),
        "inf_count": int(np.isinf(last_array).sum()),
    }
    write_json_new(output_dir / "latency_summary.json", latency_summary)

    memory = {
        "device": torch.cuda.get_device_name(0),
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        **process_memory(),
    }
    write_json_new(output_dir / "memory_summary.json", memory)
    print(json.dumps({"status": "PASS", "model_observations": len(model_rows), "e2e_observations": len(e2e_rows)}))


def telemetry_command(kind: str) -> tuple[list[str], str | None]:
    if kind == "jetson":
        executable = shutil.which("tegrastats") or "tegrastats"
        return [executable, "--interval", "1000"], None
    if kind == "rtx":
        executable = shutil.which("nvidia-smi") or "nvidia-smi"
        header = "timestamp,power_draw_w,temperature_gpu_c,utilization_gpu_percent,memory_used_mib,clocks_sm_mhz,clocks_mem_mhz\n"
        return [
            executable,
            "--query-gpu=timestamp,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
            "--loop-ms=1000",
        ], header
    raise ValueError(kind)


def run_telemetry(
    kind: str,
    phase: str,
    duration_seconds: float,
    raw_output: Path,
    run_output: Path,
    bundle_root: Path | None,
) -> None:
    if phase not in {"idle", "active", "cooldown"}:
        raise ValueError(phase)
    if raw_output.exists() or run_output.exists():
        raise FileExistsError("Refusing to overwrite telemetry output")
    raw_output.parent.mkdir(parents=True, exist_ok=True)

    model = None
    input_cuda = None
    if phase == "active":
        if bundle_root is None:
            raise ValueError("Active telemetry requires --bundle-root")
        device = torch.device("cuda")
        model, _, _, checkpoint_hash, audit = load_bundle(bundle_root, device)
        values, _, _ = load_benchmark_arrays(bundle_root)
        input_cuda = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)
        with torch.no_grad():
            for _ in range(WARMUP):
                _ = model(input_cuda)
        torch.cuda.synchronize(device)
    else:
        checkpoint_hash = None
        audit = None

    command, header = telemetry_command(kind)
    with raw_output.open("w", encoding="utf-8", newline="") as stream:
        if header:
            stream.write(header)
            stream.flush()
        sampler = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        sampler_start = time.perf_counter()
        iterations = 0
        last_array: np.ndarray | None = None
        try:
            if phase == "active":
                deadline = sampler_start + duration_seconds
                assert model is not None and input_cuda is not None
                with torch.no_grad():
                    while time.perf_counter() < deadline:
                        output_cuda = model(input_cuda)
                        torch.cuda.synchronize()
                        iterations += 1
                last_array = output_cuda.detach().cpu().numpy()
            else:
                time.sleep(duration_seconds)
        finally:
            sampler.terminate()
            try:
                sampler.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait(timeout=5)
    observed_seconds = time.perf_counter() - sampler_start
    if raw_output.stat().st_size == 0:
        raise RuntimeError("Telemetry sampler produced no output")
    nan_count = int(np.isnan(last_array).sum()) if last_array is not None else 0
    inf_count = int(np.isinf(last_array).sum()) if last_array is not None else 0
    if last_array is not None and (tuple(last_array.shape) != EXPECTED_OUTPUT_SHAPE or nan_count or inf_count):
        raise RuntimeError("Sustained inference output validation failed")
    payload = {
        "status": "PASS",
        "platform_kind": kind,
        "phase": phase,
        "requested_duration_seconds": duration_seconds,
        "observed_duration_seconds": observed_seconds,
        "sampler_command": command,
        "iterations_completed": iterations,
        "batch_size": 1 if phase == "active" else None,
        "precision": "FP32" if phase == "active" else None,
        "output_shape": list(last_array.shape) if last_array is not None else None,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "checkpoint_sha256": checkpoint_hash,
        "model_audit": audit,
    }
    write_json_new(run_output, payload)
    print(json.dumps({"status": "PASS", "phase": phase, "iterations": iterations, "seconds": observed_seconds}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    environment = sub.add_parser("environment")
    environment.add_argument("--kind", choices=("jetson", "rtx"), required=True)
    environment.add_argument("--output", type=Path, required=True)

    consistency = sub.add_parser("consistency")
    consistency.add_argument("--bundle-root", type=Path, required=True)
    consistency.add_argument("--output", type=Path, required=True)
    consistency.add_argument("--device", choices=("cuda",), default="cuda")

    latency = sub.add_parser("latency")
    latency.add_argument("--bundle-root", type=Path, required=True)
    latency.add_argument("--output-dir", type=Path, required=True)
    latency.add_argument("--device", choices=("cuda",), default="cuda")

    telemetry = sub.add_parser("telemetry")
    telemetry.add_argument("--kind", choices=("jetson", "rtx"), required=True)
    telemetry.add_argument("--phase", choices=("idle", "active", "cooldown"), required=True)
    telemetry.add_argument("--duration-seconds", type=float, default=60.0)
    telemetry.add_argument("--raw-output", type=Path, required=True)
    telemetry.add_argument("--run-output", type=Path, required=True)
    telemetry.add_argument("--bundle-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "environment":
        write_text_new(args.output, environment_report(args.kind))
        print(json.dumps({"status": "PASS", "output": str(args.output)}))
    elif args.command == "consistency":
        run_consistency(args.bundle_root.resolve(), args.output.resolve(), torch.device(args.device))
    elif args.command == "latency":
        run_latency(args.bundle_root.resolve(), args.output_dir.resolve(), torch.device(args.device))
    elif args.command == "telemetry":
        run_telemetry(
            args.kind,
            args.phase,
            args.duration_seconds,
            args.raw_output.resolve(),
            args.run_output.resolve(),
            args.bundle_root.resolve() if args.bundle_root else None,
        )
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
