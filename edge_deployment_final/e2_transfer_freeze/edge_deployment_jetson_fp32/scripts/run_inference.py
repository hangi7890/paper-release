"""Run standalone inference using only files within the E1 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from model_definition import C0LSTM, ModelConfig, count_parameters
from preprocessing import inverse_transform


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def load_bundle_model(bundle_root: Path, device: torch.device):
    model_config = json.loads((bundle_root / "config" / "model_config.json").read_text(encoding="utf-8"))
    checkpoint_path = bundle_root / "model" / model_config["checkpoint_filename"]
    observed_hash = sha256(checkpoint_path)
    if observed_hash != model_config["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {observed_hash}")

    config = ModelConfig(
        input_sequence_length=int(model_config["input_sequence_length"]),
        input_dimension=int(model_config["input_dimension"]),
        hidden_dimension=int(model_config["hidden_dimension"]),
        num_layers=int(model_config["num_layers"]),
        output_horizon=int(model_config["output_horizon"]),
        dropout=float(model_config["dropout"]),
    )
    model = C0LSTM(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("Expected metadata dictionary containing model_state_dict")
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"state_dict mismatch: {incompatible}")
    if count_parameters(model) != int(model_config["parameter_count"]):
        raise RuntimeError("Parameter count does not match model_config")
    model.eval()
    return model, model_config, checkpoint_path, observed_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    bundle_root = Path(__file__).resolve().parents[1]
    input_path = args.input if args.input.is_absolute() else bundle_root / args.input
    output_dir = args.output_dir if args.output_dir.is_absolute() else bundle_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    device = select_device(args.device)
    model, model_config, checkpoint_path, checkpoint_hash = load_bundle_model(bundle_root, device)

    values = np.load(input_path, allow_pickle=False)
    expected_tail = (int(model_config["input_sequence_length"]), int(model_config["input_dimension"]))
    if values.ndim != 3 or tuple(values.shape[1:]) != expected_tail:
        raise ValueError(f"Expected input shape (*,{expected_tail[0]},{expected_tail[1]}), got {values.shape}")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("Input must be finite numeric values")

    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)
    with torch.no_grad():
        normalized = model(tensor).detach().cpu().numpy().astype(np.float32, copy=False)
    expected_output = (values.shape[0], int(model_config["output_horizon"]))
    if normalized.shape != expected_output:
        raise RuntimeError(f"Unexpected model output shape: {normalized.shape}")

    preprocessing = json.loads((bundle_root / "config" / "preprocessing.json").read_text(encoding="utf-8"))
    physical = inverse_transform(
        normalized,
        float(preprocessing["electricity_normalization"]["mean"]),
        float(preprocessing["electricity_normalization"]["population_std"]),
    )
    nan_count = int(np.isnan(normalized).sum() + np.isnan(physical).sum())
    inf_count = int(np.isinf(normalized).sum() + np.isinf(physical).sum())
    if nan_count or inf_count:
        raise RuntimeError(f"Non-finite output: NaN={nan_count}, Inf={inf_count}")

    normalized_path = output_dir / "output_normalized.npy"
    physical_path = output_dir / "output_physical.npy"
    np.save(normalized_path, normalized, allow_pickle=False)
    np.save(physical_path, physical, allow_pickle=False)
    metadata = {
        "status": "PASS",
        "bundle_root": str(bundle_root),
        "input_path": str(input_path.resolve()),
        "input_shape": list(values.shape),
        "input_dtype": str(values.dtype),
        "output_shape": list(normalized.shape),
        "normalized_output_dtype": str(normalized.dtype),
        "physical_output_dtype": str(physical.dtype),
        "device_requested": args.device,
        "device_used": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "CPU",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "training_performed": False,
        "adaptation_performed": False,
    }
    (output_dir / "inference_run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()

