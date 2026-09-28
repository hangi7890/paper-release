"""Create and verify the final_v1 artifact size/SHA-256 manifest."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__import__("os").environ.get("PAPER_WORKSPACE", str(Path(__file__).resolve().parents[2]))).resolve()
REPORTS = ROOT / "reports"
SUPPORT = REPORTS / "final_v1"
SCRIPTS = Path(__file__).resolve().parents[1] / "final_v1"
MODELS = ROOT / "models" / "final_v1"
OUT = REPORTS / "final_v1_artifact_manifest.csv"
VALIDATION_LOG = SUPPORT / "artifact_manifest_validation.log"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect() -> list[Path]:
    files: set[Path] = set()
    for path in REPORTS.glob("final_v1*"):
        if path.is_file() and path != OUT:
            files.add(path)
    if SUPPORT.exists():
        files.update(path for path in SUPPORT.rglob("*") if path.is_file())
    for base in (SCRIPTS, MODELS):
        if base.exists():
            files.update(
                path
                for path in base.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
            )
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def category(path: Path) -> str:
    relative = path.relative_to(ROOT)
    if relative.parts[0] == "models":
        return "adaptation_checkpoint"
    if relative.parts[0] == "scripts":
        return "code_or_test"
    if "figures" in relative.parts:
        return "figure"
    if path.suffix.lower() == ".log":
        return "execution_or_test_log"
    if path.suffix.lower() == ".csv":
        return "data_or_manifest"
    if path.suffix.lower() == ".md":
        return "report"
    if path.suffix.lower() == ".json":
        return "configuration_or_marker"
    return "other"


def build_frame(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        stat = path.stat()
        rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "category": category(path),
                "size_bytes": stat.st_size,
                "sha256": sha256(path),
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def validate(frame: pd.DataFrame) -> None:
    if frame.relative_path.duplicated().any():
        raise AssertionError("Duplicate artifact path")
    for row in frame.itertuples(index=False):
        path = ROOT / Path(row.relative_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size != row.size_bytes:
            raise AssertionError(f"Size changed: {path}")
        if sha256(path) != row.sha256:
            raise AssertionError(f"Hash changed: {path}")


def main() -> None:
    if OUT.exists() or VALIDATION_LOG.exists():
        raise FileExistsError("Artifact manifest or its validation log already exists")
    initial = build_frame(collect())
    initial.to_csv(OUT, index=False, encoding="utf-8", lineterminator="\n")
    validate(initial)
    VALIDATION_LOG.write_text(
        "\n".join(
            [
                f"validated_utc={datetime.now(timezone.utc).isoformat()}",
                "status=PASS",
                "scope=all final_v1 artifacts except the self-referential artifact manifest itself",
                f"initial_validated_files={len(initial)}",
                "method=size_and_sha256_recomputed_after_manifest_write",
                "note=the validation log is added to the final manifest and the final rows are validated again",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    final = build_frame(collect())
    final.to_csv(OUT, index=False, encoding="utf-8", lineterminator="\n")
    validate(final)
    print(f"PASS artifact manifest files={len(final)} manifest_sha256={sha256(OUT)}")


if __name__ == "__main__":
    main()
