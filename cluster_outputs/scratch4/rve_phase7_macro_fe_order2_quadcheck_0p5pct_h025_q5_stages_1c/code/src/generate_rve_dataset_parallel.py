"""Parallel, checkpoint-compatible RVE dataset generator.

Run from the project root, e.g.:
    python -m src.generate_rve_dataset_parallel --output data/rve_parallel_smoke.npz \
        --work-directory data/rve_parallel_smoke_work --volume-fractions 0.02 \
        --n-deformations 3 --seed 1729 --maxh 0.30 --workers 2

This script uses process-based parallelism. Each worker constructs and caches its
own NGSolve mesh; NGSolve mesh/solver objects are never passed between processes.
It is intentionally separate from generate_rve_dataset.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from .rve_dataset import (
    DatasetConfig,
    SCHEMA_VERSION,
    SPLIT_NAMES,
    TANGENT_FLATTENING,
    _atomic_json,
    _atomic_savez,
    _configuration_sha256,
    _config_payload,
    _file_sha256,
    _source_revision_sha256,
    assign_deformation_splits,
    generate_deformation_states,
    particle_radius,
    solve_dataset_sample,
    validate_solved_sample,
)


_WORKER_MESHES: dict[float, Any] = {}


def _load_valid_checkpoint(
    path: Path,
    gradient: np.ndarray,
    volume_fraction: float,
    config: DatasetConfig,
) -> dict[str, Any]:
    """Load a checkpoint only if its fields and numerical values are valid."""
    required_shapes = {
        "volume_fraction": (),
        "deformation_gradient": (3, 3),
        "energy": (),
        "stress": (3, 3),
        "tangent": (9, 9),
        "rotation": (3, 3),
        "determinant": (),
        "condition_number": (),
        "mixed_residual": (),
        "rotation_residual": (),
        "condensed_residual": (),
        "tangent_symmetry_error": (),
        "minimum_microscopic_jacobian": (),
        "maximum_microscopic_jacobian": (),
        "mixed_iterations": (),
        "rotation_iterations": (),
        "minimum_particle_clearance": (),
        "solver_status": (),
    }

    with np.load(path, allow_pickle=False) as data:
        missing = set(required_shapes).difference(data.files)
        if missing:
            raise ValueError(
                f"Checkpoint {path} is missing fields: {sorted(missing)}"
            )

        result = {key: np.array(data[key]) for key in required_shapes}

    for key, expected_shape in required_shapes.items():
        value = np.asarray(result[key])

        if value.shape != expected_shape:
            raise ValueError(
                f"Checkpoint {path}: field {key!r} has shape "
                f"{value.shape}; expected {expected_shape}."
            )

        if key == "solver_status":
            if value.dtype.kind not in ("U", "S") or str(value.item()) != "converged":
                raise ValueError(
                    f"Checkpoint {path}: solver_status is not 'converged'."
                )
            continue

        if not np.issubdtype(value.dtype, np.number):
            raise ValueError(
                f"Checkpoint {path}: field {key!r} is not numeric."
            )

        if not np.all(np.isfinite(value)):
            raise ValueError(
                f"Checkpoint {path}: field {key!r} contains nonfinite values."
            )

    saved_gradient = np.asarray(result["deformation_gradient"], dtype=float)
    saved_vf = float(result["volume_fraction"])

    if not np.allclose(saved_gradient, gradient, rtol=0, atol=1e-12):
        raise ValueError(
            f"Checkpoint {path} does not match the expected deformation gradient."
        )

    if not np.isclose(saved_vf, volume_fraction, rtol=0, atol=1e-12):
        raise ValueError(
            f"Checkpoint {path} does not match the expected volume fraction."
        )

    if float(result["determinant"]) <= 0:
        raise ValueError(f"Checkpoint {path} has a nonpositive determinant.")

    if float(result["condition_number"]) < 1:
        raise ValueError(f"Checkpoint {path} has an invalid condition number.")

    try:
        validate_solved_sample(result, config)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        raise ValueError(f"Checkpoint {path} failed sample validation: {exc}") from exc

    return result


def _worker_solve(task: dict[str, Any]) -> dict[str, Any]:
    """Run one sample in a child process, building the mesh inside that process."""
    try:
        from .rve import make_periodic_particle_cell_mesh

        vf = float(task["volume_fraction"])
        mesh_key = vf
        if mesh_key not in _WORKER_MESHES:
            _WORKER_MESHES[mesh_key] = make_periodic_particle_cell_mesh(
                radius=particle_radius(vf),
                maxh=float(task["config"].maxh),
            )
        mesh = _WORKER_MESHES[mesh_key]
        result = solve_dataset_sample(
            mesh=mesh,
            deformation_gradient=np.asarray(task["gradient"], dtype=float),
            volume_fraction=vf,
            config=task["config"],
        )
        checkpoint = Path(task["checkpoint"])
        _atomic_savez(checkpoint, **result)
        return {
            "ok": True,
            "sample_id": task["sample_id"],
            "volume_index": task["volume_index"],
            "state_index": task["state_index"],
            "checkpoint": str(checkpoint),
        }
    except Exception as exc:
        return {
            "ok": False,
            "sample_id": task.get("sample_id", "unknown"),
            "volume_index": task.get("volume_index", -1),
            "state_index": task.get("state_index", -1),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


_EXPECTED_GRADIENTS: dict[tuple[int, int], np.ndarray] = {}
_EXPECTED_VFS: list[float] = []


def generate_parallel_dataset(
    config: DatasetConfig,
    output: Path,
    work_directory: Path | None = None,
    workers: int = 2,
    resume: bool = True,
) -> Path:
    if workers < 1:
        raise ValueError("--workers must be at least 1")

    output = output.resolve()
    work_directory = (work_directory or output.with_name(output.stem + "_work")).resolve()
    samples_dir = work_directory / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    gradients = generate_deformation_states(config)
    splits = assign_deformation_splits(len(gradients), config.seed, config.split_fractions)
    gradient_hash = hashlib.sha256(np.ascontiguousarray(gradients).tobytes()).hexdigest()
    run_payload = {
        "schema_version": SCHEMA_VERSION,
        "config": _config_payload(config),
        "configuration_sha256": _configuration_sha256(config),
        "source_revision_sha256": _source_revision_sha256(),
        "deformation_sha256": gradient_hash,
    }
    run_path = work_directory / "run.json"
    if run_path.exists():
        existing = json.loads(run_path.read_text(encoding="utf-8"))
        if existing != run_payload:
            raise ValueError(
                f"Work directory {work_directory} belongs to a different run/configuration. "
                "Choose a new --work-directory or use the matching configuration."
            )
    else:
        _atomic_json(run_path, run_payload)

    global _EXPECTED_GRADIENTS, _EXPECTED_VFS
    _EXPECTED_VFS = [float(v) for v in config.volume_fractions]
    _EXPECTED_GRADIENTS = {
        (vi, si): np.asarray(gradients[si], dtype=float)
        for vi in range(len(config.volume_fractions))
        for si in range(len(gradients))
    }

    tasks = []
    checkpoint_paths: dict[tuple[int, int], Path] = {}
    for vi, vf in enumerate(config.volume_fractions):
        for si, gradient in enumerate(gradients):
            sample_id = f"v{vi:03d}_f{si:06d}"
            checkpoint = samples_dir / f"{sample_id}.npz"
            checkpoint_paths[(vi, si)] = checkpoint
            if resume and checkpoint.exists():
                _load_valid_checkpoint(checkpoint, gradient, float(vf), config)
                continue
            tasks.append({
                "sample_id": sample_id,
                "volume_index": vi,
                "state_index": si,
                "gradient": np.asarray(gradient, dtype=float),
                "volume_fraction": float(vf),
                "config": config,
                "checkpoint": str(checkpoint),
            })

    print(f"Dataset samples: {len(config.volume_fractions) * len(gradients)}", flush=True)
    print(f"Existing valid checkpoints: {len(checkpoint_paths) - len(tasks)}", flush=True)
    print(f"Samples to solve: {len(tasks)} using {workers} worker processes", flush=True)

    failures: list[dict[str, Any]] = []
    completed = 0
    if tasks:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(_worker_solve, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "ok": False,
                        "sample_id": task["sample_id"],
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                if result.get("ok"):
                    completed += 1
                    print(f"[{completed}/{len(tasks)}] completed {result['sample_id']}", flush=True)
                else:
                    failures.append(result)
                    print(f"[FAILED] {result.get('sample_id')}: {result.get('error')}", flush=True)

    # Validate every expected checkpoint before assembling. A failed run keeps all
    # successful checkpoints and can be resumed with the same command.
    missing_or_invalid = []
    records = []
    for vi, vf in enumerate(config.volume_fractions):
        for si in range(len(gradients)):
            checkpoint = checkpoint_paths[(vi, si)]
            try:
                raw = _load_valid_checkpoint(
                    checkpoint,
                    gradients[si],
                    float(vf),
                    config,
                )
                raw["sample_id"] = np.asarray(f"v{vi:03d}_f{si:06d}")
                raw["volume_index"] = np.asarray(vi, dtype=np.int32)
                raw["state_index"] = np.asarray(si, dtype=np.int32)
                raw["split"] = np.asarray(str(splits[si]))
                records.append(raw)
            except Exception as exc:
                missing_or_invalid.append({
                    "sample_id": f"v{vi:03d}_f{si:06d}",
                    "error": f"{type(exc).__name__}: {exc}",
                })

    if failures or missing_or_invalid:
        failure_path = output.with_suffix(".failures.json")
        _atomic_json(failure_path, {
            "failed_samples": failures,
            "missing_or_invalid_checkpoints": missing_or_invalid,
        })
        print(f"Dataset not assembled. Checkpoints were retained. Details: {failure_path}", file=sys.stderr)
        return output

    keys = records[0].keys()
    dataset = {key: np.stack([record[key] for record in records]) for key in keys}
    _atomic_savez(output, **dataset)

    split_counts = {
        name: int(np.count_nonzero(dataset["split"] == name))
        for name in SPLIT_NAMES
    }
    manifest = {
        **run_payload,
        "output": str(output),
        "n_samples": int(len(records)),
        "dataset_sha256": _file_sha256(output),
        "tangent_flattening": TANGENT_FLATTENING,
        "fields": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in dataset.items()
        },
        "split_counts": split_counts,
        "ngsolve_version": _ngsolve_version(),
    }
    _atomic_json(output.with_suffix(".manifest.json"), manifest)
    print(f"Dataset saved: {output}", flush=True)
    print(f"Manifest saved: {output.with_suffix('.manifest.json')}", flush=True)
    return output


def _ngsolve_version() -> str:
    try:
        import ngsolve
        return str(getattr(ngsolve, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-directory", type=Path, default=None)
    parser.add_argument("--volume-fractions", type=float, nargs="+", default=(0.02, 0.05, 0.10))
    parser.add_argument(
        "--n-deformations",
        type=int,
        default=100,
        help="total number of deformation states, including identity and targeted tails",
    )
    parser.add_argument(
        "--split-fractions",
        type=float,
        nargs=3,
        default=(0.70, 0.15, 0.15),
        metavar=("TRAIN", "VALIDATION", "TEST"),
        help="fractions of deformation states assigned to each leakage-free split",
    )
    parser.add_argument(
        "--n-compression-deformations",
        type=int,
        default=0,
        help=(
            "number of states whose macroscopic "
            "determinants are sampled from --compression-determinant-range"
        ),
    )
    parser.add_argument(
        "--compression-determinant-range",
        type=float,
        nargs=2,
        default=(0.52, 0.70),
        metavar=("LOW", "HIGH"),
        help="target determinant interval for compression states",
    )
    parser.add_argument(
        "--n-expansion-deformations",
        type=int,
        default=0,
        help=(
            "number of states whose macroscopic "
            "determinants are sampled from --expansion-determinant-range"
        ),
    )
    parser.add_argument(
        "--expansion-determinant-range",
        type=float,
        nargs=2,
        default=(1.50, 1.80),
        metavar=("LOW", "HIGH"),
        help="target determinant interval for expansion states",
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-strain", type=float, default=0.20)
    parser.add_argument("--determinant-floor", type=float, default=0.50)
    parser.add_argument("--max-condition-number", type=float, default=3.0)
    parser.add_argument("--maxh", type=float, default=0.22)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--bulk-modulus", type=float, default=1.0e4)
    parser.add_argument("--displacement-order", type=int, default=2)
    parser.add_argument("--integration-order", type=int, default=10)
    parser.add_argument(
        "--microscopic-jacobian-floor",
        type=float,
        default=1.0e-8,
        help="reject samples whose minimum quadrature-point microscopic J is at or below this value",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = DatasetConfig(
        volume_fractions=tuple(args.volume_fractions),
        n_deformations=args.n_deformations,
        n_compression_deformations=args.n_compression_deformations,
        compression_determinant_range=tuple(args.compression_determinant_range),
        n_expansion_deformations=args.n_expansion_deformations,
        expansion_determinant_range=tuple(args.expansion_determinant_range),
        split_fractions=tuple(args.split_fractions),
        seed=args.seed,
        max_strain=args.max_strain,
        determinant_floor=args.determinant_floor,
        max_condition_number=args.max_condition_number,
        maxh=args.maxh,
        mu=args.mu,
        bulk_modulus=args.bulk_modulus,
        displacement_order=args.displacement_order,
        integration_order=args.integration_order,
        microscopic_jacobian_floor=args.microscopic_jacobian_floor,
    )
    output = generate_parallel_dataset(
        config=config,
        output=args.output,
        work_directory=args.work_directory,
        workers=args.workers,
        resume=not args.no_resume,
    )
    if not output.with_suffix(".manifest.json").is_file():
        raise SystemExit(
            "Dataset is incomplete; inspect the .failures.json report and "
            "resume from the retained checkpoints."
        )


if __name__ == "__main__":
    main()
