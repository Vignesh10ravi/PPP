"""Compare a frozen and candidate pICNN on a held-out RVE shear set."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .train_rve import _load_rve_dataset
from .train_rve_shear_context_adapter import (
    _baseline_scales,
    _load_checkpoint,
    _relative_shear_metrics,
    _to_tensors,
)


EXPECTED_AMPLITUDES = (0.035, 0.065, 0.085, 0.10)
PREVIOUSLY_USED_AMPLITUDES = {
    0.01, 0.02, 0.025, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07,
    0.075, 0.08, 0.09, 0.095,
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_holdout(
    dataset_path, manifest_path, expected_amplitudes, excluded_amplitudes
):
    dataset_path = Path(dataset_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    dataset_hash = _sha256_file(dataset_path)
    if manifest.get("dataset_sha256") != dataset_hash:
        raise ValueError("holdout dataset SHA-256 does not match its manifest")
    with np.load(dataset_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name]) for name in archive.files}

    design = manifest.get("deformation_design", {})
    if not design.get("evaluation_only"):
        raise ValueError("holdout manifest does not mark all records evaluation-only")
    if not design.get("includes_identity_reference"):
        raise ValueError("holdout is missing its identity reference state")
    if manifest.get("n_samples") != len(arrays.get("sample_id", [])):
        raise ValueError("holdout sample count does not match its manifest")

    gradients = np.asarray(arrays["deformation_gradient"], dtype=float)
    amplitudes = gradients[:, 0, 1]
    state_indices = np.asarray(arrays["state_index"], dtype=int)
    identity = state_indices == 0
    if not np.allclose(gradients[identity], np.eye(3), rtol=0.0, atol=1.0e-12):
        raise ValueError("state zero in holdout is not the identity reference")
    if not np.all(np.asarray(arrays["split"]).astype(str) == "test"):
        raise ValueError("all holdout records must be assigned to the test split")
    if not np.all(np.asarray(arrays["solver_status"]).astype(str) == "converged"):
        raise ValueError("holdout includes a nonconverged direct-RVE sample")
    if np.any(np.asarray(arrays["minimum_microscopic_jacobian"]) <= 1.0e-8):
        raise ValueError("holdout includes a sample below the microscopic-J floor")

    measured = sorted(set(np.round(amplitudes[~identity], 12).tolist()))
    expected = sorted(np.round(np.asarray(expected_amplitudes), 12).tolist())
    if measured != expected:
        raise ValueError(f"unexpected holdout amplitudes: {measured}")
    if excluded_amplitudes.intersection(measured):
        raise ValueError("holdout amplitudes overlap prior shear data")
    if not np.all(np.linalg.det(gradients) > 0.5):
        raise ValueError("holdout contains a non-positive or unsafe macro determinant")

    return arrays, manifest, dataset_hash, np.flatnonzero(~identity)


def _summarize_rows(rows):
    p01 = np.asarray([row["P01_relative_error"] for row in rows], dtype=float)
    tangent = np.asarray(
        [row["dP01_dF01_relative_error"] for row in rows], dtype=float
    )
    stress = np.asarray(
        [row["stress_frobenius_relative_error"] for row in rows], dtype=float
    )
    energy = np.asarray([row["energy_relative_error"] for row in rows], dtype=float)
    return {
        "n_samples": len(rows),
        "mean_energy_relative_error": float(energy.mean()),
        "mean_P01_relative_error": float(p01.mean()),
        "maximum_P01_relative_error": float(p01.max()),
        "mean_dP01_dF01_relative_error": float(tangent.mean()),
        "maximum_dP01_dF01_relative_error": float(tangent.max()),
        "stress_frobenius_relative_rmse": float(np.sqrt(np.mean(stress**2))),
    }


def _group_rows(rows, key):
    values = sorted({row[key] for row in rows})
    return {
        str(value): _summarize_rows([row for row in rows if row[key] == value])
        for value in values
    }


def evaluate(args):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)

    arrays, manifest, dataset_hash, indices = _load_holdout(
        args.dataset,
        args.manifest,
        tuple(args.expected_amplitudes),
        set(args.excluded_amplitudes).union(PREVIOUSLY_USED_AMPLITUDES),
    )
    adaptation, adaptation_manifest, adaptation_hash = _load_rve_dataset(
        args.adaptation_dataset, args.adaptation_manifest
    )
    if adaptation_manifest.get("dataset_sha256") != adaptation_hash:
        raise ValueError("adaptation dataset hash does not match its manifest")
    same_microstructure_seed = (
        manifest["config"]["seed"] == adaptation_manifest["config"]["seed"]
    )
    if not same_microstructure_seed and not args.allow_different_microstructure_seed:
        raise ValueError("holdout and adaptation data use different microstructure seeds")

    frozen, frozen_payload, frozen_hash = _load_checkpoint(
        args.frozen_checkpoint, args.expected_frozen_sha256
    )
    candidate, candidate_payload, candidate_hash = _load_checkpoint(
        args.candidate_checkpoint, args.expected_candidate_sha256
    )
    scales = _baseline_scales(adaptation, frozen_payload)

    selected_arrays = {key: value[indices] for key, value in arrays.items()}
    selected_tensors = _to_tensors(selected_arrays)
    frozen_summary, frozen_rows = _relative_shear_metrics(
        frozen, selected_tensors, selected_arrays, scales
    )
    candidate_summary, candidate_rows = _relative_shear_metrics(
        candidate, selected_tensors, selected_arrays, scales
    )
    if len(frozen_rows) != len(candidate_rows):
        raise RuntimeError("frozen and candidate evaluations returned different row counts")
    paired_rows = []
    for before, after in zip(frozen_rows, candidate_rows):
        if (
            before["shear_amplitude"] != after["shear_amplitude"]
            or before["volume_fraction"] != after["volume_fraction"]
            or before["P01_direct_rve"] != after["P01_direct_rve"]
            or before["dP01_dF01_direct_rve"] != after["dP01_dF01_direct_rve"]
        ):
            raise RuntimeError("paired evaluation rows or direct-RVE labels differ")
        paired_rows.append({
            "shear_amplitude": after["shear_amplitude"],
            "volume_fraction": after["volume_fraction"],
            "frozen_P01_relative_error": before["P01_relative_error"],
            "candidate_P01_relative_error": after["P01_relative_error"],
            "frozen_dP01_dF01_relative_error": before[
                "dP01_dF01_relative_error"
            ],
            "candidate_dP01_dF01_relative_error": after[
                "dP01_dF01_relative_error"
            ],
            "frozen_stress_frobenius_relative_error": before[
                "stress_frobenius_relative_error"
            ],
            "candidate_stress_frobenius_relative_error": after[
                "stress_frobenius_relative_error"
            ],
            "energy_direct_rve": after["energy_direct_rve"],
            "energy_frozen": before["energy_pICNN"],
            "energy_candidate": after["energy_pICNN"],
            "P01_direct_rve": after["P01_direct_rve"],
            "P01_frozen": before["P01_pICNN"],
            "P01_candidate": after["P01_pICNN"],
            "dP01_dF01_direct_rve": after["dP01_dF01_direct_rve"],
            "dP01_dF01_frozen": before["dP01_dF01_pICNN"],
            "dP01_dF01_candidate": after["dP01_dF01_pICNN"],
        })

    candidate_label = args.candidate_label
    scope = (
        "All nonidentity states were generated after model selection, at shear "
        "amplitudes excluded from all earlier shear datasets."
    )
    if same_microstructure_seed:
        scope += (
            " The adaptation microstructure seed is held fixed, so this tests "
            "unseen deformation states rather than new microstructure realizations."
        )
    else:
        scope += (
            " A new microstructure seed was used, so this tests both unseen "
            "deformation states and a new RVE realization."
        )
    report = {
        "status": "complete",
        "independent_evaluation": True,
        "scope": scope,
        "holdout_dataset_sha256": dataset_hash,
        "adaptation_dataset_sha256": adaptation_hash,
        "frozen_checkpoint_sha256": frozen_hash,
        "candidate_checkpoint_sha256": candidate_hash,
        "candidate_checkpoint": str(Path(args.candidate_checkpoint).resolve()),
        "candidate_label": candidate_label,
        "microstructure_seed": manifest["config"]["seed"],
        "same_microstructure_seed_as_adaptation": bool(
            manifest["config"]["seed"] == adaptation_manifest["config"]["seed"]
        ),
        "amplitudes": list(args.expected_amplitudes),
        "volume_fractions": sorted(
            set(np.asarray(arrays["volume_fraction"][indices], dtype=float).tolist())
        ),
        "identity_reference_records_excluded_from_error_metrics": int(
            np.count_nonzero(np.asarray(arrays["state_index"]) == 0)
        ),
        "overall": {
            "frozen_checkpoint": _summarize_rows(frozen_rows),
            candidate_label: _summarize_rows(candidate_rows),
        },
        "by_volume_fraction": {
            "frozen_checkpoint": _group_rows(frozen_rows, "volume_fraction"),
            candidate_label: _group_rows(candidate_rows, "volume_fraction"),
        },
        "by_shear_amplitude": {
            "frozen_checkpoint": _group_rows(frozen_rows, "shear_amplitude"),
            candidate_label: _group_rows(candidate_rows, "shear_amplitude"),
        },
        "paired_points": paired_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({
        "status": report["status"],
        "independent_evaluation": report["independent_evaluation"],
        "n_samples": report["overall"][candidate_label]["n_samples"],
        "frozen": report["overall"]["frozen_checkpoint"],
        "candidate": report["overall"][candidate_label],
        "report": str(output.resolve()),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adaptation-dataset", type=Path, required=True)
    parser.add_argument("--adaptation-manifest", type=Path, required=True)
    parser.add_argument("--frozen-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-frozen-sha256", required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--expected-amplitudes", type=float, nargs="+", default=EXPECTED_AMPLITUDES
    )
    parser.add_argument(
        "--excluded-amplitudes", type=float, nargs="*", default=()
    )
    parser.add_argument("--allow-different-microstructure-seed", action="store_true")
    parser.add_argument("--candidate-label", default="context_adapter")
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("torch-threads must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
