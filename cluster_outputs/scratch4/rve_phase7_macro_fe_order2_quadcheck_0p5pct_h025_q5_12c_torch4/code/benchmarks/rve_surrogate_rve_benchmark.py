"""Compare frozen pICNN evaluation with direct RVE solves on a fresh test set.

Selects whole deformation groups from the held-out test split, preserving all
available volume fractions for each selected group. The script does not fit or
select the model. Direct-RVE solver timing includes free-rotation equilibrium
and energy/stress recovery on a reused mesh; one-time mesh construction is
reported separately and amortized across the selected cases. Surrogate timing
includes energy and full first-Piola stress evaluation, with its reference
state setup measured and amortized separately.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.func import jacrev

from benchmarks.rve_inverse_design import _evaluate_direct_rve, _load_model


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _select_test_groups(data, requested_states):
    splits = np.asarray(data["split"]).astype(str)
    state_indices = np.asarray(data["state_index"], dtype=int)
    gradients = np.asarray(data["deformation_gradient"], dtype=float)
    volume_fractions = np.asarray(data["volume_fraction"], dtype=float)
    expected_volume_fractions = np.unique(volume_fractions[splits == "test"])
    test_states = np.unique(state_indices[splits == "test"])
    test_states = test_states[test_states != 0]
    if len(test_states) < requested_states:
        raise ValueError(
            f"test split contains {len(test_states)} non-reference groups, "
            f"fewer than requested {requested_states}"
        )

    state_info = []
    for state_index in test_states:
        indices = np.flatnonzero(
            (splits == "test") & (state_indices == state_index)
        )
        if len(indices) == 0:
            continue
        reference_F = gradients[indices[0]]
        if any(
            not np.allclose(gradients[index], reference_F, rtol=0.0, atol=1.0e-12)
            for index in indices[1:]
        ):
            raise ValueError(f"deformation state {state_index} is not grouped consistently")
        group_vfs = np.sort(volume_fractions[indices])
        if len(np.unique(group_vfs)) != len(group_vfs):
            raise ValueError(f"deformation state {state_index} has duplicate volume fractions")
        if not np.array_equal(group_vfs, expected_volume_fractions):
            raise ValueError(
                f"deformation state {state_index} does not contain every test "
                "volume fraction"
            )
        state_info.append({
            "state_index": int(state_index),
            "indices": indices,
            "F": reference_F,
            "determinant": float(np.linalg.det(reference_F)),
        })

    state_info.sort(key=lambda item: (item["determinant"], item["state_index"]))
    positions = np.rint(
        np.linspace(0, len(state_info) - 1, requested_states)
    ).astype(int)
    selected = [state_info[index] for index in positions]
    if len({item["state_index"] for item in selected}) != requested_states:
        raise RuntimeError("deterministic test-group selection produced duplicates")
    return selected


def _predict(model, F_np, volume_fraction, reference):
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    F = torch.as_tensor(F_np, dtype=dtype, device=device)
    context = torch.as_tensor(
        [[float(volume_fraction)]],
        dtype=dtype,
        device=device,
    )

    def energy_single(gradient):
        return model.energy(
            gradient.unsqueeze(0),
            context,
            ref=reference,
        ).squeeze()

    energy = energy_single(F)
    stress = jacrev(energy_single)(F)
    return (
        float(energy.detach().cpu()),
        stress.detach().cpu().numpy(),
    )


def _error_summary(errors, energy_scale, stress_scale):
    energy_errors = np.asarray([item["energy_error"] for item in errors], dtype=float)
    stress_errors = np.asarray([item["stress_error"] for item in errors], dtype=float)
    stress_scaled = stress_errors / stress_scale
    stress_per_case = np.sqrt(np.mean(stress_scaled**2, axis=(1, 2)))
    return {
        "n_cases": int(len(errors)),
        "energy_rmse": float(np.sqrt(np.mean(energy_errors**2))),
        "energy_mae": float(np.mean(np.abs(energy_errors))),
        "energy_nrmse_training_scale": float(
            np.sqrt(np.mean((energy_errors / energy_scale) ** 2))
        ),
        "stress_component_rmse": float(np.sqrt(np.mean(stress_errors**2))),
        "stress_component_mae": float(np.mean(np.abs(stress_errors))),
        "stress_nrmse_training_scale": float(
            np.sqrt(np.mean(stress_scaled**2))
        ),
        "stress_case_rmse_nrmse_training_scale": float(
            np.sqrt(np.mean(stress_per_case**2))
        ),
    }


def _runtime_summary(
    records,
    surrogate_setup_seconds,
    direct_mesh_setup_seconds,
):
    surrogate_case_medians = np.asarray(
        [np.median(item["surrogate_seconds"]) for item in records],
        dtype=float,
    )
    direct_case_medians = np.asarray(
        [np.median(item["direct_rve_seconds"]) for item in records],
        dtype=float,
    )
    paired_speedups = direct_case_medians / surrogate_case_medians
    case_count = max(1, len(records))
    surrogate_amortized = surrogate_case_medians + (
        surrogate_setup_seconds / case_count
    )
    direct_amortized = direct_case_medians + (
        sum(direct_mesh_setup_seconds.values()) / case_count
    )
    amortized_speedups = direct_amortized / surrogate_amortized
    return {
        "n_cases": int(len(records)),
        "surrogate_median_seconds_per_case": float(np.median(surrogate_case_medians)),
        "direct_rve_median_solver_seconds_per_case": float(
            np.median(direct_case_medians)
        ),
        "median_paired_solver_speedup": float(np.median(paired_speedups)),
        "p10_paired_solver_speedup": float(np.quantile(paired_speedups, 0.10)),
        "p90_paired_solver_speedup": float(np.quantile(paired_speedups, 0.90)),
        "surrogate_reference_setup_seconds_total": float(surrogate_setup_seconds),
        "direct_mesh_setup_seconds_by_volume_fraction": {
            str(key): float(value)
            for key, value in direct_mesh_setup_seconds.items()
        },
        "direct_mesh_setup_seconds_total": float(
            sum(direct_mesh_setup_seconds.values())
        ),
        "surrogate_median_amortized_seconds_per_case": float(
            np.median(surrogate_amortized)
        ),
        "direct_rve_median_amortized_seconds_per_case": float(
            np.median(direct_amortized)
        ),
        "median_paired_amortized_speedup": float(np.median(amortized_speedups)),
        "p10_paired_amortized_speedup": float(np.quantile(amortized_speedups, 0.10)),
        "p90_paired_amortized_speedup": float(np.quantile(amortized_speedups, 0.90)),
        "surrogate_timing_basis": (
            "median of repeated energy plus full first-Piola stress evaluations; "
            "reference-state setup is timed separately and amortized for the "
            "end-to-end comparison"
        ),
        "direct_rve_timing_basis": (
            "median of repeated free-rotation equilibrium, energy, and full "
            "first-Piola stress recovery on a reused mesh; mesh construction is "
            "timed separately and amortized for the end-to-end comparison"
        ),
    }


def _polar_rotation(deformation_gradient):
    left, _, right_transpose = np.linalg.svd(deformation_gradient)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return rotation


def run_benchmark(args):
    dataset_path = Path(args.dataset)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_sha256 = _sha256_file(dataset_path)
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("holdout dataset SHA-256 does not match its manifest")

    with np.load(dataset_path, allow_pickle=False) as archive:
        required = {
            "sample_id",
            "split",
            "state_index",
            "volume_fraction",
            "deformation_gradient",
            "energy",
            "stress",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"holdout dataset is missing fields: {sorted(missing)}")
        data = {name: np.array(archive[name]) for name in required}

    checkpoint_path = Path(args.checkpoint)
    model, checkpoint, checkpoint_sha256 = _load_model(checkpoint_path)
    if args.expected_checkpoint_sha256:
        if checkpoint_sha256 != args.expected_checkpoint_sha256:
            raise ValueError("checkpoint SHA-256 does not match the frozen expected hash")
    if checkpoint.get("dataset_sha256") == dataset_sha256:
        raise ValueError("benchmark dataset must be independent of the training dataset")

    if data["deformation_gradient"].shape != (len(data["sample_id"]), 3, 3):
        raise ValueError("deformation_gradient must have shape (N, 3, 3)")
    if data["stress"].shape != (len(data["sample_id"]), 3, 3):
        raise ValueError("stress must have shape (N, 3, 3)")
    if not all(
        np.all(np.isfinite(data[name]))
        for name in ("deformation_gradient", "volume_fraction", "energy", "stress")
    ):
        raise ValueError("holdout data contains nonfinite values")

    selected_groups = _select_test_groups(data, args.states)
    selected_indices = np.concatenate([group["indices"] for group in selected_groups])
    selected_indices = selected_indices[
        np.argsort(
            np.asarray([
                (
                    int(data["state_index"][index]),
                    float(data["volume_fraction"][index]),
                )
                for index in selected_indices
            ], dtype=[("state", int), ("vf", float)]),
            order=("state", "vf"),
        )
    ]

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    model = model.to(device=device, dtype=torch.float64)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    vf_values = np.unique(data["volume_fraction"][selected_indices])
    references = {}
    surrogate_setup_seconds = 0.0
    for vf in vf_values:
        context = torch.as_tensor(
            [[float(vf)]],
            dtype=torch.float64,
            device=device,
        )
        setup_start = time.perf_counter()
        references[float(vf)] = model.reference(torch.float64, device, context)
        surrogate_setup_seconds += time.perf_counter() - setup_start

    from src.rve import make_periodic_particle_cell_mesh
    from src.rve_dataset import particle_radius

    mesh_by_vf = {}
    mesh_setup_seconds = {}
    for vf in vf_values:
        setup_start = time.perf_counter()
        mesh_by_vf[float(vf)] = make_periodic_particle_cell_mesh(
            radius=particle_radius(float(vf)),
            maxh=args.maxh,
        )
        mesh_setup_seconds[float(vf)] = time.perf_counter() - setup_start

    cases = []
    for index in selected_indices:
        cases.append({
            "dataset_index": int(index),
            "sample_id": str(data["sample_id"][index]),
            "state_index": int(data["state_index"][index]),
            "split": str(data["split"][index]),
            "volume_fraction": float(data["volume_fraction"][index]),
            "deformation_gradient": np.asarray(
                data["deformation_gradient"][index],
                dtype=float,
            ),
            "energy_label": float(data["energy"][index]),
            "stress_label": np.asarray(data["stress"][index], dtype=float),
        })

    # Warm up the surrogate and finite-element runtime before collecting timings.
    _predict(
        model,
        cases[0]["deformation_gradient"],
        cases[0]["volume_fraction"],
        references[cases[0]["volume_fraction"]],
    )
    rve_args = Namespace(
        maxh=args.maxh,
        mu=args.mu,
        bulk_modulus=args.bulk_modulus,
        displacement_order=args.displacement_order,
        integration_order=args.integration_order,
        microscopic_jacobian_floor=args.microscopic_jacobian_floor,
        response="energy",
        i=0,
        j=0,
        rve_verbose=False,
    )
    print("[BENCHMARK] direct RVE warm-up", flush=True)
    rve_args.initial_rotation = _polar_rotation(cases[0]["deformation_gradient"])
    rve_args.mesh = mesh_by_vf[cases[0]["volume_fraction"]]
    _evaluate_direct_rve(
        cases[0]["volume_fraction"],
        cases[0]["deformation_gradient"],
        rve_args,
    )

    per_case = []
    error_records = []
    for case_number, case in enumerate(cases, start=1):
        vf = case["volume_fraction"]
        F_np = case["deformation_gradient"]
        print(
            f"[BENCHMARK] case={case_number}/{len(cases)} "
            f"sample={case['sample_id']} vf={vf:.6f}",
            flush=True,
        )
        surrogate_times = []
        prediction = None
        for _ in range(args.surrogate_repeats):
            start = time.perf_counter()
            prediction = _predict(model, F_np, vf, references[vf])
            surrogate_times.append(time.perf_counter() - start)

        direct_times = []
        direct_results = []
        for repeat in range(args.rve_repeats):
            print(
                f"[BENCHMARK] direct repeat={repeat + 1}/{args.rve_repeats}",
                flush=True,
            )
            rve_args.initial_rotation = _polar_rotation(F_np)
            rve_args.mesh = mesh_by_vf[vf]
            start = time.perf_counter()
            result = _evaluate_direct_rve(vf, F_np, rve_args)
            direct_times.append(time.perf_counter() - start)
            direct_results.append(result)
        direct_energy = float(np.mean([item["energy"] for item in direct_results]))
        direct_stress = np.mean(
            np.asarray([item["stress"] for item in direct_results], dtype=float),
            axis=0,
        )
        surrogate_energy, surrogate_stress = prediction
        label_energy = case["energy_label"]
        label_stress = case["stress_label"]
        item = {
            "sample_id": case["sample_id"],
            "state_index": case["state_index"],
            "split": case["split"],
            "volume_fraction": vf,
            "deformation_gradient": F_np.tolist(),
            "determinant": float(np.linalg.det(F_np)),
            "surrogate_energy": float(surrogate_energy),
            "surrogate_stress": surrogate_stress.tolist(),
            "holdout_energy_label": label_energy,
            "holdout_stress_label": label_stress.tolist(),
            "direct_rve_energy_mean": direct_energy,
            "direct_rve_stress_mean": direct_stress.tolist(),
            "direct_rve_energy_repeats": [float(result["energy"]) for result in direct_results],
            "direct_rve_seconds": [float(value) for value in direct_times],
            "surrogate_seconds": [float(value) for value in surrogate_times],
            "direct_rve_solver_statuses": [
                result["solver_status"] for result in direct_results
            ],
            "maximum_mixed_residual": max(
                result["mixed_residual"] for result in direct_results
            ),
            "maximum_rotation_residual": max(
                result["rotation_residual"] for result in direct_results
            ),
        }
        per_case.append(item)
        error_records.append({
            "energy_error": float(surrogate_energy - label_energy),
            "stress_error": surrogate_stress - label_stress,
        })
        item["surrogate_vs_holdout_energy_error"] = float(surrogate_energy - label_energy)
        item["surrogate_vs_holdout_stress_error"] = (
            surrogate_stress - label_stress
        ).tolist()
        item["direct_rve_vs_holdout_energy_error"] = float(direct_energy - label_energy)
        item["direct_rve_vs_holdout_stress_error"] = (
            direct_stress - label_stress
        ).tolist()

    normalization = checkpoint["normalization"]
    energy_scale = float(normalization["energy_scale"])
    stress_scale = float(normalization["stress_scale"])
    model_errors = _error_summary(error_records, energy_scale, stress_scale)
    direct_error_records = [
        {
            "energy_error": item["direct_rve_vs_holdout_energy_error"],
            "stress_error": np.asarray(
                item["direct_rve_vs_holdout_stress_error"],
                dtype=float,
            ),
        }
        for item in per_case
    ]
    direct_errors = _error_summary(direct_error_records, energy_scale, stress_scale)
    all_converged = all(
        all(status == "converged" for status in item["direct_rve_solver_statuses"])
        for item in per_case
    )
    report = {
        "schema_version": 1,
        "status": "complete" if all_converged else "partial",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "training_dataset_sha256": checkpoint.get("dataset_sha256"),
        "holdout_dataset_path": str(dataset_path.resolve()),
        "holdout_dataset_sha256": dataset_sha256,
        "holdout_manifest_configuration_sha256": manifest.get(
            "configuration_sha256"
        ),
        "holdout_is_independent_of_training": True,
        "evaluated_split": "test",
        "test_set_used_for_model_selection": False,
        "selected_state_indices": [
            int(item["state_index"]) for item in selected_groups
        ],
        "selected_state_determinants": [
            float(item["determinant"]) for item in selected_groups
        ],
        "n_cases": len(per_case),
        "selection_method": (
            "deterministic, evenly spaced sampling by determinant over non-reference "
            "deformation groups; all test volume fractions retained per group"
        ),
        "accuracy_vs_frozen_holdout": model_errors,
        "direct_rve_repeatability_vs_frozen_holdout": direct_errors,
        "runtime": _runtime_summary(
            per_case,
            surrogate_setup_seconds,
            mesh_setup_seconds,
        ),
        "repeats": {
            "surrogate": args.surrogate_repeats,
            "direct_rve": args.rve_repeats,
            "warmup_cases_excluded_from_timing": 1,
        },
        "rve_configuration": {
            "maxh": args.maxh,
            "mu": args.mu,
            "bulk_modulus": args.bulk_modulus,
            "displacement_order": args.displacement_order,
            "integration_order": args.integration_order,
            "microscopic_jacobian_floor": args.microscopic_jacobian_floor,
        },
        "runtime_environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "torch_threads": args.torch_threads,
        },
        "cases": per_case,
    }
    _atomic_json(args.output, report)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_id",
        "state_index",
        "volume_fraction",
        "determinant",
        "surrogate_seconds_median",
        "direct_rve_seconds_median",
        "paired_solver_speedup",
        "paired_amortized_speedup",
        "surrogate_energy",
        "holdout_energy",
        "direct_rve_energy",
        "surrogate_energy_error",
        "direct_rve_energy_error",
        "surrogate_stress_component_rmse",
        "direct_rve_stress_component_rmse",
        "maximum_mixed_residual",
        "maximum_rotation_residual",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in per_case:
            surrogate_seconds = float(np.median(item["surrogate_seconds"]))
            direct_seconds = float(np.median(item["direct_rve_seconds"]))
            case_count = len(per_case)
            surrogate_amortized = (
                surrogate_seconds + surrogate_setup_seconds / max(1, case_count)
            )
            direct_amortized = (
                direct_seconds
                + sum(mesh_setup_seconds.values()) / max(1, case_count)
            )
            surrogate_stress_error = np.asarray(
                item["surrogate_vs_holdout_stress_error"],
                dtype=float,
            )
            direct_stress_error = np.asarray(
                item["direct_rve_vs_holdout_stress_error"],
                dtype=float,
            )
            writer.writerow({
                "sample_id": item["sample_id"],
                "state_index": item["state_index"],
                "volume_fraction": item["volume_fraction"],
                "determinant": item["determinant"],
                "surrogate_seconds_median": surrogate_seconds,
                "direct_rve_seconds_median": direct_seconds,
                "paired_solver_speedup": direct_seconds / surrogate_seconds,
                "paired_amortized_speedup": direct_amortized / surrogate_amortized,
                "surrogate_energy": item["surrogate_energy"],
                "holdout_energy": item["holdout_energy_label"],
                "direct_rve_energy": item["direct_rve_energy_mean"],
                "surrogate_energy_error": item["surrogate_vs_holdout_energy_error"],
                "direct_rve_energy_error": item["direct_rve_vs_holdout_energy_error"],
                "surrogate_stress_component_rmse": float(
                    np.sqrt(np.mean(surrogate_stress_error**2))
                ),
                "direct_rve_stress_component_rmse": float(
                    np.sqrt(np.mean(direct_stress_error**2))
                ),
                "maximum_mixed_residual": item["maximum_mixed_residual"],
                "maximum_rotation_residual": item["maximum_rotation_residual"],
            })
    print(json.dumps({
        "status": report["status"],
        "n_cases": len(per_case),
        "surrogate_energy_nrmse_training_scale": model_errors[
            "energy_nrmse_training_scale"
        ],
        "surrogate_stress_nrmse_training_scale": model_errors[
            "stress_nrmse_training_scale"
        ],
        "median_paired_solver_speedup": report["runtime"][
            "median_paired_solver_speedup"
        ],
        "median_paired_amortized_speedup": report["runtime"][
            "median_paired_amortized_speedup"
        ],
        "report": str(args.output.resolve()),
        "csv": str(csv_path.resolve()),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--states", type=int, default=10)
    parser.add_argument("--surrogate-repeats", type=int, default=15)
    parser.add_argument("--rve-repeats", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--maxh", type=float, default=0.22)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--bulk-modulus", type=float, default=1.0e4)
    parser.add_argument("--displacement-order", type=int, default=2)
    parser.add_argument("--integration-order", type=int, default=10)
    parser.add_argument("--microscopic-jacobian-floor", type=float, default=1.0e-8)
    args = parser.parse_args()
    if args.states < 1 or args.surrogate_repeats < 1 or args.rve_repeats < 1:
        raise SystemExit("state count and repeat counts must be positive")
    run_benchmark(args)


if __name__ == "__main__":
    main()
