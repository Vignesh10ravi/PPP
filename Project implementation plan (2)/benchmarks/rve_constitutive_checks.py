"""Finite-difference energy/stress/tangent and objectivity checks for RVE data.

Run from the project root on a node with NGSolve installed, for example::

    python -m benchmarks.rve_constitutive_checks \
      --dataset data/rve_phase2_microj.npz \
      --output reports/rve_constitutive_checks.json

The audit uses central differences in all nine row-major macroscopic-gradient
directions and re-solves the microscopic equilibrium at each perturbed state.
"""

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np

from src.free_particle import solve_free_rotating_particle
from src.homogenization import homogenized_first_piola
from src.rve import make_periodic_particle_cell_mesh
from src.rve_dataset import (
    _atomic_json,
    _homogenized_energy,
    _source_revision_sha256,
    microscopic_jacobian_extrema,
    particle_radius,
)
from src.rve_solver import _matrix_stress
from src.rve_tangent import condensed_free_rotation_tangent


DEFAULT_SAMPLE_IDS = (
    "v000_f000047",
    "v001_f000047",
    "v002_f000086",
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_cf(matrix):
    from ngsolve import CF

    return CF(tuple(np.asarray(matrix, dtype=float).ravel()), dims=(3, 3))


def _polar_rotation(matrix):
    left, _, right_transpose = np.linalg.svd(np.asarray(matrix, dtype=float))
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return rotation


def _test_rotation():
    axis = np.asarray((1.0, 2.0, -1.0), dtype=float)
    axis /= np.linalg.norm(axis)
    skew = np.asarray(
        ((0.0, -axis[2], axis[1]),
         (axis[2], 0.0, -axis[0]),
         (-axis[1], axis[0], 0.0)),
        dtype=float,
    )
    angle = 0.37
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _relative_norm(actual, expected):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    return float(
        np.linalg.norm(actual - expected)
        / max(1.0, float(np.linalg.norm(actual)), float(np.linalg.norm(expected)))
    )


def _evaluate_state(mesh, gradient, volume_fraction, mu, bulk_modulus, order, jacobian_floor):
    state = solve_free_rotating_particle(
        mesh,
        np.asarray(gradient, dtype=float),
        initial_rotation=_polar_rotation(gradient),
        mu=mu,
        bulk_modulus=bulk_modulus,
        displacement_order=2,
        integration_order=order,
        verbose=False,
    )
    _, pressure = state.solution.components
    energy = _homogenized_energy(
        mesh,
        state.deformation_gradient,
        pressure,
        mu,
        bulk_modulus,
        order,
    )
    macroscopic_stress = homogenized_first_piola(
        mesh,
        state.stress,
        state.solution.space.components[0],
        integration_order=order,
    )
    minimum_j, maximum_j = microscopic_jacobian_extrema(
        mesh,
        state.deformation_gradient,
        order,
    )
    if minimum_j <= jacobian_floor:
        raise ValueError(
            f"microscopic J {minimum_j:.6e} is at or below floor "
            f"{jacobian_floor:.6e}"
        )
    return state, float(energy), np.asarray(macroscopic_stress), minimum_j, maximum_j


def _audit_sample(
    sample_id,
    gradient,
    volume_fraction,
    mesh,
    mu,
    bulk_modulus,
    order,
    fd_step,
    jacobian_floor,
    tolerances,
):
    base_state, base_energy, base_work_stress, base_min_j, base_max_j = _evaluate_state(
        mesh, gradient, volume_fraction, mu, bulk_modulus, order, jacobian_floor
    )
    tangent_result = condensed_free_rotation_tangent(
        mesh,
        base_state.solution,
        _matrix_cf(gradient),
        base_state.rotation,
        mu=mu,
        bulk_modulus=bulk_modulus,
        integration_order=order,
    )
    base_stress = np.asarray(tangent_result.stress, dtype=float)
    tangent = np.asarray(tangent_result.tangent, dtype=float)
    if tangent.shape != (9, 9):
        raise ValueError(f"expected a 9x9 tangent, got {tangent.shape}")

    energy_errors = []
    tangent_errors = []
    sampled_min_j = [base_min_j]
    sampled_max_j = [base_max_j]
    for column in range(9):
        direction = np.zeros((3, 3), dtype=float)
        direction.flat[column] = 1.0
        plus = _evaluate_state(
            mesh, gradient + fd_step * direction, volume_fraction,
            mu, bulk_modulus, order, jacobian_floor,
        )
        minus = _evaluate_state(
            mesh, gradient - fd_step * direction, volume_fraction,
            mu, bulk_modulus, order, jacobian_floor,
        )
        sampled_min_j.extend((plus[3], minus[3]))
        sampled_max_j.extend((plus[4], minus[4]))

        energy_fd = (plus[1] - minus[1]) / (2.0 * fd_step)
        energy_component = base_stress.flat[column]
        energy_errors.append(
            abs(energy_fd - energy_component)
            / max(1.0, abs(energy_fd), abs(energy_component))
        )

        stress_fd = (plus[2] - minus[2]).ravel() / (2.0 * fd_step)
        tangent_errors.append(
            _relative_norm(stress_fd, tangent[:, column])
        )

    work_stress_agreement = _relative_norm(base_work_stress, base_stress)

    rotation = _test_rotation()
    _, rotated_energy, rotated_stress, rotated_min_j, rotated_max_j = (
        _evaluate_state(
            mesh, rotation @ gradient, volume_fraction,
            mu, bulk_modulus, order, jacobian_floor,
        )
    )
    sampled_min_j.append(rotated_min_j)
    sampled_max_j.append(rotated_max_j)
    objectivity_energy_error = abs(rotated_energy - base_energy) / max(
        1.0, abs(rotated_energy), abs(base_energy)
    )
    objectivity_stress_error = _relative_norm(
        rotated_stress, rotation @ base_stress
    )

    metrics = {
        "max_energy_gradient_relative_error": float(max(energy_errors)),
        "max_tangent_directional_relative_error": float(max(tangent_errors)),
        "base_stress_work_vs_condensed_relative_error": work_stress_agreement,
        "objectivity_energy_relative_error": float(objectivity_energy_error),
        "objectivity_stress_relative_error": float(objectivity_stress_error),
        "minimum_sampled_microscopic_j": float(min(sampled_min_j)),
        "maximum_sampled_microscopic_j": float(max(sampled_max_j)),
        "mixed_residual": float(base_state.newton_history[-1]),
        "rotation_residual": float(base_state.rotational_residuals[-1]),
        "condensed_residual": float(max(tangent_result.linear_residuals)),
        "finite_difference_step": float(fd_step),
        "direction_count": 9,
    }
    checks = {
        "energy_gradient": metrics["max_energy_gradient_relative_error"]
        <= tolerances["energy_gradient"],
        "tangent_directional": metrics["max_tangent_directional_relative_error"]
        <= tolerances["tangent_directional"],
        "stress_agreement": metrics["base_stress_work_vs_condensed_relative_error"]
        <= tolerances["stress_agreement"],
        "objectivity_energy": metrics["objectivity_energy_relative_error"]
        <= tolerances["objectivity_energy"],
        "objectivity_stress": metrics["objectivity_stress_relative_error"]
        <= tolerances["objectivity_stress"],
        "microscopic_jacobian": metrics["minimum_sampled_microscopic_j"]
        > jacobian_floor,
    }
    return {
        "sample_id": sample_id,
        "volume_fraction": float(volume_fraction),
        "deformation_gradient": np.asarray(gradient).tolist(),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "metrics": metrics,
    }


def run_audit(
    dataset_path,
    output_path,
    sample_ids=DEFAULT_SAMPLE_IDS,
    maxh=0.22,
    integration_order=10,
    fd_step=1.0e-4,
    mu=1.0,
    bulk_modulus=1.0e4,
    jacobian_floor=1.0e-8,
    tolerances=None,
):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    tolerances = tolerances or {
        "energy_gradient": 5.0e-4,
        "tangent_directional": 5.0e-3,
        "stress_agreement": 5.0e-5,
        "objectivity_energy": 1.0e-6,
        "objectivity_stress": 5.0e-5,
    }
    if fd_step <= 0.0 or maxh <= 0.0 or integration_order < 1:
        raise ValueError("fd_step, maxh, and integration_order must be positive")

    with np.load(dataset_path, allow_pickle=False) as archive:
        sample_names = np.asarray(archive["sample_id"]).astype(str)
        gradients = np.asarray(archive["deformation_gradient"], dtype=float)
        volume_fractions = np.asarray(archive["volume_fraction"], dtype=float)
        indices = {name: index for index, name in enumerate(sample_names)}

    missing = set(sample_ids).difference(indices)
    if missing:
        raise ValueError(f"dataset is missing requested sample IDs: {sorted(missing)}")

    configuration = {
        "sample_ids": list(sample_ids),
        "maxh": float(maxh),
        "integration_order": int(integration_order),
        "fd_step": float(fd_step),
        "mu": float(mu),
        "bulk_modulus": float(bulk_modulus),
        "microscopic_jacobian_floor": float(jacobian_floor),
        "tolerances": tolerances,
    }
    report = {
        "status": "running",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": _file_sha256(dataset_path),
        "source_revision_sha256": _source_revision_sha256(),
        "audit_script_sha256": _file_sha256(__file__),
        "configuration": configuration,
        "results": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing_report = json.loads(output_path.read_text(encoding="utf-8"))
        for key in (
            "dataset_sha256",
            "source_revision_sha256",
            "audit_script_sha256",
            "configuration",
        ):
            if existing_report.get(key) != report[key]:
                raise ValueError(
                    f"existing report {output_path} has a different {key}; "
                    "choose a new output path"
                )
        report = existing_report
    else:
        _atomic_json(output_path, report)

    mesh_by_volume_fraction = {}
    results_by_id = {
        entry["sample_id"]: entry
        for entry in report.get("results", [])
        if "sample_id" in entry
    }
    for sample_id in sample_ids:
        previous = results_by_id.get(sample_id)
        if previous and previous.get("status") == "pass":
            print(f"[CONSTITUTIVE AUDIT REUSE] {sample_id}: pass", flush=True)
            continue
        index = indices[sample_id]
        gradient = gradients[index]
        volume_fraction = float(volume_fractions[index])
        if volume_fraction not in mesh_by_volume_fraction:
            mesh_by_volume_fraction[volume_fraction] = make_periodic_particle_cell_mesh(
                radius=particle_radius(volume_fraction), maxh=maxh
            )
        mesh = mesh_by_volume_fraction[volume_fraction]
        print(f"[CONSTITUTIVE AUDIT] {sample_id} vf={volume_fraction:.3f}", flush=True)
        try:
            result = _audit_sample(
                sample_id,
                gradient,
                volume_fraction,
                mesh,
                mu,
                bulk_modulus,
                integration_order,
                fd_step,
                jacobian_floor,
                tolerances,
            )
        except Exception as exc:
            result = {
                "sample_id": sample_id,
                "volume_fraction": volume_fraction,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        results_by_id[sample_id] = result
        report["results"] = [
            results_by_id[name] for name in sample_ids if name in results_by_id
        ]
        _atomic_json(output_path, report)
        print(
            f"[CONSTITUTIVE AUDIT RESULT] {sample_id}: {result['status']}",
            flush=True,
        )

    report["status"] = (
        "pass" if all(item.get("status") == "pass" for item in report["results"])
        else "fail"
    )
    _atomic_json(output_path, report)
    print(f"Constitutive audit report: {output_path}", flush=True)
    print(f"Constitutive audit status: {report['status']}", flush=True)
    if report["status"] != "pass":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-ids", nargs="+", default=DEFAULT_SAMPLE_IDS)
    parser.add_argument("--maxh", type=float, default=0.22)
    parser.add_argument("--integration-order", type=int, default=10)
    parser.add_argument("--fd-step", type=float, default=1.0e-4)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--bulk-modulus", type=float, default=1.0e4)
    parser.add_argument("--jacobian-floor", type=float, default=1.0e-8)
    args = parser.parse_args()
    run_audit(
        dataset_path=args.dataset,
        output_path=args.output,
        sample_ids=args.sample_ids,
        maxh=args.maxh,
        integration_order=args.integration_order,
        fd_step=args.fd_step,
        mu=args.mu,
        bulk_modulus=args.bulk_modulus,
        jacobian_floor=args.jacobian_floor,
    )


if __name__ == "__main__":
    main()
