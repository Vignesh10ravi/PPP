"""Check whether selected periodic RVE responses are materially isotropic.

For an isotropic hyperelastic material, a right rotation R should satisfy
W(F R) = W(F) and P(F R) = P(F) R. This diagnostic keeps the periodic cell
fixed and reports those differences; it does not assume the result must pass.
"""

import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np

from benchmarks.rve_constitutive_checks import (
    _evaluate_state,
    _file_sha256,
    _relative_norm,
)
from src.rve_dataset import _source_revision_sha256


DEFAULT_SAMPLE_IDS = (
    "phase7_v000_f000032",
    "phase7_v001_f000032",
    "phase7_v002_f000032",
    "phase7_v002_f000052",
    "phase7_v002_f000072",
    "phase7_v002_f000097",
)


def _rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.asarray(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=float
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _test_rotations():
    """Return a fixed, reproducible set of non-cubic material rotations."""
    definitions = (
        ("axis_1_2_minus1_angle_0p37", (1.0, 2.0, -1.0), 0.37),
        ("axis_minus2_1_3_angle_0p61", (-2.0, 1.0, 3.0), 0.61),
        ("axis_1_1_1_angle_0p83", (1.0, 1.0, 1.0), 0.83),
    )
    rotations = []
    for name, axis, angle in definitions:
        matrix = _rotation_matrix(axis, angle)
        if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0, atol=1.0e-12):
            raise ValueError(f"rotation {name} is not orthogonal")
        if not np.isclose(np.linalg.det(matrix), 1.0, rtol=0, atol=1.0e-12):
            raise ValueError(f"rotation {name} is not proper")
        rotations.append(
            {
                "name": name,
                "axis": (np.asarray(axis, dtype=float) / np.linalg.norm(axis)).tolist(),
                "angle_radians": float(angle),
                "matrix": matrix,
            }
        )
    return rotations


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_check(dataset_path, manifest_path, output_path, sample_ids=None,
              maxh=None, integration_order=None, jacobian_floor=None,
              csv_path=None, expected_dataset_sha256=None, mu=None,
              bulk_modulus=None, displacement_order=None):
    dataset_path = Path(dataset_path)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    sample_ids = tuple(sample_ids or DEFAULT_SAMPLE_IDS)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_sha256 = _file_sha256(dataset_path)
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("dataset hash does not match its manifest")
    if expected_dataset_sha256 and dataset_sha256 != expected_dataset_sha256:
        raise ValueError(
            "dataset SHA-256 does not match the expected frozen hash: "
            f"actual={dataset_sha256}, expected={expected_dataset_sha256}"
        )
    config = manifest["config"]
    maxh = float(config["maxh"] if maxh is None and "maxh" in config else maxh)
    integration_order = int(
        config["integration_order"]
        if integration_order is None and "integration_order" in config
        else integration_order
    )
    jacobian_floor = float(
        config["microscopic_jacobian_floor"]
        if jacobian_floor is None and "microscopic_jacobian_floor" in config
        else jacobian_floor
    )
    mu = float(config["mu"] if mu is None and "mu" in config else mu)
    bulk_modulus = float(
        config["bulk_modulus"]
        if bulk_modulus is None and "bulk_modulus" in config
        else bulk_modulus
    )
    displacement_order = int(
        config["displacement_order"]
        if displacement_order is None and "displacement_order" in config
        else displacement_order
    )
    if displacement_order != 2:
        raise ValueError(
            "this audit currently uses the P2 direct-RVE solver; "
            "displacement_order must be 2"
        )
    if maxh <= 0 or integration_order < 1 or jacobian_floor <= 0:
        raise ValueError("maxh, integration_order, and jacobian_floor must be positive")
    if mu <= 0 or bulk_modulus <= 0:
        raise ValueError("mu and bulk_modulus must be positive")

    with np.load(dataset_path, allow_pickle=False) as archive:
        sample_names = np.asarray(archive["sample_id"]).astype(str)
        gradients = np.asarray(archive["deformation_gradient"], dtype=float)
        volume_fractions = np.asarray(archive["volume_fraction"], dtype=float)
        splits = np.asarray(archive["split"]).astype(str)
        indices = {name: index for index, name in enumerate(sample_names)}

    missing = set(sample_ids).difference(indices)
    if missing:
        raise ValueError(f"dataset is missing requested sample IDs: {sorted(missing)}")

    rotations = _test_rotations()

    settings = {
        "sample_ids": list(sample_ids),
        "maxh": maxh,
        "integration_order": integration_order,
        "microscopic_jacobian_floor": jacobian_floor,
        "mu": mu,
        "bulk_modulus": bulk_modulus,
        "displacement_order": displacement_order,
        "right_rotations": [
            {
                "name": item["name"],
                "axis": item["axis"],
                "angle_radians": item["angle_radians"],
                "matrix": item["matrix"].tolist(),
            }
            for item in rotations
        ],
        "rotation_count_per_state": len(rotations),
    }
    report = {
        "status": "running",
        "purpose": "diagnostic_multi_rotation_material_isotropy_check",
        "dataset_path": str(dataset_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "source_revision_sha256": _source_revision_sha256(),
        "audit_script_sha256": _file_sha256(__file__),
        "settings": settings,
        "results": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, report)

    mesh_by_vf = {}
    results = []
    for sample_id in sample_ids:
        index = indices[sample_id]
        gradient = gradients[index]
        volume_fraction = float(volume_fractions[index])
        result = {
            "sample_id": sample_id,
            "split": str(splits[index]),
            "state_index": int(
                int(sample_id.rsplit("f", 1)[1])
                if sample_id.rsplit("f", 1)[1].isdigit()
                else -1
            ),
            "volume_fraction": volume_fraction,
            "deformation_gradient": gradient.tolist(),
        }
        try:
            if volume_fraction not in mesh_by_vf:
                from src.rve import make_periodic_particle_cell_mesh
                from src.rve_dataset import particle_radius

                mesh_by_vf[volume_fraction] = make_periodic_particle_cell_mesh(
                    radius=particle_radius(volume_fraction), maxh=maxh
                )
            mesh = mesh_by_vf[volume_fraction]
            base = _evaluate_state(
                mesh,
                gradient,
                volume_fraction,
                mu,
                bulk_modulus,
                integration_order,
                jacobian_floor,
            )
            base_state, base_energy, base_stress, base_min_j, base_max_j = base
            result.update(
                {
                    "stress_base": np.asarray(base_stress).tolist(),
                    "energy_base": float(base_energy),
                    "base_microscopic_j_range": [float(base_min_j), float(base_max_j)],
                    "base_mixed_residual": float(base_state.newton_history[-1]),
                    "base_rotation_residual": float(
                        base_state.rotational_residuals[-1]
                    ),
                }
            )
            rotation_results = []
            for rotation_spec in rotations:
                rotation = rotation_spec["matrix"]
                rotation_result = {
                    "rotation_name": rotation_spec["name"],
                    "rotation_matrix": rotation.tolist(),
                    "rotated_deformation_gradient": (gradient @ rotation).tolist(),
                }
                try:
                    rotated_gradient = gradient @ rotation
                    rotated = _evaluate_state(
                        mesh,
                        rotated_gradient,
                        volume_fraction,
                        mu,
                        bulk_modulus,
                        integration_order,
                        jacobian_floor,
                    )
                    rotated_state, rotated_energy, rotated_stress, rotated_min_j, rotated_max_j = rotated
                    expected_rotated_stress = base_stress @ rotation
                    rotation_result.update(
                        {
                            "status": "complete",
                            "energy_right_rotated": float(rotated_energy),
                            "energy_absolute_difference": float(abs(rotated_energy - base_energy)),
                            "energy_relative_difference": float(
                                abs(rotated_energy - base_energy)
                                / max(1.0, abs(base_energy), abs(rotated_energy))
                            ),
                            "stress_right_rotated": np.asarray(rotated_stress).tolist(),
                            "stress_expected_if_isotropic": expected_rotated_stress.tolist(),
                            "stress_relative_difference": _relative_norm(
                                rotated_stress, expected_rotated_stress
                            ),
                            "rotated_mixed_residual": float(rotated_state.newton_history[-1]),
                            "rotated_rotation_residual": float(rotated_state.rotational_residuals[-1]),
                            "rotated_microscopic_j_range": [float(rotated_min_j), float(rotated_max_j)],
                        }
                    )
                    print(
                        f"[MATERIAL ISOTROPY] {sample_id} {rotation_spec['name']} "
                        f"energy_rel={rotation_result['energy_relative_difference']:.6e} "
                        f"stress_rel={rotation_result['stress_relative_difference']:.6e}",
                        flush=True,
                    )
                except Exception as exc:
                    rotation_result.update(
                        {
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(
                        f"[MATERIAL ISOTROPY ERROR] {sample_id} "
                        f"{rotation_spec['name']}: {exc}",
                        flush=True,
                    )
                rotation_results.append(rotation_result)
            result["rotation_results"] = rotation_results
            result["status"] = (
                "complete"
                if all(item.get("status") == "complete" for item in rotation_results)
                else "completed_with_errors"
            )
        except Exception as exc:
            result.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[MATERIAL ISOTROPY ERROR] {sample_id}: {exc}", flush=True)
        results.append(result)
        report["results"] = results
        report["status"] = "running"
        _atomic_json(output_path, report)

    report["status"] = (
        "complete" if all(item.get("status") == "complete" for item in results)
        else "completed_with_errors"
    )
    successful_rotations = [
        rotated
        for item in results
        for rotated in item.get("rotation_results", [])
        if rotated.get("status") == "complete"
    ]
    report["summary"] = {
        "n_base_states": len(results),
        "n_rotation_cases": sum(
            len(item.get("rotation_results", [])) for item in results
        ),
        "n_successful_rotation_cases": len(successful_rotations),
        "maximum_energy_relative_difference": (
            max(item["energy_relative_difference"] for item in successful_rotations)
            if successful_rotations else None
        ),
        "median_energy_relative_difference": (
            float(np.median([
                item["energy_relative_difference"] for item in successful_rotations
            ]))
            if successful_rotations else None
        ),
        "maximum_stress_relative_difference": (
            max(item["stress_relative_difference"] for item in successful_rotations)
            if successful_rotations else None
        ),
        "median_stress_relative_difference": (
            float(np.median([
                item["stress_relative_difference"] for item in successful_rotations
            ]))
            if successful_rotations else None
        ),
        "pass_fail_threshold_applied": False,
    }
    _atomic_json(output_path, report)
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with csv_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(
                [
                    "sample_id", "split", "state_index", "volume_fraction",
                    "rotation_name", "energy_relative_difference",
                    "stress_relative_difference", "base_mixed_residual",
                    "rotated_mixed_residual", "base_rotation_residual",
                    "rotated_rotation_residual", "status",
                ]
            )
            for item in results:
                for rotated in item.get("rotation_results", []):
                    writer.writerow(
                        [
                            item["sample_id"], item["split"], item["state_index"],
                            item["volume_fraction"], rotated["rotation_name"],
                            rotated.get("energy_relative_difference"),
                            rotated.get("stress_relative_difference"),
                            item.get("base_mixed_residual"),
                            rotated.get("rotated_mixed_residual"),
                            item.get("base_rotation_residual"),
                            rotated.get("rotated_rotation_residual"),
                            rotated.get("status", item.get("status")),
                        ]
                    )
        print(f"Material isotropy CSV: {csv_path}", flush=True)
    print(f"Material isotropy report: {output_path}", flush=True)
    print(f"Status: {report['status']}", flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-ids", nargs="+", default=DEFAULT_SAMPLE_IDS)
    parser.add_argument("--maxh", type=float)
    parser.add_argument("--integration-order", type=int)
    parser.add_argument("--jacobian-floor", type=float)
    parser.add_argument("--csv")
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--mu", type=float)
    parser.add_argument("--bulk-modulus", type=float)
    parser.add_argument("--displacement-order", type=int)
    args = parser.parse_args()
    run_check(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        output_path=args.output,
        sample_ids=args.sample_ids,
        maxh=args.maxh,
        integration_order=args.integration_order,
        jacobian_floor=args.jacobian_floor,
        csv_path=args.csv,
        expected_dataset_sha256=args.expected_dataset_sha256,
        mu=args.mu,
        bulk_modulus=args.bulk_modulus,
        displacement_order=args.displacement_order,
    )


if __name__ == "__main__":
    main()
