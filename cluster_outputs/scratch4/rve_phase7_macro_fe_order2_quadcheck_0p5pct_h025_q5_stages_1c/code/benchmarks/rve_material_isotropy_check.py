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
    _test_rotation,
)
from src.rve_dataset import _source_revision_sha256


DEFAULT_SAMPLE_IDS = (
    "v000_f000032",
    "v001_f000032",
    "v002_f000032",
    "v002_f000052",
    "v002_f000072",
    "v002_f000097",
)


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
              maxh=None, integration_order=None, jacobian_floor=None):
    dataset_path = Path(dataset_path)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    sample_ids = tuple(sample_ids or DEFAULT_SAMPLE_IDS)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_sha256 = _file_sha256(dataset_path)
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("dataset hash does not match its manifest")
    config = manifest["config"]
    maxh = float(config["maxh"] if maxh is None else maxh)
    integration_order = int(
        config["integration_order"]
        if integration_order is None
        else integration_order
    )
    jacobian_floor = float(
        config["microscopic_jacobian_floor"]
        if jacobian_floor is None
        else jacobian_floor
    )
    if maxh <= 0 or integration_order < 1 or jacobian_floor <= 0:
        raise ValueError("maxh, integration_order, and jacobian_floor must be positive")

    with np.load(dataset_path, allow_pickle=False) as archive:
        sample_names = np.asarray(archive["sample_id"]).astype(str)
        gradients = np.asarray(archive["deformation_gradient"], dtype=float)
        volume_fractions = np.asarray(archive["volume_fraction"], dtype=float)
        splits = np.asarray(archive["split"]).astype(str)
        indices = {name: index for index, name in enumerate(sample_names)}

    missing = set(sample_ids).difference(indices)
    if missing:
        raise ValueError(f"dataset is missing requested sample IDs: {sorted(missing)}")

    rotation = _test_rotation()
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-12):
        raise ValueError("test rotation is not orthogonal")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-12):
        raise ValueError("test rotation is not proper")

    settings = {
        "sample_ids": list(sample_ids),
        "maxh": maxh,
        "integration_order": integration_order,
        "microscopic_jacobian_floor": jacobian_floor,
        "mu": float(config["mu"]),
        "bulk_modulus": float(config["bulk_modulus"]),
        "displacement_order": int(config["displacement_order"]),
        "right_rotation": rotation.tolist(),
        "rotation_axis_angle_radians": 0.37,
    }
    report = {
        "status": "running",
        "purpose": "diagnostic_material_isotropy_check",
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
            "rotated_deformation_gradient": (gradient @ rotation).tolist(),
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
                float(config["mu"]),
                float(config["bulk_modulus"]),
                integration_order,
                jacobian_floor,
            )
            rotated_gradient = gradient @ rotation
            rotated = _evaluate_state(
                mesh,
                rotated_gradient,
                volume_fraction,
                float(config["mu"]),
                float(config["bulk_modulus"]),
                integration_order,
                jacobian_floor,
            )
            base_state, base_energy, base_stress, base_min_j, base_max_j = base
            rotated_state, rotated_energy, rotated_stress, rotated_min_j, rotated_max_j = rotated
            expected_rotated_stress = base_stress @ rotation
            result.update(
                {
                    "status": "complete",
                    "energy_base": float(base_energy),
                    "energy_right_rotated": float(rotated_energy),
                    "energy_absolute_difference": float(
                        abs(rotated_energy - base_energy)
                    ),
                    "energy_relative_difference": float(
                        abs(rotated_energy - base_energy)
                        / max(1.0, abs(base_energy), abs(rotated_energy))
                    ),
                    "stress_base": np.asarray(base_stress).tolist(),
                    "stress_right_rotated": np.asarray(rotated_stress).tolist(),
                    "stress_expected_if_isotropic": expected_rotated_stress.tolist(),
                    "stress_relative_difference": _relative_norm(
                        rotated_stress, expected_rotated_stress
                    ),
                    "base_mixed_residual": float(base_state.newton_history[-1]),
                    "rotated_mixed_residual": float(
                        rotated_state.newton_history[-1]
                    ),
                    "base_rotation_residual": float(
                        base_state.rotational_residuals[-1]
                    ),
                    "rotated_rotation_residual": float(
                        rotated_state.rotational_residuals[-1]
                    ),
                    "base_microscopic_j_range": [
                        float(base_min_j), float(base_max_j)
                    ],
                    "rotated_microscopic_j_range": [
                        float(rotated_min_j), float(rotated_max_j)
                    ],
                }
            )
            print(
                f"[MATERIAL ISOTROPY] {sample_id} "
                f"energy_rel={result['energy_relative_difference']:.6e} "
                f"stress_rel={result['stress_relative_difference']:.6e}",
                flush=True,
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
    _atomic_json(output_path, report)
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
    args = parser.parse_args()
    run_check(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        output_path=args.output,
        sample_ids=args.sample_ids,
        maxh=args.maxh,
        integration_order=args.integration_order,
        jacobian_floor=args.jacobian_floor,
    )


if __name__ == "__main__":
    main()
