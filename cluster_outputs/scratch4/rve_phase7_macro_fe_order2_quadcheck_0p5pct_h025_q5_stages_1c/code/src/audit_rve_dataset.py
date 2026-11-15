"""Audit a completed RVE NPZ dataset and its manifest without rerunning solves."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .rve_dataset import _source_revision_sha256


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(dataset_path, manifest_path):
    dataset_path = Path(dataset_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(dataset_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}

    config = manifest["config"]
    config_digest = hashlib.sha256(
        json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if manifest.get("configuration_sha256") != config_digest:
        raise ValueError("configuration hash does not match the manifest configuration")
    data_digest = _sha256_file(dataset_path)
    if manifest.get("dataset_sha256") != data_digest:
        raise ValueError("NPZ hash does not match the manifest")
    if manifest.get("tangent_flattening") != "row-major: (i,J) -> 3*i + J":
        raise ValueError("tangent flattening convention is missing or unexpected")
    if manifest.get("source_revision_sha256") != _source_revision_sha256():
        raise ValueError("source revision hash does not match the current solver sources")
    n_states = int(config["n_deformations"])
    volume_fractions = np.asarray(config["volume_fractions"], dtype=float)
    n_expected = n_states * len(volume_fractions)
    required = {
        "volume_fraction",
        "deformation_gradient",
        "energy",
        "stress",
        "tangent",
        "rotation",
        "determinant",
        "condition_number",
        "mixed_residual",
        "rotation_residual",
        "condensed_residual",
        "tangent_symmetry_error",
        "minimum_microscopic_jacobian",
        "maximum_microscopic_jacobian",
        "mixed_iterations",
        "rotation_iterations",
        "minimum_particle_clearance",
        "solver_status",
        "state_index",
        "split",
        "sample_id",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"dataset is missing fields: {sorted(missing)}")
    if manifest["n_samples"] != n_expected or len(data["sample_id"]) != n_expected:
        raise ValueError(f"expected {n_expected} samples")

    for name, spec in manifest["fields"].items():
        if name not in data or list(data[name].shape) != spec["shape"]:
            raise ValueError(f"field {name!r} shape does not match the manifest")
    for name, values in data.items():
        if np.issubdtype(values.dtype, np.number) and not np.all(np.isfinite(values)):
            raise ValueError(f"field {name!r} contains nonfinite values")

    expected_ids = np.asarray([
        f"v{volume_index:03d}_f{state_index:06d}"
        for volume_index in range(len(volume_fractions))
        for state_index in range(n_states)
    ])
    expected_vf = np.repeat(volume_fractions, n_states)
    expected_state = np.tile(np.arange(n_states), len(volume_fractions))
    if not np.array_equal(data["sample_id"], expected_ids):
        raise ValueError("sample IDs are missing, duplicated, or out of order")
    if not np.allclose(data["volume_fraction"], expected_vf, rtol=0, atol=1e-12):
        raise ValueError("volume fractions do not match the manifest")
    if not np.array_equal(data["state_index"], expected_state):
        raise ValueError("state indices do not match the expected records")

    gradients = data["deformation_gradient"]
    determinants = np.linalg.det(gradients)
    condition_numbers = np.linalg.cond(gradients)
    if np.any(determinants <= config["determinant_floor"]):
        raise ValueError("a macroscopic determinant is below the configured floor")
    if np.any(condition_numbers > config["max_condition_number"]):
        raise ValueError("a macroscopic condition number exceeds its configured limit")
    if not np.allclose(determinants, data["determinant"], rtol=1e-9, atol=1e-10):
        raise ValueError("stored determinants do not match the deformation gradients")
    if not np.allclose(
        condition_numbers,
        data["condition_number"],
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError("stored condition numbers do not match the deformation gradients")

    jacobian_floor = float(config["microscopic_jacobian_floor"])
    if np.any(data["minimum_microscopic_jacobian"] <= jacobian_floor):
        raise ValueError("a sample is at or below the microscopic-J safety floor")
    if np.any(
        data["maximum_microscopic_jacobian"]
        < data["minimum_microscopic_jacobian"]
    ):
        raise ValueError("a sample's maximum microscopic J is below its minimum")
    if np.any(data["minimum_particle_clearance"] <= 0):
        raise ValueError("a sample has nonpositive particle clearance")
    expected_clearance = 0.5 - np.cbrt(
        3.0 * data["volume_fraction"] / (4.0 * np.pi)
    )
    if not np.allclose(
        data["minimum_particle_clearance"],
        expected_clearance,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("stored particle clearance does not match its volume fraction")
    if not np.all(data["solver_status"] == "converged"):
        raise ValueError("not all samples have converged solver status")

    for name in ("mixed_residual", "rotation_residual", "condensed_residual"):
        if np.any(data[name] >= 1e-8):
            raise ValueError(f"{name} exceeds the 1e-8 convergence threshold")
    if np.any(data["tangent_symmetry_error"] >= 1e-8):
        raise ValueError("tangent symmetry error exceeds 1e-8")
    for name in ("mixed_iterations", "rotation_iterations"):
        if np.any(data[name] < 0):
            raise ValueError(f"{name} contains a negative iteration count")

    for state_index in range(n_states):
        if len(np.unique(data["split"][data["state_index"] == state_index])) != 1:
            raise ValueError(f"split leakage for deformation state {state_index}")
    for split, expected_count in manifest["split_counts"].items():
        if np.count_nonzero(data["split"] == split) != expected_count:
            raise ValueError(f"split count mismatch for {split}")

    reference = data["state_index"] == 0
    if not np.allclose(gradients[reference], np.eye(3), rtol=0, atol=1e-12):
        raise ValueError("identity reference states are missing")
    if np.max(np.abs(data["energy"][reference])) >= 1e-8:
        raise ValueError("identity-state energy is not approximately zero")
    if np.max(np.abs(data["stress"][reference])) >= 1e-8:
        raise ValueError("identity-state stress is not approximately zero")

    rotations = data["rotation"]
    orthogonality_error = np.max(
        np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3))
    )
    if orthogonality_error >= 1e-8 or not np.allclose(
        np.linalg.det(rotations), 1.0, rtol=0, atol=1e-8
    ):
        raise ValueError("particle rotations are not proper orthogonal matrices")

    print(f"PASS: {n_expected} samples passed stored-data integrity checks")
    print(f"Dataset SHA-256: {manifest['dataset_sha256']}")
    print(f"Configuration SHA-256: {manifest['configuration_sha256']}")
    print(f"Source revision SHA-256: {manifest['source_revision_sha256']}")
    print(
        "Microscopic J range at integration quadrature points: "
        f"[{data['minimum_microscopic_jacobian'].min():.6e}, "
        f"{data['maximum_microscopic_jacobian'].max():.6e}]"
    )
    print(f"Microscopic-J safety floor: {jacobian_floor:.6e}")
    print(f"Max mixed residual: {data['mixed_residual'].max():.3e}")
    print(f"Max rotation residual: {data['rotation_residual'].max():.3e}")
    print(f"Max condensed residual: {data['condensed_residual'].max():.3e}")
    print(f"Max tangent symmetry error: {data['tangent_symmetry_error'].max():.3e}")
    print(
        "Note: microscopic extrema are sampled at the configured integration "
        "quadrature points."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    audit_dataset(args.dataset, args.manifest)


if __name__ == "__main__":
    main()
