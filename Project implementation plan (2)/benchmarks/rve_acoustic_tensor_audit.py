"""Screen stored constitutive tangents for sampled strong ellipticity.

This post-processes the pICNN and direct-RVE tangents saved by
``rve_picnn_tangent_validation``. For each reference direction N, it forms
the acoustic tensor Q_ik = A_iJkL N_J N_L and records its smallest symmetric
eigenvalue and corresponding polarization. The finite direction set is a
screening audit, not a proof of strong ellipticity for every direction.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _unit_directions(fibonacci_count):
    if fibonacci_count < 1:
        raise ValueError("fibonacci direction count must be positive")

    index = np.arange(fibonacci_count, dtype=float) + 0.5
    z = 1.0 - 2.0 * index / fibonacci_count
    angle = np.pi * (1.0 + np.sqrt(5.0)) * index
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    directions = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))

    # Include exact coordinate and body-diagonal directions in addition to
    # the quasi-uniform sphere sample.
    special = []
    for x in (-1.0, 0.0, 1.0):
        for y in (-1.0, 0.0, 1.0):
            for z_value in (-1.0, 0.0, 1.0):
                vector = np.asarray((x, y, z_value), dtype=float)
                if np.all(vector == 0.0):
                    continue
                first_nonzero = vector[np.flatnonzero(vector)[0]]
                if first_nonzero < 0.0:
                    continue
                special.append(vector / np.linalg.norm(vector))
    return np.vstack((directions, np.asarray(special, dtype=float)))


def _acoustic_tensor(tangent, direction):
    # The stored 9x9 tangent uses row-major flattening of (i,J) and (k,L).
    elasticity = np.asarray(tangent, dtype=float).reshape(3, 3, 3, 3)
    acoustic = np.einsum("iJkL,J,L->ik", elasticity, direction, direction)
    return 0.5 * (acoustic + acoustic.T), float(
        np.linalg.norm(acoustic - acoustic.T)
        / max(1.0, np.linalg.norm(acoustic))
    )


def _scan_tangent(state, label, tangent, directions):
    tangent = np.asarray(tangent, dtype=float)
    if tangent.shape != (9, 9) or not np.isfinite(tangent).all():
        raise ValueError(f"{state} {label} tangent must be a finite 9x9 matrix")

    best = None
    maximum_symmetry_error = 0.0
    for direction in directions:
        acoustic, symmetry_error = _acoustic_tensor(tangent, direction)
        maximum_symmetry_error = max(maximum_symmetry_error, symmetry_error)
        eigenvalues, eigenvectors = np.linalg.eigh(acoustic)
        candidate = {
            "minimum_eigenvalue": float(eigenvalues[0]),
            "reference_direction": direction.tolist(),
            "polarization": eigenvectors[:, 0].tolist(),
        }
        if best is None or candidate["minimum_eigenvalue"] < best[
            "minimum_eigenvalue"
        ]:
            best = candidate

    return {
        "state": state,
        "tangent_source": label,
        **best,
        "maximum_acoustic_tensor_symmetry_error": maximum_symmetry_error,
        "sampled_direction_count": int(len(directions)),
        "sampled_sign": (
            "positive" if best["minimum_eigenvalue"] > 0.0 else "nonpositive"
        ),
    }


def audit(input_path, output_json, output_csv, direction_count):
    source = json.loads(input_path.read_text(encoding="utf-8"))
    if source.get("status") != "complete":
        raise ValueError("input tangent report must have status 'complete'")
    if not source.get("results"):
        raise ValueError("input tangent report has no state results")

    directions = _unit_directions(direction_count)
    records = []
    for result in source["results"]:
        state = str(result["state"])
        for label, key in (
            ("pICNN", "tangent_pICNN"),
            ("direct_rve", "tangent_direct_rve"),
        ):
            if key not in result:
                raise ValueError(f"{state} is missing {key}")
            records.append(_scan_tangent(
                state,
                label,
                result[key],
                directions,
            ))

    report = {
        "schema_version": 1,
        "status": "complete",
        "scope": (
            "Sampled-direction acoustic-tensor screening of stored frozen-pICNN "
            "and direct-RVE tangents. Positive sampled minima do not prove "
            "strong ellipticity over all reference directions. No model or "
            "RVE solves are run by this postprocessor."
        ),
        "source_tangent_report": str(input_path.resolve()),
        "checkpoint_sha256": source.get("checkpoint_sha256"),
        "volume_fraction": source.get("volume_fraction"),
        "fibonacci_direction_count": int(direction_count),
        "total_direction_count": int(len(directions)),
        "results": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "state",
        "tangent_source",
        "minimum_eigenvalue",
        "reference_direction",
        "polarization",
        "sampled_sign",
        "sampled_direction_count",
        "maximum_acoustic_tensor_symmetry_error",
    )
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({
                **record,
                "reference_direction": json.dumps(record["reference_direction"]),
                "polarization": json.dumps(record["polarization"]),
            })

    print(json.dumps({
        "status": report["status"],
        "sampled_directions_per_tangent": len(directions),
        "results": [
            {
                "state": row["state"],
                "tangent_source": row["tangent_source"],
                "minimum_eigenvalue": row["minimum_eigenvalue"],
                "sampled_sign": row["sampled_sign"],
            }
            for row in records
        ],
        "report": str(output_json.resolve()),
        "csv": str(output_csv.resolve()),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--fibonacci-directions", type=int, default=642)
    args = parser.parse_args()
    audit(
        args.input,
        args.output_json,
        args.output_csv,
        args.fibonacci_directions,
    )


if __name__ == "__main__":
    main()
