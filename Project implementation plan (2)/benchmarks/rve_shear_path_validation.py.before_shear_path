"""Compare frozen-pICNN and direct-RVE responses along a simple-shear path.

The study adds direct-RVE solves at 2.5% and 7.5% shear for three volume
fractions. Existing 5% direct-RVE points are reused from the volume-fraction
map, yielding a nine-point amplitude-by-volume-fraction comparison.
"""

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from benchmarks.rve_inverse_design import _evaluate_direct_rve, _load_model
from benchmarks.rve_volume_fraction_map import _model_values, _polar_rotation


def _as_float(value):
    return float(value.detach().cpu())


def _simple_shear_gradient(amplitude):
    return np.asarray(
        [[1.0, float(amplitude), 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _record(mode, amplitude, volume_fraction, prediction, direct, error_floor):
    predicted_stress = np.asarray(prediction["stress"], dtype=float)
    direct_stress = np.asarray(direct["stress"], dtype=float)
    stress_error = predicted_stress - direct_stress
    direct_energy = float(direct["energy"])
    predicted_energy = float(prediction["energy"])
    direct_p01 = float(direct_stress[0, 1])
    predicted_p01 = float(predicted_stress[0, 1])
    return {
        "loading_mode": mode,
        "shear_amplitude": float(amplitude),
        "volume_fraction": float(volume_fraction),
        "deformation_gradient": _simple_shear_gradient(amplitude).tolist(),
        "energy_pICNN": predicted_energy,
        "energy_direct_rve": direct_energy,
        "energy_absolute_error": abs(predicted_energy - direct_energy),
        "energy_relative_error": abs(predicted_energy - direct_energy)
        / max(abs(direct_energy), error_floor),
        "stress_pICNN": predicted_stress.tolist(),
        "stress_direct_rve": direct_stress.tolist(),
        "stress_component_rmse": float(np.sqrt(np.mean(stress_error**2))),
        "stress_frobenius_absolute_error": float(np.linalg.norm(stress_error)),
        "stress_frobenius_relative_error": float(
            np.linalg.norm(stress_error)
            / max(float(np.linalg.norm(direct_stress)), error_floor)
        ),
        "P01_pICNN": predicted_p01,
        "P01_direct_rve": direct_p01,
        "P01_absolute_error": abs(predicted_p01 - direct_p01),
        "P01_relative_error": abs(predicted_p01 - direct_p01)
        / max(abs(direct_p01), error_floor),
        "direct_rve_solver_status": direct["solver_status"],
        "direct_rve_mixed_residual": direct["mixed_residual"],
        "direct_rve_rotation_residual": direct["rotation_residual"],
        "direct_rve_minimum_microscopic_jacobian": direct[
            "minimum_microscopic_jacobian"
        ],
    }


def run_study(args):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    model, checkpoint, checkpoint_sha256 = _load_model(args.checkpoint)
    if (
        args.expected_checkpoint_sha256
        and checkpoint_sha256 != args.expected_checkpoint_sha256
    ):
        raise ValueError("checkpoint SHA-256 does not match the frozen expected hash")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    baseline = json.loads(args.baseline_map.read_text(encoding="utf-8"))
    if baseline.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("baseline map was generated with a different checkpoint")
    baseline_rows = {
        round(float(item["volume_fraction"]), 12): item
        for item in baseline.get("surrogate_map", [])
        if item.get("load_case") == "simple_shear_5pct"
        and item.get("direct_rve") is not None
    }
    requested_vfs = [float(value) for value in args.volume_fractions]
    if len(set(requested_vfs)) != len(requested_vfs):
        raise ValueError("volume fractions must be unique")
    if any(round(vf, 12) not in baseline_rows for vf in requested_vfs):
        raise ValueError(
            "baseline map must contain direct-RVE simple-shear rows at every "
            "requested volume fraction"
        )
    if any(
        baseline_rows[round(vf, 12)]["direct_rve"].get("solver_status")
        != "converged"
        for vf in requested_vfs
    ):
        raise ValueError("baseline map contains a nonconverged direct-RVE point")
    if (
        not args.amplitudes
        or len(set(args.amplitudes)) != len(args.amplitudes)
        or any(
            not np.isfinite(value) or value <= 0.0 or value > 0.10
            for value in args.amplitudes
        )
    ):
        raise ValueError("new shear amplitudes must be unique and lie in (0, 0.10]")
    if not np.isfinite(args.error_floor) or args.error_floor <= 0.0:
        raise ValueError("error_floor must be finite and positive")
    if any(np.isclose(value, 0.05, rtol=0.0, atol=1.0e-12) for value in args.amplitudes):
        raise ValueError("5% shear is reused from the baseline map; do not rerun it")

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
    records = []
    for vf in requested_vfs:
        baseline_row = baseline_rows[round(vf, 12)]
        records.append(_record(
            "simple_shear",
            0.05,
            vf,
            {
                "energy": baseline_row["energy"],
                "stress": baseline_row["stress"],
            },
            baseline_row["direct_rve"],
            args.error_floor,
        ))

    for amplitude in args.amplitudes:
        F_np = _simple_shear_gradient(amplitude)
        for vf in requested_vfs:
            print(
                f"[SHEAR PATH] direct-RVE gamma={amplitude:.4f} vf={vf:.4f}",
                flush=True,
            )
            energy, stress = _model_values(model, F_np, vf)
            prediction = {
                "energy": _as_float(energy),
                "stress": stress.detach().cpu().numpy().tolist(),
            }
            rve_args.initial_rotation = _polar_rotation(F_np)
            direct = _evaluate_direct_rve(vf, F_np, rve_args)
            records.append(_record(
                "simple_shear",
                amplitude,
                vf,
                prediction,
                direct,
                args.error_floor,
            ))

    records.sort(key=lambda item: (item["shear_amplitude"], item["volume_fraction"]))
    p01_relative = [item["P01_relative_error"] for item in records]
    report = {
        "schema_version": 1,
        "status": "complete",
        "scope": (
            "Direct-RVE validation along simple shear at gamma=0.025, 0.05, and "
            "0.075 for volume fractions 0.02, 0.05, and 0.10. The 5% points are "
            "reused from the completed volume-fraction map. This is validation "
            "of a frozen checkpoint and does not modify training data."
        ),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "training_dataset_sha256": checkpoint.get("dataset_sha256"),
        "baseline_map_path": str(args.baseline_map.resolve()),
        "baseline_map_checkpoint_sha256": baseline.get("checkpoint_sha256"),
        "new_direct_rve_solves": len(args.amplitudes) * len(requested_vfs),
        "reused_baseline_points": len(requested_vfs),
        "volume_fractions": requested_vfs,
        "shear_amplitudes": sorted([0.05, *[float(x) for x in args.amplitudes]]),
        "error_floor": float(args.error_floor),
        "rve_configuration": {
            "maxh": args.maxh,
            "mu": args.mu,
            "bulk_modulus": args.bulk_modulus,
            "displacement_order": args.displacement_order,
            "integration_order": args.integration_order,
            "microscopic_jacobian_floor": args.microscopic_jacobian_floor,
        },
        "summary": {
            "n_points": len(records),
            "n_direct_rve_points": sum(
                item["direct_rve_solver_status"] == "converged" for item in records
            ),
            "maximum_P01_absolute_error": max(
                item["P01_absolute_error"] for item in records
            ),
            "maximum_P01_relative_error": max(p01_relative),
            "maximum_stress_component_rmse": max(
                item["stress_component_rmse"] for item in records
            ),
            "maximum_stress_frobenius_relative_error": max(
                item["stress_frobenius_relative_error"] for item in records
            ),
        },
        "points": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_path = args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "shear_amplitude", "volume_fraction", "energy_pICNN",
        "energy_direct_rve", "energy_absolute_error", "energy_relative_error",
        "stress_component_rmse", "stress_frobenius_absolute_error",
        "stress_frobenius_relative_error", "P01_pICNN", "P01_direct_rve",
        "P01_absolute_error", "P01_relative_error", "direct_rve_solver_status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: item[key] for key in columns} for item in records)
    print(json.dumps({
        "status": report["status"],
        "summary": report["summary"],
        "report": str(args.output.resolve()),
        "csv": str(csv_path.resolve()),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=(0.025, 0.075))
    parser.add_argument("--volume-fractions", type=float, nargs="+", default=(0.02, 0.05, 0.10))
    parser.add_argument("--error-floor", type=float, default=1.0e-8)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--maxh", type=float, default=0.22)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--bulk-modulus", type=float, default=1.0e4)
    parser.add_argument("--displacement-order", type=int, default=2)
    parser.add_argument("--integration-order", type=int, default=10)
    parser.add_argument("--microscopic-jacobian-floor", type=float, default=1.0e-8)
    args = parser.parse_args()
    run_study(args)


if __name__ == "__main__":
    main()
