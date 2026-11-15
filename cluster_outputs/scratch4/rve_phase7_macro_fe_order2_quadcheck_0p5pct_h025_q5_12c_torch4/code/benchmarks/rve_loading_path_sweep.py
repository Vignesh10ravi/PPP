"""Evaluate frozen-pICNN response curves along representative 3D loading paths.

This is a surrogate-only exploratory study. It does not consume holdout,
inverse-design, volume-map, or direct-RVE benchmark results. All loading paths
remain in the trained volume-fraction interval and are checked for positive
determinant and bounded condition number.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.func import jacrev

from benchmarks.rve_inverse_design import _finite_difference_derivative, _load_model


LOAD_MODES = (
    "uniaxial_tension",
    "uniaxial_compression",
    "simple_shear",
    "equibiaxial_tension",
    "volumetric_expansion",
)


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


def _path_state(mode, amplitude):
    F = np.eye(3, dtype=float)
    dF = np.zeros((3, 3), dtype=float)
    if mode == "uniaxial_tension":
        F[0, 0] = 1.0 + amplitude
        dF[0, 0] = 1.0
    elif mode == "uniaxial_compression":
        F[0, 0] = 1.0 - amplitude
        dF[0, 0] = -1.0
    elif mode == "simple_shear":
        F[0, 1] = amplitude
        dF[0, 1] = 1.0
    elif mode == "equibiaxial_tension":
        F[0, 0] = 1.0 + amplitude
        F[1, 1] = 1.0 + amplitude
        dF[0, 0] = 1.0
        dF[1, 1] = 1.0
    elif mode == "volumetric_expansion":
        stretch = (1.0 + amplitude) ** (1.0 / 3.0)
        derivative = (1.0 / 3.0) * (1.0 + amplitude) ** (-2.0 / 3.0)
        F *= stretch
        dF = derivative * np.eye(3)
    else:
        raise ValueError(f"unknown loading mode: {mode}")
    return F, dF


def _energy_at(model, F_np, reference, context):
    F = torch.as_tensor(F_np, dtype=torch.float64, device="cpu")
    return float(
        model.energy(F.unsqueeze(0), context, ref=reference)
        .reshape(())
        .detach()
        .cpu()
    )


def _evaluate_point(model, F_np, dF_da_np, context, reference):
    F = torch.as_tensor(F_np, dtype=torch.float64, device="cpu")
    dF_da = torch.as_tensor(dF_da_np, dtype=torch.float64, device="cpu")

    def energy_single(gradient):
        return model.energy(
            gradient.unsqueeze(0),
            context,
            ref=reference,
        ).squeeze()

    energy = energy_single(F)
    stress = jacrev(energy_single)(F)
    tangent = jacrev(jacrev(energy_single))(F).reshape(9, 9)
    generalized_force = torch.sum(stress * dF_da)
    directional_tangent = (
        dF_da.reshape(9) @ tangent @ dF_da.reshape(9)
    )
    tangent_symmetry_error = torch.max(torch.abs(tangent - tangent.T)) / torch.clamp(
        torch.max(torch.abs(tangent)),
        min=1.0,
    )
    values = np.asarray([
        float(energy.detach().cpu()),
        float(generalized_force.detach().cpu()),
        float(directional_tangent.detach().cpu()),
        float(tangent_symmetry_error.detach().cpu()),
        *stress.detach().cpu().numpy().ravel(),
    ])
    if not np.isfinite(values).all():
        raise FloatingPointError("pICNN produced a nonfinite loading-path response")
    return {
        "energy": float(energy.detach().cpu()),
        "stress": stress.detach().cpu().numpy().tolist(),
        "generalized_force_from_stress": float(generalized_force.detach().cpu()),
        "directional_tangent": float(directional_tangent.detach().cpu()),
        "tangent_major_symmetry_error": float(tangent_symmetry_error.detach().cpu()),
    }


def run_sweep(args):
    if not (0.0 < args.max_amplitude <= 0.20):
        raise ValueError("max amplitude must lie in (0, 0.20] to stay in-domain")
    if args.steps < 3:
        raise ValueError("at least three loading-path points are required")
    volume_fractions = np.asarray(args.volume_fractions, dtype=float)
    if (
        not np.isfinite(volume_fractions).all()
        or np.any(volume_fractions < 0.02)
        or np.any(volume_fractions > 0.10)
        or len(np.unique(volume_fractions)) != len(volume_fractions)
    ):
        raise ValueError("unique volume fractions must lie in the trained interval [0.02, 0.10]")
    if args.fd_step <= 0 or args.derivative_tolerance <= 0:
        raise ValueError("finite-difference step and derivative tolerance must be positive")

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

    amplitude_values = np.linspace(0.0, args.max_amplitude, args.steps)
    records = []
    derivative_checks = []
    maximum_derivative_error = 0.0
    maximum_tangent_symmetry_error = 0.0
    for vf in volume_fractions:
        context = torch.as_tensor(
            [[float(vf)]],
            dtype=torch.float64,
            device="cpu",
        )
        reference = model.reference(torch.float64, torch.device("cpu"), context)
        for mode in LOAD_MODES:
            print(f"[LOAD PATH] mode={mode} vf={vf:.6f}", flush=True)

            def energy_along_path(amplitude):
                F_np, _ = _path_state(mode, float(amplitude))
                return _energy_at(model, F_np, reference, context)

            for amplitude in amplitude_values:
                F_np, dF_da_np = _path_state(mode, float(amplitude))
                determinant = float(np.linalg.det(F_np))
                condition = float(np.linalg.cond(F_np))
                if (
                    determinant <= 0.0
                    or determinant < args.minimum_determinant
                    or determinant > args.maximum_determinant
                    or condition > args.max_condition_number
                ):
                    raise ValueError(
                        f"{mode} at amplitude {amplitude:.6g} leaves the "
                        "configured deformation domain"
                    )
                prediction = _evaluate_point(
                    model,
                    F_np,
                    dF_da_np,
                    context,
                    reference,
                )
                fd_step = min(
                    args.fd_step,
                    0.25 * args.max_amplitude,
                )
                fd_slope = _finite_difference_derivative(
                    energy_along_path,
                    float(amplitude),
                    0.0,
                    args.max_amplitude,
                    fd_step,
                )
                analytic_slope = prediction["generalized_force_from_stress"]
                scaled_error = abs(analytic_slope - fd_slope) / max(
                    1.0,
                    abs(analytic_slope),
                    abs(fd_slope),
                )
                if not np.isfinite([analytic_slope, fd_slope, scaled_error]).all():
                    raise FloatingPointError("nonfinite work-conjugacy diagnostic")
                maximum_derivative_error = max(
                    maximum_derivative_error,
                    scaled_error,
                )
                maximum_tangent_symmetry_error = max(
                    maximum_tangent_symmetry_error,
                    prediction["tangent_major_symmetry_error"],
                )
                derivative_checks.append({
                    "load_mode": mode,
                    "volume_fraction": float(vf),
                    "amplitude": float(amplitude),
                    "stress_contracted_with_path_tangent": analytic_slope,
                    "finite_difference_energy_slope": float(fd_slope),
                    "scaled_error": float(scaled_error),
                    "step": float(fd_step),
                })
                records.append({
                    "load_mode": mode,
                    "volume_fraction": float(vf),
                    "amplitude": float(amplitude),
                    "deformation_gradient": F_np.tolist(),
                    "determinant": determinant,
                    "condition_number": condition,
                    **prediction,
                    "finite_difference_energy_slope": float(fd_slope),
                    "work_conjugacy_scaled_error": float(scaled_error),
                })

    if maximum_derivative_error > args.derivative_tolerance:
        raise RuntimeError(
            "stress/energy path consistency failed: "
            f"{maximum_derivative_error:.3e} exceeds "
            f"{args.derivative_tolerance:.3e}"
        )
    if maximum_tangent_symmetry_error > args.tangent_symmetry_tolerance:
        raise RuntimeError(
            "pICNN tangent major-symmetry check failed: "
            f"{maximum_tangent_symmetry_error:.3e} exceeds "
            f"{args.tangent_symmetry_tolerance:.3e}"
        )
    report = {
        "schema_version": 1,
        "status": "pass",
        "validation_scope": "surrogate_only; direct-RVE labels are not used",
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "training_dataset_sha256": checkpoint.get("dataset_sha256"),
        "best_epoch": checkpoint.get("best_epoch"),
        "load_modes": list(LOAD_MODES),
        "volume_fractions": [float(value) for value in volume_fractions],
        "amplitudes": [float(value) for value in amplitude_values],
        "deformation_domain": {
            "determinant_bounds": [
                args.minimum_determinant,
                args.maximum_determinant,
            ],
            "maximum_condition_number": args.max_condition_number,
            "maximum_amplitude": args.max_amplitude,
        },
        "summary": {
            "n_path_points": int(len(records)),
            "maximum_work_conjugacy_scaled_error": float(maximum_derivative_error),
            "work_conjugacy_tolerance": float(args.derivative_tolerance),
            "maximum_tangent_major_symmetry_error": float(
                maximum_tangent_symmetry_error
            ),
            "tangent_major_symmetry_tolerance": float(
                args.tangent_symmetry_tolerance
            ),
            "work_conjugacy_status": (
                "pass"
                if maximum_derivative_error <= args.derivative_tolerance
                else "failed"
            ),
        },
        "work_conjugacy_checks": derivative_checks,
        "load_path": records,
        "runtime_environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "torch_threads": args.torch_threads,
        },
    }
    _atomic_json(args.output, report)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "load_mode",
        "volume_fraction",
        "amplitude",
        "determinant",
        "condition_number",
        "energy",
        "generalized_force_from_stress",
        "finite_difference_energy_slope",
        "work_conjugacy_scaled_error",
        "directional_tangent",
        "tangent_major_symmetry_error",
        "p00",
        "p01",
        "p11",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in records:
            stress = item["stress"]
            writer.writerow({
                "load_mode": item["load_mode"],
                "volume_fraction": item["volume_fraction"],
                "amplitude": item["amplitude"],
                "determinant": item["determinant"],
                "condition_number": item["condition_number"],
                "energy": item["energy"],
                "generalized_force_from_stress": item["generalized_force_from_stress"],
                "finite_difference_energy_slope": item["finite_difference_energy_slope"],
                "work_conjugacy_scaled_error": item["work_conjugacy_scaled_error"],
                "directional_tangent": item["directional_tangent"],
                "tangent_major_symmetry_error": item["tangent_major_symmetry_error"],
                "p00": stress[0][0],
                "p01": stress[0][1],
                "p11": stress[1][1],
            })
    print(json.dumps({
        "status": report["status"],
        "path_points": len(records),
        "maximum_work_conjugacy_scaled_error": maximum_derivative_error,
        "maximum_tangent_major_symmetry_error": maximum_tangent_symmetry_error,
        "report": str(args.output.resolve()),
        "csv": str(csv_path.resolve()),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--volume-fractions", type=float, nargs="+", default=(0.02, 0.05, 0.10))
    parser.add_argument("--max-amplitude", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=11)
    parser.add_argument("--fd-step", type=float, default=1.0e-5)
    parser.add_argument("--derivative-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--tangent-symmetry-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--minimum-determinant", type=float, default=0.50)
    parser.add_argument("--maximum-determinant", type=float, default=1.80)
    parser.add_argument("--max-condition-number", type=float, default=3.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
