"""Map pICNN responses and volume-fraction sensitivities for three load cases.

The surrogate evaluates energy, first-Piola stress, and the full material
tangent at three in-domain particle volume fractions. Direct RVE solves can
validate every map point. The report includes autograd-versus-finite-difference
checks for energy and selected stress sensitivities with respect to volume
fraction.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.func import jacrev

from benchmarks.rve_inverse_design import (
    _evaluate_direct_rve,
    _finite_difference_derivative,
    _load_model,
)


LOAD_CASES = {
    "uniaxial_5pct": np.diag([1.05, 1.0, 1.0]),
    "simple_shear_5pct": np.asarray(
        [[1.0, 0.05, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    ),
    "uniaxial_compression_5pct": np.diag([0.95, 1.0, 1.0]),
}


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


def _as_float(value):
    return float(value.detach().cpu())


def _model_values(model, F_np, volume_fraction, reference=None):
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    F = torch.as_tensor(F_np, dtype=dtype, device=device)
    context = torch.as_tensor(
        [[float(volume_fraction)]],
        dtype=dtype,
        device=device,
    )
    if reference is None:
        reference = model.reference(dtype, device, context)

    def energy_single(gradient):
        return model.energy(
            gradient.unsqueeze(0),
            context,
            ref=reference,
        ).squeeze()

    energy = energy_single(F)
    stress = jacrev(energy_single)(F)
    return energy, stress


def _predict_with_sensitivities(model, F_np, volume_fraction):
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    F = torch.as_tensor(F_np, dtype=dtype, device=device)
    context = torch.tensor(
        [[float(volume_fraction)]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    reference = model.reference(dtype, device, context)

    def energy_single(gradient):
        return model.energy(
            gradient.unsqueeze(0),
            context,
            ref=reference,
        ).squeeze()

    energy = energy_single(F)
    stress = jacrev(energy_single)(F)
    tangent = jacrev(jacrev(energy_single))(F).reshape(9, 9)
    energy_sensitivity = torch.autograd.grad(
        energy,
        context,
        retain_graph=True,
    )[0].reshape(())
    p00_sensitivity = torch.autograd.grad(
        stress[0, 0],
        context,
        retain_graph=True,
    )[0].reshape(())
    p01_sensitivity = torch.autograd.grad(
        stress[0, 1],
        context,
    )[0].reshape(())
    return {
        "energy": _as_float(energy),
        "stress": stress.detach().cpu().numpy().tolist(),
        "tangent_frobenius_norm": float(torch.linalg.vector_norm(tangent).detach().cpu()),
        "d_energy_d_vf": _as_float(energy_sensitivity),
        "d_p00_d_vf": _as_float(p00_sensitivity),
        "d_p01_d_vf": _as_float(p01_sensitivity),
    }


def _selected_stress_values(model, F_np, volume_fraction):
    energy, stress = _model_values(model, F_np, volume_fraction)
    return {
        "energy": _as_float(energy),
        "p00": _as_float(stress[0, 0]),
        "p01": _as_float(stress[0, 1]),
    }


def _scaled_error(analytical, finite_difference):
    return abs(analytical - finite_difference) / max(
        1.0,
        abs(analytical),
        abs(finite_difference),
    )


def _polar_rotation(deformation_gradient):
    left, _, right_transpose = np.linalg.svd(deformation_gradient)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return rotation


def run_map(args):
    lower, upper = map(float, args.volume_fraction_bounds)
    volume_fractions = np.asarray(args.volume_fractions, dtype=float)
    if (
        not np.isfinite(volume_fractions).all()
        or np.any(volume_fractions < lower)
        or np.any(volume_fractions > upper)
        or len(np.unique(volume_fractions)) != len(volume_fractions)
    ):
        raise ValueError("all unique volume fractions must lie inside the configured bounds")
    if not (0.0 < lower < upper < np.pi / 6.0):
        raise ValueError("volume-fraction bounds must satisfy 0 < lower < upper < pi/6")
    if args.gradient_fd_step <= 0 or args.gradient_tolerance <= 0:
        raise ValueError("gradient finite-difference step and tolerance must be positive")

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

    records = []
    derivative_checks = []
    max_scaled_derivative_error = 0.0
    max_repeatability_error = 0.0
    for load_name, F_np in LOAD_CASES.items():
        determinant = float(np.linalg.det(F_np))
        condition = float(np.linalg.cond(F_np))
        if determinant <= 0.0 or condition > args.max_condition_number:
            raise ValueError(f"{load_name} lies outside the configured admissible range")
        print(
            f"[SURROGATE MAP] load={load_name} detF={determinant:.6f}",
            flush=True,
        )
        for vf in volume_fractions:
            prediction = _predict_with_sensitivities(model, F_np, float(vf))
            repeated_prediction = _predict_with_sensitivities(
                model,
                F_np,
                float(vf),
            )
            finite_values = np.asarray([
                prediction["energy"],
                prediction["tangent_frobenius_norm"],
                prediction["d_energy_d_vf"],
                prediction["d_p00_d_vf"],
                prediction["d_p01_d_vf"],
                *np.asarray(prediction["stress"], dtype=float).ravel(),
            ])
            if not np.isfinite(finite_values).all():
                raise FloatingPointError(
                    f"nonfinite surrogate response for {load_name}, vf={vf}"
                )
            repeatability_error = max(
                _scaled_error(prediction["energy"], repeated_prediction["energy"]),
                _scaled_error(
                    prediction["tangent_frobenius_norm"],
                    repeated_prediction["tangent_frobenius_norm"],
                ),
                _scaled_error(
                    prediction["d_energy_d_vf"],
                    repeated_prediction["d_energy_d_vf"],
                ),
                _scaled_error(
                    prediction["d_p00_d_vf"],
                    repeated_prediction["d_p00_d_vf"],
                ),
                _scaled_error(
                    prediction["d_p01_d_vf"],
                    repeated_prediction["d_p01_d_vf"],
                ),
                float(
                    np.linalg.norm(
                        np.asarray(prediction["stress"])
                        - np.asarray(repeated_prediction["stress"])
                    )
                    / max(
                        1.0,
                        np.linalg.norm(np.asarray(prediction["stress"])),
                        np.linalg.norm(np.asarray(repeated_prediction["stress"])),
                    )
                ),
            )
            max_repeatability_error = max(
                max_repeatability_error,
                repeatability_error,
            )
            step = min(
                args.gradient_fd_step,
                0.25 * (float(vf) - lower) if vf > lower else args.gradient_fd_step,
                0.25 * (upper - float(vf)) if vf < upper else args.gradient_fd_step,
            )
            finite_difference = {}
            for response_name, analytic_name in (
                ("energy", "d_energy_d_vf"),
                ("p00", "d_p00_d_vf"),
                ("p01", "d_p01_d_vf"),
            ):
                response = lambda value, key=response_name: _selected_stress_values(
                    model,
                    F_np,
                    value,
                )[key]
                fd = _finite_difference_derivative(
                    response,
                    float(vf),
                    lower,
                    upper,
                    step,
                )
                analytical = prediction[analytic_name]
                error = _scaled_error(analytical, fd)
                finite_difference[response_name] = float(fd)
                derivative_checks.append({
                    "load_case": load_name,
                    "volume_fraction": float(vf),
                    "response": response_name,
                    "autograd_derivative": float(analytical),
                    "finite_difference_derivative": float(fd),
                    "scaled_error": float(error),
                    "step": float(step),
                })
                max_scaled_derivative_error = max(
                    max_scaled_derivative_error,
                    error,
                )
            records.append({
                "load_case": load_name,
                "deformation_gradient": F_np.tolist(),
                "deformation_determinant": determinant,
                "deformation_condition_number": condition,
                "volume_fraction": float(vf),
                **prediction,
                "repeatability_scaled_error": float(repeatability_error),
                "finite_difference_sensitivities": finite_difference,
                "direct_rve": None,
            })

    if max_scaled_derivative_error > args.gradient_tolerance:
        raise RuntimeError(
            "volume-fraction derivative verification failed: "
            f"{max_scaled_derivative_error:.3e} exceeds "
            f"{args.gradient_tolerance:.3e}"
        )
    if max_repeatability_error > args.repeatability_tolerance:
        raise RuntimeError(
            "surrogate repeatability check failed: "
            f"{max_repeatability_error:.3e} exceeds "
            f"{args.repeatability_tolerance:.3e}"
        )

    validation_summary = {
        "status": "not_run" if args.skip_direct_rve_validation else "pending",
        "n_direct_rve_cases": 0,
        "energy_rmse": None,
        "stress_component_rmse": None,
    }
    if not args.skip_direct_rve_validation:
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
        energy_errors = []
        stress_errors = []
        for record in records:
            F_np = np.asarray(record["deformation_gradient"], dtype=float)
            vf = record["volume_fraction"]
            rve_args.initial_rotation = _polar_rotation(F_np)
            direct = _evaluate_direct_rve(vf, F_np, rve_args)
            record["direct_rve"] = direct
            energy_errors.append(record["energy"] - direct["energy"])
            stress_errors.append(
                np.asarray(record["stress"], dtype=float)
                - np.asarray(direct["stress"], dtype=float)
            )
        energy_errors = np.asarray(energy_errors, dtype=float)
        stress_errors = np.asarray(stress_errors, dtype=float)
        validation_summary = {
            "status": "pass" if np.isfinite(energy_errors).all()
            and np.isfinite(stress_errors).all() else "failed",
            "n_direct_rve_cases": int(len(records)),
            "energy_rmse": float(np.sqrt(np.mean(energy_errors**2))),
            "energy_mae": float(np.mean(np.abs(energy_errors))),
            "stress_component_rmse": float(np.sqrt(np.mean(stress_errors**2))),
            "stress_component_mae": float(np.mean(np.abs(stress_errors))),
            "stress_frobenius_rmse": float(
                np.sqrt(np.mean(np.sum(stress_errors**2, axis=(1, 2))))
            ),
            "all_direct_rve_solvers_converged": all(
                item["direct_rve"]["solver_status"] == "converged"
                for item in records
            ),
        }

    report = {
        "schema_version": 1,
        "status": (
            "derivative_checks_pass_direct_rve_not_run"
            if validation_summary["status"] == "not_run"
            and max_scaled_derivative_error <= args.gradient_tolerance
            and max_repeatability_error <= args.repeatability_tolerance
            else "pass"
            if max_scaled_derivative_error <= args.gradient_tolerance
            and max_repeatability_error <= args.repeatability_tolerance
            and validation_summary["status"] == "pass"
            else "failed"
        ),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "training_dataset_sha256": checkpoint.get("dataset_sha256"),
        "best_epoch": checkpoint.get("best_epoch"),
        "load_cases": {
            name: {
                "deformation_gradient": F.tolist(),
                "determinant": float(np.linalg.det(F)),
                "condition_number": float(np.linalg.cond(F)),
            }
            for name, F in LOAD_CASES.items()
        },
        "volume_fractions": [float(value) for value in volume_fractions],
        "volume_fraction_bounds": [lower, upper],
        "surrogate_map": records,
        "verification": {
            "autograd_vs_finite_difference": derivative_checks,
            "maximum_scaled_derivative_error": float(max_scaled_derivative_error),
            "tolerance": float(args.gradient_tolerance),
            "status": (
                "pass"
                if max_scaled_derivative_error <= args.gradient_tolerance
                and max_repeatability_error <= args.repeatability_tolerance
                else "failed"
            ),
            "finite_difference_rule": (
                "central difference in the interior; second-order one-sided "
                "difference at either volume-fraction bound"
            ),
            "maximum_repeatability_scaled_error": float(max_repeatability_error),
            "repeatability_tolerance": float(args.repeatability_tolerance),
            "repeatability_status": (
                "pass"
                if max_repeatability_error <= args.repeatability_tolerance
                else "failed"
            ),
        },
        "direct_rve_validation": validation_summary,
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
    }
    _atomic_json(args.output, report)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "load_case",
        "volume_fraction",
        "determinant",
        "energy",
        "p00",
        "p01",
        "tangent_frobenius_norm",
        "repeatability_scaled_error",
        "d_energy_d_vf",
        "d_p00_d_vf",
        "d_p01_d_vf",
        "fd_d_energy_d_vf",
        "fd_d_p00_d_vf",
        "fd_d_p01_d_vf",
        "direct_energy",
        "direct_p00",
        "direct_p01",
        "energy_relative_error",
        "stress_frobenius_relative_error",
        "direct_solver_status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            direct = record["direct_rve"]
            writer.writerow({
                "load_case": record["load_case"],
                "volume_fraction": record["volume_fraction"],
                "determinant": record["deformation_determinant"],
                "energy": record["energy"],
                "p00": record["stress"][0][0],
                "p01": record["stress"][0][1],
                "tangent_frobenius_norm": record["tangent_frobenius_norm"],
                "repeatability_scaled_error": record["repeatability_scaled_error"],
                "d_energy_d_vf": record["d_energy_d_vf"],
                "d_p00_d_vf": record["d_p00_d_vf"],
                "d_p01_d_vf": record["d_p01_d_vf"],
                "fd_d_energy_d_vf": record["finite_difference_sensitivities"]["energy"],
                "fd_d_p00_d_vf": record["finite_difference_sensitivities"]["p00"],
                "fd_d_p01_d_vf": record["finite_difference_sensitivities"]["p01"],
                "direct_energy": None if direct is None else direct["energy"],
                "direct_p00": None if direct is None else direct["stress"][0][0],
                "direct_p01": None if direct is None else direct["stress"][0][1],
                "energy_relative_error": (
                    None if direct is None
                    else abs(record["energy"] - direct["energy"])
                    / max(1.0, abs(direct["energy"]))
                ),
                "stress_frobenius_relative_error": (
                    None if direct is None
                    else float(np.linalg.norm(
                        np.asarray(record["stress"]) - np.asarray(direct["stress"])
                    ) / max(1.0, np.linalg.norm(direct["stress"])))
                ),
                "direct_solver_status": None if direct is None else direct["solver_status"],
            })
    print(json.dumps({
        "status": report["status"],
        "checkpoint_sha256": checkpoint_sha256,
        "map_points": len(records),
        "direct_rve_points": validation_summary["n_direct_rve_cases"],
        "maximum_scaled_derivative_error": max_scaled_derivative_error,
        "maximum_repeatability_scaled_error": max_repeatability_error,
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
    parser.add_argument("--volume-fraction-bounds", type=float, nargs=2, default=(0.02, 0.10))
    parser.add_argument("--gradient-fd-step", type=float, default=1.0e-5)
    parser.add_argument("--gradient-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--repeatability-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--max-condition-number", type=float, default=3.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--maxh", type=float, default=0.22)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--bulk-modulus", type=float, default=1.0e4)
    parser.add_argument("--displacement-order", type=int, default=2)
    parser.add_argument("--integration-order", type=int, default=10)
    parser.add_argument("--microscopic-jacobian-floor", type=float, default=1.0e-8)
    parser.add_argument("--skip-direct-rve-validation", action="store_true")
    args = parser.parse_args()
    run_map(args)


if __name__ == "__main__":
    main()
