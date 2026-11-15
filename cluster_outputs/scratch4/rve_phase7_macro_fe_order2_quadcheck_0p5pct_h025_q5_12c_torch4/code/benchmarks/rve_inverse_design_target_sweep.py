"""Verify several bounded inverse-design targets with the frozen pICNN and RVE.

Targets are set at fractions of the direct-RVE response span between the
volume-fraction endpoints for one admissible deformation. Each target is
optimized independently with the frozen surrogate and checked at the selected
volume fraction with the direct RVE solver. This is a multi-target extension
of the inverse-design proof of concept, not validation against external
experimental targets.
"""

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from benchmarks.rve_inverse_design import (
    _atomic_json,
    _evaluate_direct_rve,
    _load_model,
    _make_response_function,
    _verify_response_derivative,
    optimize_volume_fraction,
)


def _polar_rotation(deformation_gradient):
    left, _, right_transpose = np.linalg.svd(deformation_gradient)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return rotation


def run_sweep(args):
    lower, upper = map(float, args.volume_fraction_bounds)
    target_fractions = np.asarray(args.target_fractions, dtype=float)
    F_np = np.asarray(args.deformation_gradient, dtype=float).reshape(3, 3)
    condition_number = float(np.linalg.cond(F_np))
    determinant = float(np.linalg.det(F_np))
    if not (0.0 < lower < upper < np.pi / 6.0):
        raise ValueError("volume-fraction bounds must satisfy 0 < lower < upper < pi/6")
    if (
        not np.isfinite(target_fractions).all()
        or np.any(target_fractions <= 0.0)
        or np.any(target_fractions >= 1.0)
        or len(np.unique(target_fractions)) != len(target_fractions)
    ):
        raise ValueError("target fractions must be unique values strictly between zero and one")
    if determinant <= 0.0 or condition_number > args.max_condition_number:
        raise ValueError("deformation gradient must be admissible and well-conditioned")
    if args.gradient_fd_step <= 0.0 or args.gradient_tolerance <= 0.0:
        raise ValueError("finite-difference step and derivative tolerance must be positive")
    if args.target_error_limit < 0.0:
        raise ValueError("target-error limit cannot be negative")
    if args.response == "stress" and not (
        0 <= args.i < 3 and 0 <= args.j < 3
    ):
        raise ValueError("stress indices must each be in {0, 1, 2}")

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    model, checkpoint, checkpoint_sha256 = _load_model(args.checkpoint)
    if (
        args.expected_checkpoint_sha256
        and checkpoint_sha256 != args.expected_checkpoint_sha256
    ):
        raise ValueError("checkpoint SHA-256 does not match the frozen expected hash")

    response_fn, value_and_derivative = _make_response_function(
        model,
        torch.as_tensor(F_np, dtype=torch.float64),
        args.response,
        args.i,
        args.j,
    )
    rve_args = Namespace(
        maxh=args.maxh,
        mu=args.mu,
        bulk_modulus=args.bulk_modulus,
        displacement_order=args.displacement_order,
        integration_order=args.integration_order,
        microscopic_jacobian_floor=args.microscopic_jacobian_floor,
        response=args.response,
        i=args.i,
        j=args.j,
        rve_verbose=False,
    )

    direct_cache = {}
    endpoint_results = []
    for vf in (lower, upper):
        print(f"[TARGET SWEEP] direct-RVE endpoint vf={vf:.6f}", flush=True)
        rve_args.initial_rotation = _polar_rotation(F_np)
        result = _evaluate_direct_rve(vf, F_np, rve_args)
        direct_cache[float(vf)] = result
        endpoint_results.append(result)
    endpoint_responses = [item["response"] for item in endpoint_results]
    response_low = float(min(endpoint_responses))
    response_high = float(max(endpoint_responses))
    response_span = response_high - response_low
    if response_span <= 1.0e-12 * max(1.0, abs(response_low), abs(response_high)):
        raise ValueError("direct-RVE endpoint response span is too small for target sweep")

    target_records = []
    derivative_points = {
        lower + 0.25 * (upper - lower),
        lower + 0.50 * (upper - lower),
        lower + 0.75 * (upper - lower),
    }
    for fraction in target_fractions:
        target = response_low + float(fraction) * response_span
        optimization = optimize_volume_fraction(
            response_fn,
            target,
            lower,
            upper,
            grid_size=args.grid_size,
        )
        vf_star = float(optimization["volume_fraction"])
        derivative_points.add(vf_star)
        print(
            f"[TARGET SWEEP] fraction={fraction:.3f} vf={vf_star:.6f} "
            f"predicted={optimization['predicted_response']:.8g}",
            flush=True,
        )
        if vf_star not in direct_cache:
            rve_args.initial_rotation = _polar_rotation(F_np)
            direct_cache[vf_star] = _evaluate_direct_rve(vf_star, F_np, rve_args)
        direct_result = dict(direct_cache[vf_star])
        scale = max(abs(target), args.target_scale_floor)
        direct_error = abs(direct_result["response"] - target) / scale
        surrogate_to_direct_error = abs(
            direct_result["response"] - optimization["predicted_response"]
        ) / max(abs(direct_result["response"]), args.target_scale_floor)
        target_in_surrogate_range = bool(
            optimization["target_in_predicted_range"]
        )
        predicted_target_pass = (
            optimization["relative_target_error"] <= args.target_error_limit
        )
        direct_target_pass = direct_error <= args.target_error_limit
        if not target_in_surrogate_range:
            design_status = "surrogate_target_out_of_range"
        elif direct_target_pass and predicted_target_pass:
            design_status = "pass"
        elif not direct_target_pass:
            design_status = "direct_target_miss"
        else:
            design_status = "surrogate_target_miss"
        target_records.append({
            "target_fraction_of_endpoint_span": float(fraction),
            "target_response": float(target),
            "target_source": "fraction_of_direct_rve_endpoint_response_span",
            "predicted_volume_fraction": vf_star,
            "predicted_response": float(optimization["predicted_response"]),
            "predicted_relative_target_error": float(
                optimization["relative_target_error"]
            ),
            "target_in_predicted_response_range": target_in_surrogate_range,
            "direct_rve_result": direct_result,
            "direct_rve_relative_target_error": float(direct_error),
            "surrogate_to_direct_relative_error": float(surrogate_to_direct_error),
            "direct_rve_target_status": (
                "pass" if direct_target_pass else "target_miss"
            ),
            "status": design_status,
            "optimizer": optimization,
        })

    derivative_checks = _verify_response_derivative(
        value_and_derivative,
        response_fn,
        sorted(derivative_points),
        lower,
        upper,
        args.gradient_fd_step,
    )
    maximum_derivative_error = max(
        item["scaled_error"] for item in derivative_checks
    )
    if maximum_derivative_error > args.gradient_tolerance:
        raise RuntimeError(
            "pICNN response derivative check failed: "
            f"{maximum_derivative_error:.3e} exceeds "
            f"{args.gradient_tolerance:.3e}"
        )
    pass_count = sum(item["status"] == "pass" for item in target_records)
    out_of_range_count = sum(
        item["status"] == "surrogate_target_out_of_range"
        for item in target_records
    )
    direct_pass_count = sum(
        item["direct_rve_target_status"] == "pass"
        for item in target_records
    )
    report = {
        "schema_version": 1,
        "status": (
            "pass"
            if pass_count == len(target_records)
            else "no_surrogate_feasible_targets"
            if pass_count == 0 and out_of_range_count == len(target_records)
            else "partial"
        ),
        "inverse_design_acceptance_rule": (
            "A target passes only when it lies in the surrogate-predicted response "
            "range, the surrogate target error meets the limit, and the direct-RVE "
            "target error meets the limit. Direct-RVE target status is also reported "
            "separately."
        ),
        "validation_scope": (
            "multiple targets derived from the direct-RVE endpoint span; "
            "not external experimental targets"
        ),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "training_dataset_sha256": checkpoint.get("dataset_sha256"),
        "best_epoch": checkpoint.get("best_epoch"),
        "response": args.response,
        "stress_component_ij": [args.i, args.j] if args.response == "stress" else None,
        "deformation_gradient": F_np.tolist(),
        "deformation_determinant": determinant,
        "deformation_condition_number": condition_number,
        "volume_fraction_bounds": [lower, upper],
        "direct_rve_endpoint_results": endpoint_results,
        "direct_rve_endpoint_response_range": [response_low, response_high],
        "target_error_limit": float(args.target_error_limit),
        "targets": target_records,
        "summary": {
            "n_targets": len(target_records),
            "n_passed": int(pass_count),
            "n_missed": int(len(target_records) - pass_count),
            "n_surrogate_out_of_range": int(out_of_range_count),
            "n_direct_rve_target_passed": int(direct_pass_count),
            "n_direct_rve_target_missed": int(
                len(target_records) - direct_pass_count
            ),
            "maximum_direct_target_error": float(max(
                item["direct_rve_relative_target_error"]
                for item in target_records
            )),
            "maximum_surrogate_to_direct_error": float(max(
                item["surrogate_to_direct_relative_error"]
                for item in target_records
            )),
        },
        "verification": {
            "autograd_vs_finite_difference": derivative_checks,
            "maximum_scaled_derivative_error": float(maximum_derivative_error),
            "tolerance": float(args.gradient_tolerance),
            "status": "pass",
        },
        "rve_configuration": {
            "maxh": args.maxh,
            "mu": args.mu,
            "bulk_modulus": args.bulk_modulus,
            "displacement_order": args.displacement_order,
            "integration_order": args.integration_order,
            "microscopic_jacobian_floor": args.microscopic_jacobian_floor,
        },
    }
    _atomic_json(args.output, report)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "target_fraction_of_endpoint_span",
        "target_response",
        "predicted_volume_fraction",
        "predicted_response",
        "predicted_relative_target_error",
        "direct_rve_response",
        "direct_rve_relative_target_error",
        "surrogate_to_direct_relative_error",
        "target_in_predicted_response_range",
        "direct_rve_target_status",
        "status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in target_records:
            writer.writerow({
                "target_fraction_of_endpoint_span": item[
                    "target_fraction_of_endpoint_span"
                ],
                "target_response": item["target_response"],
                "predicted_volume_fraction": item["predicted_volume_fraction"],
                "predicted_response": item["predicted_response"],
                "predicted_relative_target_error": item[
                    "predicted_relative_target_error"
                ],
                "direct_rve_response": item["direct_rve_result"]["response"],
                "direct_rve_relative_target_error": item[
                    "direct_rve_relative_target_error"
                ],
                "surrogate_to_direct_relative_error": item[
                    "surrogate_to_direct_relative_error"
                ],
                "target_in_predicted_response_range": item[
                    "target_in_predicted_response_range"
                ],
                "direct_rve_target_status": item["direct_rve_target_status"],
                "status": item["status"],
            })
    print(json.dumps({
        "status": report["status"],
        "summary": report["summary"],
        "maximum_scaled_derivative_error": maximum_derivative_error,
        "report": str(args.output.resolve()),
        "csv": str(csv_path.resolve()),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--target-fractions", type=float, nargs="+", default=(0.25, 0.50, 0.75))
    parser.add_argument("--response", choices=("energy", "stress"), default="stress")
    parser.add_argument("--i", type=int, default=0)
    parser.add_argument("--j", type=int, default=0)
    parser.add_argument(
        "--deformation-gradient",
        type=float,
        nargs=9,
        default=(1.05, 0, 0, 0, 1, 0, 0, 0, 1),
    )
    parser.add_argument("--volume-fraction-bounds", type=float, nargs=2, default=(0.02, 0.10))
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--target-scale-floor", type=float, default=1.0e-8)
    parser.add_argument("--gradient-fd-step", type=float, default=1.0e-5)
    parser.add_argument("--gradient-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--target-error-limit", type=float, default=0.05)
    parser.add_argument("--max-condition-number", type=float, default=3.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--maxh", type=float, default=0.22)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--bulk-modulus", type=float, default=1.0e4)
    parser.add_argument("--displacement-order", type=int, default=2)
    parser.add_argument("--integration-order", type=int, default=10)
    parser.add_argument("--microscopic-jacobian-floor", type=float, default=1.0e-8)
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
