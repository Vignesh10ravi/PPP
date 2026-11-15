"""Bounded pICNN inverse design with optional direct-RVE validation.

The optimizer searches particle volume fraction within the trained domain for
one prescribed deformation and scalar energy/stress response. If no target is
provided, direct RVE endpoint responses define a midpoint target. The selected
design and two nearby volume fractions are then checked with the direct RVE
solver. Run ``--self-check`` to verify the bounded search and pICNN response
derivative without a trained checkpoint or NGSolve solve.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.func import jacrev

from src.picnn import HyperelasticNN


DEFAULT_F = np.asarray(
    [[1.0, 0.05, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=float,
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
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary_path = Path(stream.name)
    temporary_path.replace(path)


def _response_tensor(model, deformation_gradient, volume_fraction, response, i, j):
    """Evaluate scalar energy or one first-Piola stress component."""
    if not isinstance(volume_fraction, torch.Tensor):
        volume_fraction = torch.as_tensor(
            volume_fraction,
            dtype=deformation_gradient.dtype,
            device=deformation_gradient.device,
        )
    context = volume_fraction.reshape(1, 1)

    def energy_single(f):
        return model.energy(f.unsqueeze(0), context).squeeze()

    if response == "energy":
        return energy_single(deformation_gradient)
    return jacrev(energy_single)(deformation_gradient)[i, j]


def _make_response_function(model, deformation_gradient, response, i, j):
    def evaluate(volume_fraction):
        with torch.enable_grad():
            value = _response_tensor(
                model,
                deformation_gradient,
                float(volume_fraction),
                response,
                i,
                j,
            )
        return float(value.detach().cpu())

    def value_and_derivative(volume_fraction):
        vf = torch.tensor(
            [float(volume_fraction)],
            dtype=deformation_gradient.dtype,
            device=deformation_gradient.device,
            requires_grad=True,
        )
        value = _response_tensor(
            model,
            deformation_gradient,
            vf,
            response,
            i,
            j,
        )
        derivative = torch.autograd.grad(value, vf, create_graph=False)[0]
        return float(value.detach().cpu()), float(derivative.detach().cpu())

    return evaluate, value_and_derivative


def _objective(response_fn, target, scale_floor):
    scale = max(abs(float(target)), float(scale_floor))

    def evaluate(volume_fraction):
        relative_error = (response_fn(volume_fraction) - target) / scale
        return relative_error * relative_error

    return evaluate


def _golden_section_minimize(function, lower, upper, tolerance, max_iterations):
    """Minimize a scalar function on a closed interval deterministically."""
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = float(lower)
    right = float(upper)
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = function(x1)
    f2 = function(x2)
    evaluations = 2

    for _ in range(max_iterations):
        if right - left <= tolerance:
            break
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = function(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = function(x2)
        evaluations += 1

    midpoint = 0.5 * (left + right)
    candidates = ((left, function(left)), (right, function(right)),
                  (midpoint, function(midpoint)))
    evaluations += 3
    best_x, best_value = min(candidates, key=lambda item: (item[1], item[0]))
    return best_x, best_value, evaluations


def optimize_volume_fraction(
    response_fn,
    target,
    lower,
    upper,
    grid_size=101,
    scale_floor=1.0e-8,
    tolerance=1.0e-10,
    max_iterations=100,
):
    """Grid-bracket every sampled local minimum, then refine each bracket."""
    if not np.isfinite([target, lower, upper]).all() or lower >= upper:
        raise ValueError("target and increasing finite volume-fraction bounds are required")
    if grid_size < 5:
        raise ValueError("grid_size must be at least five")

    response_cache = {}

    def evaluate_response(volume_fraction):
        key = float(volume_fraction)
        if key not in response_cache:
            response_cache[key] = float(response_fn(key))
        return response_cache[key]

    values = np.linspace(lower, upper, grid_size)
    predicted = np.asarray([evaluate_response(value) for value in values], dtype=float)
    if not np.isfinite(predicted).all():
        raise FloatingPointError("pICNN produced a nonfinite response in the search domain")
    if np.ptp(predicted) <= 1.0e-14:
        raise ValueError("pICNN response is constant over the requested volume-fraction range")
    target_in_predicted_range = bool(predicted.min() <= target <= predicted.max())

    objective = _objective(evaluate_response, target, scale_floor)
    losses = np.asarray([objective(value) for value in values], dtype=float)
    local_indices = []
    if losses[0] <= losses[1]:
        local_indices.append(0)
    for index in range(1, grid_size - 1):
        is_minimum = losses[index] <= losses[index - 1] and losses[index] <= losses[index + 1]
        is_strict_on_one_side = losses[index] < losses[index - 1] or losses[index] < losses[index + 1]
        if is_minimum and is_strict_on_one_side:
            local_indices.append(index)
    if losses[-1] <= losses[-2]:
        local_indices.append(grid_size - 1)
    if not local_indices:
        local_indices = [int(np.argmin(losses))]

    candidates = [(float(values[index]), float(losses[index])) for index in local_indices]
    for index in local_indices:
        bracket_lower = values[max(0, index - 1)]
        bracket_upper = values[min(grid_size - 1, index + 1)]
        if bracket_upper <= bracket_lower:
            continue
        result = _golden_section_minimize(
            objective,
            bracket_lower,
            bracket_upper,
            tolerance,
            max_iterations,
        )
        candidates.append((result[0], result[1]))

    best_vf, best_loss = min(
        candidates,
        key=lambda item: (item[1], item[0]),
    )
    scale = max(abs(float(target)), float(scale_floor))
    predicted_response = evaluate_response(best_vf)
    return {
        "volume_fraction": float(best_vf),
        "predicted_response": float(predicted_response),
        "target_response": float(target),
        "relative_target_error": float(abs(predicted_response - target) / scale),
        "predicted_response_range": [float(predicted.min()), float(predicted.max())],
        "target_in_predicted_range": target_in_predicted_range,
        "grid_size": int(grid_size),
        "sampled_local_minima": int(len(local_indices)),
        "evaluations": int(len(response_cache)),
        "search_grid": [
            {"volume_fraction": float(vf), "predicted_response": float(value)}
            for vf, value in zip(values, predicted)
        ],
    }


def _finite_difference_derivative(function, value, lower, upper, step):
    left_room = value - lower
    right_room = upper - value
    if left_room >= step and right_room >= step:
        return (function(value + step) - function(value - step)) / (2.0 * step)
    if right_room >= 2.0 * step:
        return (-3.0 * function(value) + 4.0 * function(value + step)
                - function(value + 2.0 * step)) / (2.0 * step)
    if left_room >= 2.0 * step:
        return (3.0 * function(value) - 4.0 * function(value - step)
                + function(value - 2.0 * step)) / (2.0 * step)
    raise ValueError("gradient-check step is too large near the volume-fraction bound")


def _verify_response_derivative(
    value_and_derivative,
    response_fn,
    points,
    lower,
    upper,
    step,
):
    records = []
    for point in points:
        analytical_value, analytical_derivative = value_and_derivative(point)
        finite_difference = _finite_difference_derivative(
            response_fn,
            point,
            lower,
            upper,
            step,
        )
        error = abs(analytical_derivative - finite_difference)
        scaled_error = error / max(1.0, abs(analytical_derivative), abs(finite_difference))
        records.append({
            "volume_fraction": float(point),
            "response": float(analytical_value),
            "autograd_derivative": float(analytical_derivative),
            "finite_difference_derivative": float(finite_difference),
            "scaled_error": float(scaled_error),
        })
    return records


def _load_model(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("checkpoint must contain a state_dict")
    model = HyperelasticNN(n_ctx=1).to(dtype=torch.float64, device="cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload, _sha256_file(checkpoint_path)


def _rve_response(sample, response, i, j):
    if response == "energy":
        return float(np.asarray(sample["energy"]))
    stress = np.asarray(sample["stress"], dtype=float).reshape(3, 3)
    return float(stress[i, j])


def _evaluate_direct_rve(volume_fraction, deformation_gradient, args):
    from src.free_particle import solve_free_rotating_particle
    from src.homogenization import homogenized_first_piola
    from src.rve import make_periodic_particle_cell_mesh
    from src.rve_dataset import (
        DatasetConfig,
        _homogenized_energy,
        microscopic_jacobian_extrema,
        particle_radius,
    )

    config = DatasetConfig(
        volume_fractions=(float(volume_fraction),),
        maxh=args.maxh,
        mu=args.mu,
        bulk_modulus=args.bulk_modulus,
        displacement_order=args.displacement_order,
        integration_order=args.integration_order,
        microscopic_jacobian_floor=args.microscopic_jacobian_floor,
    )
    config.validate()
    print(f"[DIRECT RVE] start vf={volume_fraction:.6f}", flush=True)
    mesh = getattr(args, "mesh", None)
    if mesh is None:
        mesh = make_periodic_particle_cell_mesh(
            radius=particle_radius(volume_fraction),
            maxh=args.maxh,
        )
    free_state = solve_free_rotating_particle(
        mesh,
        deformation_gradient,
        initial_rotation=getattr(args, "initial_rotation", np.eye(3)),
        mu=args.mu,
        bulk_modulus=args.bulk_modulus,
        displacement_order=args.displacement_order,
        integration_order=args.integration_order,
        verbose=getattr(args, "rve_verbose", True),
    )
    print(
        "[DIRECT RVE] equilibrium converged "
        f"vf={volume_fraction:.6f} mixed={free_state.newton_history[-1]:.3e} "
        f"rotation={free_state.rotational_residuals[-1]:.3e}",
        flush=True,
    )
    macro_stress = homogenized_first_piola(
        mesh,
        free_state.stress,
        free_state.solution.space.components[0],
        integration_order=args.integration_order,
    )
    print(f"[DIRECT RVE] homogenized stress recovered vf={volume_fraction:.6f}", flush=True)
    pressure = free_state.solution.components[1]
    energy = _homogenized_energy(
        mesh,
        free_state.deformation_gradient,
        pressure,
        args.mu,
        args.bulk_modulus,
        args.integration_order,
    )
    minimum_j, maximum_j = microscopic_jacobian_extrema(
        mesh,
        free_state.deformation_gradient,
        args.integration_order,
    )
    mixed_residual = float(free_state.newton_history[-1])
    rotation_residual = float(free_state.rotational_residuals[-1])
    minimum_clearance = 0.5 - particle_radius(volume_fraction)
    if mixed_residual >= 1.0e-8:
        raise RuntimeError(f"direct RVE mixed residual did not converge: {mixed_residual:.3e}")
    if rotation_residual >= 1.0e-8:
        raise RuntimeError(f"direct RVE rotation residual did not converge: {rotation_residual:.3e}")
    if minimum_j <= args.microscopic_jacobian_floor:
        raise RuntimeError(
            "direct RVE microscopic Jacobian did not meet its floor: "
            f"{minimum_j:.6e} <= {args.microscopic_jacobian_floor:.6e}"
        )
    if minimum_clearance <= 0.0:
        raise RuntimeError("direct RVE particle clearance is nonpositive")
    solver_status = "converged"
    response_sample = {
        "energy": np.asarray(energy),
        "stress": np.asarray(macro_stress),
    }
    return {
        "volume_fraction": float(volume_fraction),
        "response": _rve_response(response_sample, args.response, args.i, args.j),
        "energy": float(energy),
        "stress": np.asarray(macro_stress, dtype=float).reshape(3, 3).tolist(),
        "rotation": np.asarray(free_state.rotation, dtype=float).reshape(3, 3).tolist(),
        "minimum_microscopic_jacobian": float(minimum_j),
        "maximum_microscopic_jacobian": float(maximum_j),
        "minimum_particle_clearance": float(minimum_clearance),
        "mixed_residual": mixed_residual,
        "rotation_residual": rotation_residual,
        "solver_status": solver_status,
    }


def _run_self_check():
    lower, upper = 0.02, 0.10
    expected_vf = 0.073
    response_fn = lambda vf: 1.7 * vf + 0.2
    target = response_fn(expected_vf)
    result = optimize_volume_fraction(
        response_fn,
        target,
        lower,
        upper,
        grid_size=81,
    )
    if abs(result["volume_fraction"] - expected_vf) > 1.0e-8:
        raise AssertionError("bounded optimizer failed the analytic linear-response check")
    out_of_range_result = optimize_volume_fraction(
        response_fn,
        10.0,
        lower,
        upper,
        grid_size=81,
    )
    if out_of_range_result["target_in_predicted_range"] or abs(
        out_of_range_result["volume_fraction"] - upper
    ) > 1.0e-8:
        raise AssertionError("out-of-range target was not reported at the closest bound")

    multimodal = lambda vf: (vf - 0.035) ** 2 * (vf - 0.085) ** 2
    multimodal_result = optimize_volume_fraction(
        multimodal,
        0.0,
        lower,
        upper,
        grid_size=81,
    )
    if min(
        abs(multimodal_result["volume_fraction"] - 0.035),
        abs(multimodal_result["volume_fraction"] - 0.085),
    ) > 1.0e-7:
        raise AssertionError("bounded optimizer failed the multimodal response check")

    model = HyperelasticNN(n_ctx=1).to(dtype=torch.float64, device="cpu")
    model.eval()
    deformation_gradient = torch.as_tensor(DEFAULT_F, dtype=torch.float64)
    response_fn, value_and_derivative = _make_response_function(
        model,
        deformation_gradient,
        "stress",
        0,
        1,
    )
    derivative_records = _verify_response_derivative(
        value_and_derivative,
        response_fn,
        [0.035, 0.065, 0.095],
        0.02,
        0.10,
        1.0e-5,
    )
    maximum_error = max(record["scaled_error"] for record in derivative_records)
    if maximum_error > 1.0e-5:
        raise AssertionError(
            f"pICNN context derivative check exceeded tolerance: {maximum_error:.3e}"
        )
    print(json.dumps({
        "status": "pass",
        "analytic_optimizer_vf": result["volume_fraction"],
        "out_of_range_closest_vf": out_of_range_result["volume_fraction"],
        "multimodal_optimizer_vf": multimodal_result["volume_fraction"],
        "pICNN_context_derivative_max_scaled_error": maximum_error,
        "derivative_checks": derivative_records,
    }, indent=2))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/rve_inverse_design.json"))
    parser.add_argument(
        "--target",
        type=float,
        help="desired response; default is the midpoint of direct-RVE endpoint responses",
    )
    parser.add_argument("--response", choices=("energy", "stress"), default="stress")
    parser.add_argument("--i", type=int, default=0, help="stress row index")
    parser.add_argument("--j", type=int, default=1, help="stress column index")
    parser.add_argument("--deformation-gradient", type=float, nargs=9, default=DEFAULT_F.ravel())
    parser.add_argument("--volume-fraction-bounds", type=float, nargs=2, default=(0.02, 0.10))
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--neighbor-distance", type=float, default=0.01)
    parser.add_argument("--target-scale-floor", type=float, default=1.0e-8)
    parser.add_argument("--gradient-fd-step", type=float, default=1.0e-5)
    parser.add_argument("--gradient-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--target-error-limit", type=float, default=0.05)
    parser.add_argument("--maxh", type=float, default=0.22)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--bulk-modulus", type=float, default=1.0e4)
    parser.add_argument("--displacement-order", type=int, default=2)
    parser.add_argument("--integration-order", type=int, default=10)
    parser.add_argument("--microscopic-jacobian-floor", type=float, default=1.0e-8)
    parser.add_argument("--max-condition-number", type=float, default=3.0)
    parser.add_argument("--surrogate-only", action="store_true",
                        help="skip direct RVE validation; result is only a candidate")
    parser.add_argument("--self-check", action="store_true",
                        help="run analytic optimizer and autograd checks without a checkpoint")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.self_check:
        _run_self_check()
        return
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --self-check is used")
    if args.target is not None and not np.isfinite(args.target):
        raise SystemExit("--target must be finite")
    if args.target is None and args.surrogate_only:
        raise SystemExit("--target is required when --surrogate-only is used")
    if args.response == "stress" and not (0 <= args.i < 3 and 0 <= args.j < 3):
        raise SystemExit("stress indices --i and --j must each be in {0, 1, 2}")
    if (
        args.neighbor_distance <= 0
        or args.target_scale_floor <= 0
        or args.gradient_fd_step <= 0
        or args.gradient_tolerance <= 0
        or args.target_error_limit < 0
    ):
        raise SystemExit(
            "neighbor distance, scales, and tolerances must be positive; "
            "target error limit cannot be negative"
        )

    lower, upper = map(float, args.volume_fraction_bounds)
    if not (0.0 < lower < upper < np.pi / 6.0):
        raise SystemExit("volume-fraction bounds must satisfy 0 < lower < upper < pi/6")
    deformation_gradient_np = np.asarray(args.deformation_gradient, dtype=float).reshape(3, 3)
    determinant = float(np.linalg.det(deformation_gradient_np))
    condition_number = float(np.linalg.cond(deformation_gradient_np))
    if not np.isfinite(deformation_gradient_np).all() or determinant <= 0.0:
        raise SystemExit("deformation gradient must be finite and orientation-preserving")
    if condition_number > args.max_condition_number:
        raise SystemExit(
            f"deformation condition number {condition_number:.6g} exceeds "
            f"the configured limit {args.max_condition_number:.6g}"
        )

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model, checkpoint, checkpoint_sha256 = _load_model(args.checkpoint)
    deformation_gradient = torch.as_tensor(deformation_gradient_np, dtype=torch.float64)

    endpoint_results = []
    if not args.surrogate_only:
        endpoint_results = [
            _evaluate_direct_rve(vf, deformation_gradient_np, args)
            for vf in (lower, upper)
        ]
        endpoint_responses = [item["response"] for item in endpoint_results]
        endpoint_span = max(endpoint_responses) - min(endpoint_responses)
        if endpoint_span <= 1.0e-12 * max(1.0, *map(abs, endpoint_responses)):
            raise RuntimeError(
                "direct-RVE endpoint responses do not vary enough to define a useful target"
            )
        if args.target is None:
            target = 0.5 * (endpoint_responses[0] + endpoint_responses[1])
            target_source = "midpoint_of_direct_rve_endpoint_responses"
        else:
            target = float(args.target)
            if not min(endpoint_responses) <= target <= max(endpoint_responses):
                raise ValueError(
                    "explicit target is outside the direct-RVE endpoint response range "
                    f"[{min(endpoint_responses):.8g}, {max(endpoint_responses):.8g}]"
                )
            target_source = "user_specified_with_direct_rve_feasibility_check"
        for item in endpoint_results:
            item["relative_target_error"] = abs(item["response"] - target) / max(
                abs(target), args.target_scale_floor
            )
        print(
            f"[DESIGN] endpoint responses={endpoint_responses}; "
            f"target={target:.8g} source={target_source}",
            flush=True,
        )
    else:
        target = float(args.target)
        target_source = "user_specified_surrogate_only"

    response_fn, value_and_derivative = _make_response_function(
        model,
        deformation_gradient,
        args.response,
        args.i,
        args.j,
    )
    optimization = optimize_volume_fraction(
        response_fn,
        target,
        lower,
        upper,
        grid_size=args.grid_size,
        scale_floor=args.target_scale_floor,
    )
    print(
        "[DESIGN] predicted optimum "
        f"vf={optimization['volume_fraction']:.6f} "
        f"response={optimization['predicted_response']:.8g} "
        f"target={target:.8g} in_predicted_range="
        f"{optimization['target_in_predicted_range']}",
        flush=True,
    )

    derivative_points = sorted(set([
        lower + 0.25 * (upper - lower),
        lower + 0.50 * (upper - lower),
        lower + 0.75 * (upper - lower),
        optimization["volume_fraction"],
    ]))
    derivative_checks = _verify_response_derivative(
        value_and_derivative,
        response_fn,
        derivative_points,
        lower,
        upper,
        args.gradient_fd_step,
    )
    maximum_derivative_error = max(item["scaled_error"] for item in derivative_checks)
    if maximum_derivative_error > args.gradient_tolerance:
        raise RuntimeError(
            "autograd response derivative check failed: "
            f"max scaled error {maximum_derivative_error:.3e} exceeds "
            f"{args.gradient_tolerance:.3e}"
        )

    report = {
        "schema_version": 1,
        "status": "candidate_only" if args.surrogate_only else "running_direct_rve_validation",
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "training_dataset_sha256": checkpoint.get("dataset_sha256"),
        "best_epoch": checkpoint.get("best_epoch"),
        "response": args.response,
        "stress_component_ij": [args.i, args.j] if args.response == "stress" else None,
        "target_response": float(target),
        "target_source": target_source,
        "deformation_gradient": deformation_gradient_np.tolist(),
        "deformation_determinant": determinant,
        "deformation_condition_number": condition_number,
        "volume_fraction_bounds": [lower, upper],
        "optimization": optimization,
        "verification": {
            "autograd_vs_finite_difference": derivative_checks,
            "maximum_scaled_derivative_error": float(maximum_derivative_error),
            "tolerance": float(args.gradient_tolerance),
            "status": "pass",
        },
        "validation": {
            "status": "not_run" if args.surrogate_only else "pending",
            "endpoint_rve_results": endpoint_results,
            "direct_rve_results": [],
            "target_error_limit": float(args.target_error_limit),
        },
        "rve_configuration": {
            "maxh": float(args.maxh),
            "mu": float(args.mu),
            "bulk_modulus": float(args.bulk_modulus),
            "displacement_order": int(args.displacement_order),
            "integration_order": int(args.integration_order),
            "microscopic_jacobian_floor": float(args.microscopic_jacobian_floor),
        },
    }

    if not args.surrogate_only:
        delta = min(args.neighbor_distance, 0.25 * (upper - lower))
        selected_vfs = sorted(set([
            max(lower, optimization["volume_fraction"] - delta),
            optimization["volume_fraction"],
            min(upper, optimization["volume_fraction"] + delta),
        ]))
        direct_cache = {item["volume_fraction"]: item for item in endpoint_results}
        for vf in selected_vfs:
            if vf not in direct_cache:
                direct_cache[vf] = _evaluate_direct_rve(vf, deformation_gradient_np, args)
        direct_results = [direct_cache[vf] for vf in selected_vfs]
        scale = max(abs(target), args.target_scale_floor)
        for item in direct_results:
            item["relative_target_error"] = abs(item["response"] - target) / scale
        optimum_result = min(
            direct_results,
            key=lambda item: abs(item["volume_fraction"] - optimization["volume_fraction"]),
        )
        achieved_error = float(optimum_result["relative_target_error"])
        local_error_check = (
            len(direct_results) == 1
            or all(
                achieved_error <= item["relative_target_error"] + 1.0e-12
                for item in direct_results
                if item is not optimum_result
            )
        )
        if achieved_error <= args.target_error_limit and local_error_check:
            validation_status = "pass"
        elif achieved_error > args.target_error_limit:
            validation_status = "target_miss"
        else:
            validation_status = "local_check_failed"
        report["validation"] = {
            "status": validation_status,
            "endpoint_rve_results": endpoint_results,
            "direct_rve_results": direct_results,
            "verified_optimum_response": float(optimum_result["response"]),
            "verified_target_relative_error": achieved_error,
            "target_error_limit": float(args.target_error_limit),
            "local_target_error_check": bool(local_error_check),
        }

    _atomic_json(args.output, report)
    print(json.dumps({
        "report": str(args.output.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "predicted_volume_fraction": optimization["volume_fraction"],
        "predicted_response": optimization["predicted_response"],
        "validation_status": report["validation"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
