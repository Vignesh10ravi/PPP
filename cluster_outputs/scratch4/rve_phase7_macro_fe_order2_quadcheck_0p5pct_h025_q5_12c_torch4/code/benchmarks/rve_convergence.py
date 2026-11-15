"""Checkpointed mesh and quadrature convergence study for the periodic RVE.

Example on the HPC cluster::

    python -m benchmarks.rve_convergence \
      --output reports/rve_convergence.json

The reference for each case and volume fraction is the finest requested mesh
with the highest requested integration order. The report stores energy, stress,
tangent, rotation, microscopic-J extrema, residuals, and relative errors against
that reference. A completed setting is checkpointed to the JSON file, so a
restarted run reuses completed cases. Failed cases are also kept and skipped by
default; pass ``--retry-failed`` to retry them deliberately.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ngsolve import Det, GridFunction
from ngsolve.comp import IntegrationRuleSpace

from src.free_particle import solve_free_rotating_particle
from src.rve import make_periodic_particle_cell_mesh
from src.rve_dataset import _atomic_json, _homogenized_energy, particle_radius
from src.rve_tangent import condensed_free_rotation_tangent


def _matrix_cf(matrix):
    from ngsolve import CF

    return CF(tuple(np.asarray(matrix, dtype=float).ravel()), dims=(3, 3))


def _polar_rotation(matrix):
    """Return the proper rotation in the polar decomposition of a gradient."""
    left, _, right_transpose = np.linalg.svd(np.asarray(matrix, dtype=float))
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return rotation


def _load_cases():
    identity = np.eye(3)
    return {
        "hydrostatic_tension": 1.05 * identity,
        "hydrostatic_compression": 0.95 * identity,
        "uniaxial_tension": np.diag((1.08, 1.0, 1.0)),
        "biaxial_tension": np.diag((1.04, 1.04, 1.0)),
        "simple_shear": np.array(
            ((1.0, 0.10, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        ),
        "combined_nonsymmetric": np.array(
            ((1.04, 0.06, -0.02), (0.015, 0.98, 0.03), (-0.01, 0.02, 1.03))
        ),
        # The last two reproduce difficult accepted states from seed 1729.
        "boundary_state_77": np.array(
            ((0.914652137039033, 0.159471010151787, 0.107330871658285),
             (0.121485439216357, 0.922234491495797, -0.185146323741166),
             (0.062217426261375, -0.109112254951983, 0.960429358028566))
        ),
        "boundary_state_79": np.array(
            ((0.818017940629399, -0.157173921538109, 0.164308838411521),
             (-0.115748537177731, 0.920118891269092, 0.117002764466739),
             (0.155008760720238, -0.124064718099128, 0.808093984358208))
        ),
    }


def _key(case_name, volume_fraction, maxh, integration_order):
    return (
        f"{case_name}|vf={volume_fraction:.12g}|h={maxh:.12g}|"
        f"q={integration_order}"
    )


def _quadrature_jacobian_extrema(mesh, deformation_gradient, order):
    quadrature_space = IntegrationRuleSpace(mesh, order=order)
    sampled_jacobian = GridFunction(quadrature_space)
    sampled_jacobian.Interpolate(Det(deformation_gradient))
    values = np.asarray(sampled_jacobian.vec.FV().NumPy(), dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise FloatingPointError("quadrature Jacobian samples are empty or nonfinite")
    return float(values.min()), float(values.max())


def _solve_case(mesh, macro_gradient, volume_fraction, maxh, order, mu, bulk_modulus):
    free_state = solve_free_rotating_particle(
        mesh,
        macro_gradient,
        initial_rotation=_polar_rotation(macro_gradient),
        mu=mu,
        bulk_modulus=bulk_modulus,
        displacement_order=2,
        integration_order=order,
        verbose=False,
    )
    tangent_result = condensed_free_rotation_tangent(
        mesh,
        free_state.solution,
        _matrix_cf(macro_gradient),
        free_state.rotation,
        mu=mu,
        bulk_modulus=bulk_modulus,
        integration_order=order,
    )
    _, pressure = free_state.solution.components
    energy = _homogenized_energy(
        mesh,
        free_state.deformation_gradient,
        pressure,
        mu,
        bulk_modulus,
        order,
    )
    minimum_j, maximum_j = _quadrature_jacobian_extrema(
        mesh,
        free_state.deformation_gradient,
        order,
    )
    return {
        "status": "ok",
        "volume_fraction": float(volume_fraction),
        "maxh": float(maxh),
        "integration_order": int(order),
        "mesh_elements": int(mesh.ne),
        "particle_clearance": float(0.5 - particle_radius(volume_fraction)),
        "energy": float(energy),
        "stress": np.asarray(tangent_result.stress).tolist(),
        "tangent": np.asarray(tangent_result.tangent).tolist(),
        "rotation": np.asarray(free_state.rotation).tolist(),
        "minimum_quadrature_j": minimum_j,
        "maximum_quadrature_j": maximum_j,
        "mixed_residual": float(free_state.newton_history[-1]),
        "mixed_newton_iterations": int(len(free_state.newton_history) - 1),
        "rotation_residual": float(free_state.rotational_residuals[-1]),
        "rotation_iterations": int(len(free_state.rotational_residuals)),
        "condensed_residual": float(max(tangent_result.linear_residuals)),
        "tangent_symmetry_error": float(
            np.linalg.norm(tangent_result.tangent - tangent_result.tangent.T)
            / max(1.0, np.linalg.norm(tangent_result.tangent))
        ),
    }


def _relative_error(value, reference):
    value = np.asarray(value, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return float(
        np.linalg.norm(value - reference)
        / max(float(np.linalg.norm(reference)), 1.0e-12)
    )


def _add_reference_errors(results, case_names, volume_fractions, mesh_sizes, orders):
    lookup = {entry["key"]: entry for entry in results}
    reference_h = min(mesh_sizes)
    reference_order = max(orders)
    for case_name in case_names:
        for volume_fraction in volume_fractions:
            reference_key = _key(
                case_name, volume_fraction, reference_h, reference_order
            )
            reference_entry = lookup.get(reference_key)
            if (
                not reference_entry
                or reference_entry["result"].get("status") != "ok"
            ):
                continue
            reference = reference_entry["result"]
            for entry in results:
                if (
                    entry["case"] != case_name
                    or entry["result"].get("volume_fraction") != volume_fraction
                    or entry["result"].get("status") != "ok"
                ):
                    continue
                current = entry["result"]
                entry["errors_from_finest"] = {
                    "energy_relative": _relative_error(
                        current["energy"], reference["energy"]
                    ),
                    "stress_relative": _relative_error(
                        current["stress"], reference["stress"]
                    ),
                    "tangent_relative": _relative_error(
                        current["tangent"], reference["tangent"]
                    ),
                    "rotation_relative": _relative_error(
                        current["rotation"], reference["rotation"]
                    ),
                    "minimum_j_absolute": abs(
                        current["minimum_quadrature_j"]
                        - reference["minimum_quadrature_j"]
                    ),
                }


def _recommend_resolution(results, case_names, volume_fractions, mesh_sizes, orders):
    for maxh in sorted(mesh_sizes, reverse=True):
        for order in sorted(orders):
            passed_every_case = True
            for case_name in case_names:
                for volume_fraction in volume_fractions:
                    key = _key(case_name, volume_fraction, maxh, order)
                    match = next(
                        (entry for entry in results if entry["key"] == key), None
                    )
                    errors = None if match is None else match.get("errors_from_finest")
                    if (
                        match is None
                        or match["result"].get("status") != "ok"
                        or errors is None
                        or errors["energy_relative"] >= 0.002
                        or errors["stress_relative"] >= 0.005
                        or errors["tangent_relative"] >= 0.01
                    ):
                        passed_every_case = False
                        break
                if not passed_every_case:
                    break
            if passed_every_case:
                return {"maxh": float(maxh), "integration_order": int(order)}
    return None


def main(
    output="reports/rve_convergence.json",
    volume_fractions=(0.02, 0.10),
    mesh_sizes=(0.30, 0.22, 0.16),
    integration_orders=(6, 10),
    cases=None,
    mu=1.0,
    bulk_modulus=1.0e4,
    retry_failed=False,
):
    all_cases = _load_cases()
    case_names = tuple(all_cases) if cases is None else tuple(cases)
    unknown = set(case_names).difference(all_cases)
    if unknown:
        raise ValueError(f"unknown load cases: {sorted(unknown)}")
    volume_fractions = tuple(float(value) for value in volume_fractions)
    mesh_sizes = tuple(float(value) for value in mesh_sizes)
    integration_orders = tuple(int(value) for value in integration_orders)
    if not volume_fractions or not mesh_sizes or not integration_orders:
        raise ValueError("volume fractions, mesh sizes, and orders must be nonempty")

    config = {
        "volume_fractions": list(volume_fractions),
        "mesh_sizes": list(mesh_sizes),
        "integration_orders": list(integration_orders),
        "cases": list(case_names),
        "mu": float(mu),
        "bulk_modulus": float(bulk_modulus),
        "displacement_order": 2,
        "accuracy_thresholds": {
            "energy_relative": 0.002,
            "stress_relative": 0.005,
            "tangent_relative": 0.01,
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results_by_key = {}
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("config") != config:
            raise ValueError(
                f"{output} belongs to a different convergence-study configuration"
            )
        results_by_key = {
            entry["key"]: entry
            for entry in existing.get("results", [])
            if (
                not retry_failed
                or entry.get("result", {}).get("status") == "ok"
            )
        }

    total = (
        len(volume_fractions)
        * len(mesh_sizes)
        * len(integration_orders)
        * len(case_names)
    )
    completed = len(results_by_key)
    for volume_fraction in volume_fractions:
        radius = particle_radius(volume_fraction)
        for maxh in mesh_sizes:
            mesh = make_periodic_particle_cell_mesh(radius=radius, maxh=maxh)
            for order in integration_orders:
                for case_name in case_names:
                    key = _key(case_name, volume_fraction, maxh, order)
                    if key in results_by_key:
                        continue
                    macro_gradient = all_cases[case_name]
                    print(
                        f"[{completed + 1}/{total}] {case_name} "
                        f"vf={volume_fraction:.3f} maxh={maxh:.3f} q={order}",
                        flush=True,
                    )
                    try:
                        result = _solve_case(
                            mesh,
                            macro_gradient,
                            volume_fraction,
                            maxh,
                            order,
                            mu,
                            bulk_modulus,
                        )
                    except Exception as exc:
                        result = {
                            "status": "failed",
                            "volume_fraction": volume_fraction,
                            "maxh": maxh,
                            "integration_order": order,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    results_by_key[key] = {
                        "key": key,
                        "case": case_name,
                        "deformation_gradient": macro_gradient.tolist(),
                        "result": result,
                    }
                    completed += 1
                    _atomic_json(output, {
                        "config": config,
                        "results": list(results_by_key.values()),
                        "recommended_resolution": None,
                    })

    results = list(results_by_key.values())
    _add_reference_errors(
        results,
        case_names,
        volume_fractions,
        mesh_sizes,
        integration_orders,
    )
    recommendation = _recommend_resolution(
        results,
        case_names,
        volume_fractions,
        mesh_sizes,
        integration_orders,
    )
    _atomic_json(output, {
        "config": config,
        "results": results,
        "recommended_resolution": recommendation,
    })
    print(f"Study report: {output}", flush=True)
    print(f"Recommended resolution: {recommendation}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/rve_convergence.json")
    parser.add_argument(
        "--volume-fractions", type=float, nargs="+", default=(0.02, 0.10)
    )
    parser.add_argument(
        "--mesh-sizes", type=float, nargs="+", default=(0.30, 0.22, 0.16)
    )
    parser.add_argument(
        "--integration-orders", type=int, nargs="+", default=(6, 10)
    )
    parser.add_argument("--cases", nargs="+", choices=tuple(_load_cases()))
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--bulk-modulus", type=float, default=1.0e4)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="rerun cases previously recorded as failed in the output JSON",
    )
    main(**vars(parser.parse_args()))
