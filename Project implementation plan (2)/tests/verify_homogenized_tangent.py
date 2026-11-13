"""P2-V6 -- Verify the exact condensed prescribed-particle RVE tangent.

Run:
    python -m tests.verify_homogenized_tangent

Nonlinear Taylor remainders validate the tangent; perturbed nonlinear solves
are never used to construct or numerically differentiate the tangent.
"""
import argparse

import numpy as np
from ngsolve import CF, Det, Grad, InnerProduct, Inv

from src.homogenization import homogenized_first_piola
from src.rve import make_periodic_particle_cell_mesh
from src.rve_solver import solve_periodic_rigid_particle
from src.rve_tangent import condensed_macroscopic_tangent


def _matrix_cf(matrix):
    return CF(tuple(np.asarray(matrix).ravel()), dims=(3, 3))


def main(maxh=0.22):
    center = np.array((0.5, 0.5, 0.5))
    mu = 1.0
    bulk_modulus = 1.0e4
    base_gradient = 1.005 * np.eye(3)
    macro_gradient = _matrix_cf(base_gradient)
    mesh = make_periodic_particle_cell_mesh(center=center, maxh=maxh)
    print("P2-V6: equilibrating the base RVE", flush=True)
    solution, history, _, stress = solve_periodic_rigid_particle(
        mesh,
        macro_gradient,
        particle_displacement=CF(tuple((base_gradient - np.eye(3)) @ center)),
        mu=mu,
        bulk_modulus=bulk_modulus,
    )
    print("P2-V6: assembling and condensing the energy Hessian", flush=True)
    result = condensed_macroscopic_tangent(
        mesh,
        solution,
        macro_gradient,
        center=center,
        mu=mu,
        bulk_modulus=bulk_modulus,
    )
    fluctuation, pressure = solution.components
    recovered_stress = homogenized_first_piola(
        mesh, stress, fluctuation.space, center=center,
    )
    stress_error = np.linalg.norm(result.stress - recovered_stress) / max(
        1.0, np.linalg.norm(recovered_stress)
    )
    symmetry_error = np.linalg.norm(result.tangent - result.tangent.T) / max(
        1.0, np.linalg.norm(result.tangent)
    )

    # Independent differentiation of the explicit first-Piola expression,
    # rather than of the augmented energy used to assemble the Schur tangent.
    variable_gradient = (macro_gradient + Grad(fluctuation)).MakeVariable()
    variable_pressure = pressure.MakeVariable()
    jacobian = Det(variable_gradient)
    inverse_transpose = Inv(variable_gradient).trans
    invariant = InnerProduct(variable_gradient, variable_gradient)
    explicit_stress = (
        mu * jacobian ** (-2.0 / 3.0)
        * (variable_gradient - invariant / 3.0 * inverse_transpose)
        + variable_pressure * jacobian * inverse_transpose
    )
    differentiated_tangent = np.zeros((9, 9))
    print("P2-V6: checking all nine differentiated stress columns", flush=True)
    for column, sensitivity in enumerate(result.sensitivities):
        basis = np.zeros((3, 3))
        basis.ravel()[column] = 1.0
        gradient_rate = _matrix_cf(basis) + Grad(sensitivity.components[0])
        stress_rate = (
            explicit_stress.Diff(variable_gradient, gradient_rate)
            + explicit_stress.Diff(variable_pressure, sensitivity.components[1])
        ).Compile()
        differentiated_tangent[:, column] = homogenized_first_piola(
            mesh, stress_rate, fluctuation.space, center=center,
        ).ravel()
        print(f"  sensitivity column {column + 1}/9 checked", flush=True)
    sensitivity_error = np.linalg.norm(
        differentiated_tangent - result.tangent
    ) / max(1.0, np.linalg.norm(result.tangent))

    # Independent nonlinear check: a correct first-order prediction leaves a
    # second-order remainder. No stress difference quotient defines Abar.
    direction = np.array((
        (0.35, 0.40, -0.20),
        (0.15, -0.10, 0.25),
        (-0.05, 0.10, 0.20),
    ))
    direction /= np.linalg.norm(direction)
    linear_stress_rate = (result.tangent @ direction.ravel()).reshape(3, 3)
    steps = (1.0e-3, 5.0e-4, 2.5e-4)
    remainders = []
    perturbed_residuals = []
    print("P2-V6: checking nonlinear Taylor remainders", flush=True)
    for step in steps:
        perturbed_gradient = base_gradient + step * direction
        perturbed_solution, perturbed_history, _, perturbed_stress = (
            solve_periodic_rigid_particle(
                mesh,
                _matrix_cf(perturbed_gradient),
                particle_displacement=CF(tuple(
                    (perturbed_gradient - np.eye(3)) @ center
                )),
                mu=mu,
                bulk_modulus=bulk_modulus,
                verbose=False,
            )
        )
        perturbed_macro_stress = homogenized_first_piola(
            mesh,
            perturbed_stress,
            perturbed_solution.components[0].space,
            center=center,
        )
        remainders.append(np.linalg.norm(
            perturbed_macro_stress - result.stress - step * linear_stress_rate
        ))
        perturbed_residuals.append(perturbed_history[-1])
        print(f"  step {step:.2e}: remainder {remainders[-1]:.6e}", flush=True)
    orders = np.log2(np.asarray(remainders[:-1]) / remainders[1:])
    finest_relative_remainder = remainders[-1] / max(
        1.0, np.linalg.norm(result.stress)
    )

    print(f"\nmesh elements:                     {mesh.ne}")
    print(f"macroscopic tangent shape:         {result.tangent.shape}")
    print("flattening: row i*3+J, column k*3+L")
    print(f"energy-gradient stress error:      {stress_error:.3e}")
    print(f"major-symmetry error:              {symmetry_error:.3e}")
    print(f"independent sensitivity error:     {sensitivity_error:.3e}")
    print(f"max condensed linear residual:     {max(result.linear_residuals):.3e}")
    print(f"final nonlinear residual:          {history[-1]:.3e}")
    for step, remainder in zip(steps, remainders):
        print(f"Taylor step {step:.2e}: remainder   {remainder:.6e}")
    print("Taylor remainder orders:          ", "  ".join(f"{p:.3f}" for p in orders))
    print(f"finest relative Taylor remainder:  {finest_relative_remainder:.3e}")

    assert result.tangent.shape == (9, 9), "wrong macroscopic tangent shape"
    assert np.all(np.isfinite(result.tangent)), "nonfinite tangent entries"
    assert stress_error < 1.0e-10, "energy gradient disagrees with FE reaction stress"
    assert symmetry_error < 1.0e-10, "condensed energy Hessian is not symmetric"
    assert sensitivity_error < 1.0e-9, "independent stress sensitivities disagree"
    assert max(result.linear_residuals) < 1.0e-9, "linearized equilibrium failed"
    assert history[-1] < 1.0e-9, "base RVE is not converged"
    assert max(perturbed_residuals) < 1.0e-9, "Taylor-check RVE is not converged"
    assert np.all(np.isfinite(orders)) and min(orders) > 1.8, (
        "nonlinear Taylor remainder is not second order"
    )
    assert finest_relative_remainder < 1.0e-5, "Taylor prediction is inaccurate"
    print("P2-V6 PASS: exact condensed prescribed-particle tangent is consistent.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxh", type=float, default=0.22)
    main(**vars(parser.parse_args()))
