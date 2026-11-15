"""Mixed finite-strain matrix formulation for near-incompressible RVEs.

The matrix energy is written in displacement-pressure form

    Pi(F, p) = mu/2 (J^(-2/3) I1 - 3)
               + p (J - 1) - p^2/(2 kappa).

Stationarity with respect to pressure gives p = kappa (J - 1), while the
Taylor--Hood P2/P1 pair avoids the volumetric locking of a displacement-only
low-order formulation as kappa/mu becomes large.
"""
from ngsolve import (
    BilinearForm,
    Det,
    Grad,
    GridFunction,
    H1,
    Id,
    InnerProduct,
    Inv,
    Norm,
    Periodic,
    Projector,
    Trace,
    VectorH1,
    dx,
)


def make_mixed_space(mesh, displacement_order=2, periodic=False, dirichlet=""):
    """Create a Taylor--Hood displacement-pressure finite-element space."""
    if displacement_order < 2:
        raise ValueError("Taylor-Hood displacement order must be at least two")

    displacement = VectorH1(
        mesh,
        order=displacement_order,
        dirichlet=dirichlet,
    )
    pressure = H1(mesh, order=displacement_order - 1)
    if periodic:
        displacement = Periodic(displacement)
        pressure = Periodic(pressure)
    return displacement * pressure


def mixed_matrix_residual_form(fes, mu=1.0, bulk_modulus=1.0e4, measure=dx):
    """Build the nonlinear mixed residual whose linearization is the tangent."""
    if mu <= 0:
        raise ValueError("mu must be positive")
    if bulk_modulus <= 0:
        raise ValueError("bulk_modulus must be positive")

    (displacement, pressure), (test_displacement, test_pressure) = (
        fes.TrialFunction(),
        fes.TestFunction(),
    )
    deformation_gradient = Id(3) + Grad(displacement)
    jacobian = Det(deformation_gradient)
    first_invariant = Trace(
        deformation_gradient.trans * deformation_gradient
    )
    inverse_transpose = Inv(deformation_gradient).trans
    isochoric_stress = (
        mu
        * jacobian ** (-2.0 / 3.0)
        * (
            deformation_gradient
            - (first_invariant / 3.0) * inverse_transpose
        )
    )
    cofactor = jacobian * inverse_transpose
    total_stress = isochoric_stress + pressure * cofactor

    residual = BilinearForm(fes)
    residual += (
        InnerProduct(total_stress, Grad(test_displacement))
        + test_pressure
        * (jacobian - 1.0 - pressure / bulk_modulus)
    ) * measure
    return residual


def solve_mixed_matrix(
    mesh,
    boundary_displacement,
    mu=1.0,
    bulk_modulus=1.0e4,
    displacement_order=2,
    dirichlet=".*",
    tol=1.0e-11,
    maxit=15,
    verbose=True,
):
    """Solve a Dirichlet-driven mixed matrix problem by consistent Newton."""
    fes = make_mixed_space(
        mesh,
        displacement_order=displacement_order,
        dirichlet=dirichlet,
    )
    solution = GridFunction(fes)
    solution.components[0].Set(boundary_displacement)
    nonlinear_form = mixed_matrix_residual_form(
        fes,
        mu=mu,
        bulk_modulus=bulk_modulus,
    )

    free_projector = Projector(fes.FreeDofs(), True)
    residual = solution.vec.CreateVector()
    history = []
    for iteration in range(maxit):
        nonlinear_form.Apply(solution.vec, residual)
        residual.data = free_projector * residual
        residual_norm = Norm(residual)
        history.append(residual_norm)
        if verbose:
            print(f"Mixed Newton {iteration:2d}: |R| = {residual_norm:.3e}")
        if residual_norm < tol:
            break

        nonlinear_form.AssembleLinearization(solution.vec)
        update = nonlinear_form.mat.Inverse(
            fes.FreeDofs(),
            inverse="umfpack",
        ) * residual
        solution.vec.data -= update

    return solution, history
