"""Exact condensed macroscopic tangent for the prescribed rigid-particle RVE.

Augment the mixed energy Hessian with nine macroscopic coordinates and form

    Abar = (H_FF - H_Fz H_zz^-1 H_zF) / cell_volume.

The internal coordinates z are the free periodic fluctuation and pressure
DOFs. NGSolve differentiates the energy symbolically; no finite differences,
dense micro-matrix inverse, or post-hoc symmetrization are used.

The particle orientation is prescribed and its center follows (Fbar-I) X_c,
as in P2-V5. Free particle rotations are not condensed by this formulation.
"""
from dataclasses import dataclass

import numpy as np
from ngsolve import (
    BilinearForm,
    CF,
    Det,
    ET,
    FESpace,
    Grad,
    GridFunction,
    IntegrationRule,
    InnerProduct,
    Norm,
    NumberSpace,
    Projector,
    Variation,
    dx,
)

from .homogenization import particle_macro_lift


@dataclass(frozen=True)
class MacroscopicTangentResult:
    stress: np.ndarray
    tangent: np.ndarray
    sensitivities: tuple
    linear_residuals: np.ndarray
    equilibrium_residual: float


def condensed_macroscopic_tangent(
    mesh,
    solution,
    macro_gradient,
    center=(0.5, 0.5, 0.5),
    mu=1.0,
    bulk_modulus=1.0e4,
    cell_volume=1.0,
    integration_order=10,
    equilibrium_tol=1.0e-8,
):
    """Return Pbar, row-major 9x9 dPbar/dFbar, and nine mixed sensitivities.

    ``solution`` must be an equilibrated mixed RVE state with the same
    material, quadrature, and prescribed particle motion as P2-V5.
    Each sensitivity contains the TOTAL fluctuation derivative, including
    its nonzero particle trace, and the pressure derivative.
    """
    if mu <= 0 or bulk_modulus <= 0 or cell_volume <= 0:
        raise ValueError("material moduli and cell_volume must be positive")

    fluctuation, pressure = solution.components
    displacement_space, pressure_space = solution.space.components
    macro_basis = []
    lifts = []
    for column in range(9):
        basis = np.zeros((3, 3))
        basis.ravel()[column] = 1.0
        macro_basis.append(CF(tuple(basis.ravel()), dims=(3, 3)))
        lifts.append(particle_macro_lift(
            mesh, displacement_space, basis, center=center,
        ))

    # Global NumberSpace coordinates add just nine DOFs, not nine fields
    # resolved over the mesh. Only internal free DOFs enter the factorization.
    augmented_space = FESpace(
        [displacement_space, pressure_space]
        + [NumberSpace(mesh) for _ in range(9)]
    )
    delta_fluctuation, delta_pressure, *macro_coordinates = (
        augmented_space.TrialFunction()
    )
    deformation_gradient = macro_gradient + Grad(fluctuation)
    deformation_gradient += Grad(delta_fluctuation)
    for coordinate, basis, lift in zip(macro_coordinates, macro_basis, lifts):
        deformation_gradient += coordinate * (basis + Grad(lift))
    total_pressure = pressure + delta_pressure
    jacobian = Det(deformation_gradient)
    first_invariant = InnerProduct(deformation_gradient, deformation_gradient)
    density = (
        0.5 * mu * (jacobian ** (-2.0 / 3.0) * first_invariant - 3.0)
        + total_pressure * (jacobian - 1.0)
        - total_pressure ** 2 / (2.0 * bulk_modulus)
    )
    measure = dx(intrules={
        ET.TET: IntegrationRule(ET.TET, order=integration_order),
    })
    energy = BilinearForm(augmented_space)
    energy += Variation(density.Compile() * measure)
    zero_increment = GridFunction(augmented_space)
    gradient = zero_increment.vec.CreateVector()
    energy.Apply(zero_increment.vec, gradient)
    energy.AssembleLinearization(zero_increment.vec)

    macro_offset = displacement_space.ndof + pressure_space.ndof
    macro_dofs = np.arange(macro_offset, macro_offset + 9)
    internal_free = augmented_space.FreeDofs()
    for dof in macro_dofs:
        internal_free[int(dof)] = False
    internal_projector = Projector(internal_free, True)
    equilibrium_residual = Norm(internal_projector * gradient)
    if equilibrium_residual > equilibrium_tol:
        raise ValueError(
            f"RVE is not equilibrated: free residual {equilibrium_residual:.3e}"
        )

    stress = gradient.FV().NumPy()[macro_dofs].copy().reshape(3, 3) / cell_volume
    inverse = energy.mat.Inverse(internal_free, inverse="umfpack")
    tangent = np.zeros((9, 9))
    residuals = np.zeros(9)
    sensitivities = []
    for column, macro_dof in enumerate(macro_dofs):
        macro_increment = zero_increment.vec.CreateVector()
        macro_increment[:] = 0.0
        macro_increment.FV().NumPy()[macro_dof] = 1.0
        forcing = zero_increment.vec.CreateVector()
        forcing.data = energy.mat * macro_increment
        relaxation = GridFunction(augmented_space)
        relaxation.vec.data = -(inverse * forcing)
        total_increment = zero_increment.vec.CreateVector()
        total_increment.data = macro_increment + relaxation.vec
        response = zero_increment.vec.CreateVector()
        response.data = energy.mat * total_increment

        tangent[:, column] = response.FV().NumPy()[macro_dofs] / cell_volume
        residuals[column] = Norm(internal_projector * response) / max(
            1.0, Norm(internal_projector * forcing)
        )
        sensitivity = GridFunction(solution.space)
        sensitivity.components[0].vec.data = (
            relaxation.components[0].vec + lifts[column].vec
        )
        sensitivity.components[1].vec.data = relaxation.components[1].vec
        sensitivities.append(sensitivity)

    return MacroscopicTangentResult(
        stress=stress,
        tangent=tangent,
        sensitivities=tuple(sensitivities),
        linear_residuals=residuals,
        equilibrium_residual=equilibrium_residual,
    )


def condensed_free_rotation_tangent(
    mesh,
    solution,
    macro_gradient,
    rotation,
    center=(0.5, 0.5, 0.5),
    mu=1.0,
    bulk_modulus=1.0e4,
    cell_volume=1.0,
    integration_order=10,
    equilibrium_tol=1.0e-8,
):
    """Return the consistent macroscopic tangent with particle rotation free.

    The discrete mixed energy is augmented by nine macroscopic deformation
    coordinates and three left-multiplicative particle-rotation coordinates.
    All internal fluctuation, pressure, and rotation coordinates are relaxed
    by one Schur-complement solve. The rotation parameterization uses
    ``exp([dq]x) = I + [dq]x + 0.5 [dq]x**2 + O(|dq|**3)``, matching the
    rotational equilibrium formulation in :mod:`src.free_particle`.
    """
    from ngsolve import Id, x, y, z
    from .rve import PARTICLE_BOUNDARY

    if mu <= 0 or bulk_modulus <= 0 or cell_volume <= 0:
        raise ValueError("material moduli and cell_volume must be positive")

    rotation = np.asarray(rotation, dtype=float)
    center = np.asarray(center, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if center.shape != (3,):
        raise ValueError("center must have shape (3,)")

    fluctuation, pressure = solution.components
    displacement_space, pressure_space = solution.space.components
    particle_boundary = mesh.Boundaries(PARTICLE_BOUNDARY)
    reference_arm = CF((x - center[0], y - center[1], z - center[2]))
    rotated_arm = CF(tuple(rotation.ravel()), dims=(3, 3)) * reference_arm

    # Boundary lifts for the three infinitesimal left-rotation coordinates.
    rotation_lifts = []
    for row in range(3):
        for column in range(3):
            lift = GridFunction(displacement_space)
            lift.Set(
                CF(tuple(
                    rotated_arm[column] if component == row else 0.0
                    for component in range(3)
                )),
                definedon=particle_boundary,
            )
            rotation_lifts.append(lift)

    macro_bases = []
    macro_lifts = []
    for column in range(9):
        basis = np.zeros((3, 3))
        basis.ravel()[column] = 1.0
        macro_bases.append(CF(tuple(basis.ravel()), dims=(3, 3)))
        macro_lifts.append(
            particle_macro_lift(
                mesh, displacement_space, basis, center=center,
            )
        )

    augmented_space = FESpace(
        [displacement_space, pressure_space]
        + [NumberSpace(mesh) for _ in range(9)]
        + [NumberSpace(mesh) for _ in range(3)]
    )
    trial = augmented_space.TrialFunction()
    delta_fluctuation, delta_pressure = trial[0], trial[1]
    macro_coordinates = list(trial[2:11])
    qx, qy, qz = trial[11], trial[12], trial[13]

    skew_increment = CF((
        0.0, -qz, qy,
        qz, 0.0, -qx,
        -qy, qx, 0.0,
    ), dims=(3, 3))
    rotation_increment = skew_increment + 0.5 * skew_increment * skew_increment

    deformation_gradient = (
        macro_gradient + Grad(fluctuation) + Grad(delta_fluctuation)
    )
    for coordinate, basis, lift in zip(
        macro_coordinates, macro_bases, macro_lifts
    ):
        deformation_gradient += coordinate * (basis + Grad(lift))
    for index, lift in enumerate(rotation_lifts):
        row, column = divmod(index, 3)
        deformation_gradient += (
            rotation_increment[row, column] * Grad(lift)
        )

    total_pressure = pressure + delta_pressure
    jacobian = Det(deformation_gradient)
    first_invariant = InnerProduct(deformation_gradient, deformation_gradient)
    density = (
        0.5 * mu * (jacobian ** (-2.0 / 3.0) * first_invariant - 3.0)
        + total_pressure * (jacobian - 1.0)
        - total_pressure ** 2 / (2.0 * bulk_modulus)
    )
    measure = dx(intrules={
        ET.TET: IntegrationRule(ET.TET, order=integration_order),
    })
    energy = BilinearForm(augmented_space)
    energy += Variation(density.Compile() * measure)

    zero_increment = GridFunction(augmented_space)
    gradient = zero_increment.vec.CreateVector()
    energy.Apply(zero_increment.vec, gradient)
    energy.AssembleLinearization(zero_increment.vec)

    macro_offset = displacement_space.ndof + pressure_space.ndof
    macro_dofs = np.arange(macro_offset, macro_offset + 9)
    rotation_dofs = np.arange(macro_offset + 9, macro_offset + 12)
    internal_free = augmented_space.FreeDofs()
    for dof in macro_dofs:
        internal_free[int(dof)] = False
    internal_projector = Projector(internal_free, True)
    equilibrium_residual = Norm(internal_projector * gradient)
    if equilibrium_residual > equilibrium_tol:
        raise ValueError(
            "RVE is not equilibrated in fluctuation/pressure/rotation DOFs: "
            f"free residual {equilibrium_residual:.3e}"
        )

    stress = (
        gradient.FV().NumPy()[macro_dofs].copy().reshape(3, 3)
        / cell_volume
    )
    inverse = energy.mat.Inverse(internal_free, inverse="umfpack")
    tangent = np.zeros((9, 9))
    residuals = np.zeros(9)
    sensitivities = []
    for column, macro_dof in enumerate(macro_dofs):
        macro_increment = zero_increment.vec.CreateVector()
        macro_increment[:] = 0.0
        macro_increment.FV().NumPy()[macro_dof] = 1.0

        forcing = zero_increment.vec.CreateVector()
        forcing.data = energy.mat * macro_increment
        relaxation = GridFunction(augmented_space)
        relaxation.vec.data = -(inverse * forcing)

        total_increment = zero_increment.vec.CreateVector()
        total_increment.data = macro_increment + relaxation.vec
        response = zero_increment.vec.CreateVector()
        response.data = energy.mat * total_increment
        tangent[:, column] = (
            response.FV().NumPy()[macro_dofs] / cell_volume
        )
        residuals[column] = Norm(internal_projector * response) / max(
            1.0, Norm(internal_projector * forcing)
        )

        # Total displacement sensitivity includes the macro boundary lift and
        # the relaxed infinitesimal rotation's particle-boundary lift.
        sensitivity = GridFunction(solution.space)
        sensitivity = GridFunction(solution.space)
        # The three generalized q coordinates multiply the nine matrix
        # entries through the skew operator. Only the first-order skew term
        # contributes to the total displacement sensitivity at q = 0.
        q_values = relaxation.vec.FV().NumPy()[rotation_dofs].copy()
        # du_rot = dq x (R (X-Xc)); combine the corresponding row/column lifts.
        rotation_sensitivity = sensitivity.components[0].vec.CreateVector()
        rotation_sensitivity[:] = 0.0
        for q_index, q_value in enumerate(q_values):
            if q_value == 0.0:
                continue
            axis_skew = np.zeros((3, 3))
            if q_index == 0:
                axis_skew[1, 2], axis_skew[2, 1] = -1.0, 1.0
            elif q_index == 1:
                axis_skew[0, 2], axis_skew[2, 0] = 1.0, -1.0
            else:
                axis_skew[0, 1], axis_skew[1, 0] = -1.0, 1.0
            for row in range(3):
                for col in range(3):
                    coefficient = axis_skew[row, col]
                    if coefficient:
                        rotation_sensitivity.data += (
                            q_value * coefficient * rotation_lifts[3 * row + col].vec
                        )
        sensitivity.components[0].vec.data = (
            relaxation.components[0].vec
            + macro_lifts[column].vec
            + rotation_sensitivity
        )
        sensitivity.components[1].vec.data = relaxation.components[1].vec
        sensitivities.append(sensitivity)

    return MacroscopicTangentResult(
        stress=stress,
        tangent=tangent,
        sensitivities=tuple(sensitivities),
        linear_residuals=residuals,
        equilibrium_residual=equilibrium_residual,
    )
