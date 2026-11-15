"""Macroscopic stress and Hill--Mandel work for the unit periodic RVE."""
import numpy as np
from ngsolve import (
    BND,
    CF,
    ET,
    Grad,
    GridFunction,
    Integrate,
    IntegrationRule,
    InnerProduct,
    LinearForm,
    dx,
    specialcf,
    x,
    y,
    z,
)

from .rve import PARTICLE_BOUNDARY


OUTER_CELL_BOUNDARIES = "xmin|xmax|ymin|ymax|zmin|zmax"


def particle_macro_lift(
    mesh,
    displacement_space,
    macroscopic_gradient_rate,
    center=(0.5, 0.5, 0.5),
):
    """Lift the trace -H (X-center) into the periodic fluctuation space."""
    center = np.asarray(center, dtype=float)
    rate = np.asarray(macroscopic_gradient_rate, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center must contain three finite coordinates")
    if rate.shape != (3, 3) or not np.all(np.isfinite(rate)):
        raise ValueError("macroscopic_gradient_rate must be a finite 3x3 matrix")
    if PARTICLE_BOUNDARY not in mesh.GetBoundaries():
        raise ValueError("mesh must have a named particle boundary")

    rate_cf = CF(tuple(rate.ravel()), dims=(3, 3))
    arm = CF((x - center[0], y - center[1], z - center[2]))
    lift = GridFunction(displacement_space)
    lift.Set(
        -rate_cf * arm,
        definedon=mesh.Boundaries(PARTICLE_BOUNDARY),
    )
    return lift


def boundary_first_piola(
    mesh,
    stress,
    cell_volume=1.0,
    integration_order=10,
):
    """Estimate macroscopic first-Piola stress from exterior boundary tractions.

    Uses ``Pbar = int((P N) tensor X, outer boundary) / cell_volume``.
    Raw constitutive stress traces are not equilibrated FE nodal reactions.
    This estimate is therefore subject to stress-recovery/discretization
    error; exact discrete work conjugacy is not guaranteed.
    """
    if cell_volume <= 0:
        raise ValueError("cell_volume must be positive")

    outer_boundary = mesh.Boundaries(OUTER_CELL_BOUNDARIES)
    reference_position = CF((x, y, z))
    traction = stress * specialcf.normal(3)
    macroscopic_stress = np.zeros((3, 3))
    for row in range(3):
        for column in range(3):
            macroscopic_stress[row, column] = Integrate(
                traction[row] * reference_position[column],
                mesh,
                BND,
                definedon=outer_boundary,
                order=integration_order,
            ) / cell_volume
    return macroscopic_stress


def homogenized_first_piola(
    mesh,
    stress,
    displacement_space,
    center=(0.5, 0.5, 0.5),
    cell_volume=1.0,
    integration_order=10,
):
    """Compute work-conjugate stress for the prescribed rigid-particle RVE.

    The particle orientation is fixed and its center follows the macroscopic
    motion: t = (Fbar - I) center.  For a unit macro-gradient increment E_iJ,
    the fluctuation lift b_iJ has particle trace -E_iJ (X - center).

    Differentiating the discrete potential with these constraints gives
    Pbar_iJ = (int(P_iJ) + r_u . b_iJ) / cell_volume.
    Here r_u is the FULL assembled displacement residual, including particle
    reactions.  Projecting it onto free DOFs would discard the required term.
    The lifts are constructed independently of the solved fluctuation.
    """
    if cell_volume <= 0:
        raise ValueError("cell_volume must be positive")
    measure = dx(intrules={
        ET.TET: IntegrationRule(ET.TET, order=integration_order),
    })
    test_displacement = displacement_space.TestFunction()
    reactions = LinearForm(displacement_space)
    reactions += InnerProduct(stress, Grad(test_displacement)) * measure
    reactions.Assemble()

    macroscopic_stress = np.zeros((3, 3))
    for row in range(3):
        for column in range(3):
            rate = np.zeros((3, 3))
            rate[row, column] = 1.0
            lift = particle_macro_lift(
                mesh, displacement_space, rate, center=center,
            )
            matrix_work = Integrate(
                stress[row, column],
                mesh,
                order=integration_order,
            )
            reaction_work = InnerProduct(reactions.vec, lift.vec)
            macroscopic_stress[row, column] = (
                matrix_work + reaction_work
            ) / cell_volume
    return macroscopic_stress


def microscopic_stress_power(
    mesh,
    stress,
    deformation_gradient_rate,
    integration_order=10,
):
    """Integrate microscopic first-Piola stress power over the matrix."""
    return Integrate(
        InnerProduct(stress, deformation_gradient_rate),
        mesh,
        order=integration_order,
    )


def hill_mandel_error(
    mesh,
    stress,
    macroscopic_stress,
    macroscopic_gradient_rate,
    deformation_gradient_rate,
    cell_volume=1.0,
    integration_order=10,
):
    """Return microscopic power, macroscopic power, and relative mismatch."""
    microscopic_power = microscopic_stress_power(
        mesh,
        stress,
        deformation_gradient_rate,
        integration_order=integration_order,
    ) / cell_volume
    macroscopic_power = float(
        np.sum(
            np.asarray(macroscopic_stress)
            * np.asarray(macroscopic_gradient_rate)
        )
    )
    scale = max(1.0, abs(microscopic_power), abs(macroscopic_power))
    relative_error = abs(microscopic_power - macroscopic_power) / scale
    return microscopic_power, macroscopic_power, relative_error
