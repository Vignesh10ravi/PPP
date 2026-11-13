"""P2-V5 -- Verify homogenized stress and Hill--Mandel macrohomogeneity.

Run:
    python -m tests.verify_hill_mandel
"""
import argparse
import numpy as np
from ngsolve import (
    CF,
    Grad,
    GridFunction,
    Id,
    Projector,
    cos,
    pi,
    sin,
    x,
    y,
    z,
)

from src.homogenization import (
    boundary_first_piola,
    homogenized_first_piola,
    hill_mandel_error,
)
from src.rve import make_periodic_particle_cell_mesh
from src.rve_solver import solve_periodic_rigid_particle


def main(
    radius=0.2,
    stretch=1.005,
    mu=1.0,
    bulk_modulus=1.0e4,
    maxh=0.22,
    diagnostic=False,
):
    center = np.array((0.5, 0.5, 0.5))
    load_parameter = stretch - 1.0
    mesh = make_periodic_particle_cell_mesh(
        radius=radius,
        center=center,
        maxh=maxh,
    )
    macro_gradient = CF(
        (
            stretch, 0.0, 0.0,
            0.0, stretch, 0.0,
            0.0, 0.0, stretch,
        ),
        dims=(3, 3),
    )
    particle_translation = load_parameter * center
    solution, history, _, stress = solve_periodic_rigid_particle(
        mesh,
        macro_gradient,
        particle_displacement=CF(tuple(particle_translation)),
        mu=mu,
        bulk_modulus=bulk_modulus,
        displacement_order=2,
        verbose=True,
    )
    fluctuation, _ = solution.components

    macroscopic_stress = homogenized_first_piola(
        mesh,
        stress,
        fluctuation.space,
        center=center,
    )

    # A constant tensor must be recovered over the TOTAL cell, even though
    # only the matrix is meshed.  Unequal off-diagonals catch transposition;
    # this also checks the particle-reaction sign and volume normalization.
    constant_stress = np.array((
        (1.2, 0.4, -0.3),
        (0.1, 2.3, 0.5),
        (-0.7, 0.6, 1.9),
    ))
    recovered_constant_stress = homogenized_first_piola(
        mesh,
        CF(tuple(constant_stress.ravel()), dims=(3, 3)),
        fluctuation.space,
        center=center,
    )
    constant_stress_error = np.max(
        np.abs(recovered_constant_stress - constant_stress)
    )

    # This virtual field is admissible: it is periodic and its total trace on
    # the particle is the rigid translation X_c.  It need not be a finite-
    # difference sensitivity; Hill--Mandel holds for every admissible virtual
    # velocity field at equilibrium.
    macroscopic_gradient_rate = np.eye(3)
    deformation_gradient_rate = (
        Id(3) + Grad(fluctuation) / load_parameter
    )
    microscopic_power, macroscopic_power, relative_error = hill_mandel_error(
        mesh,
        stress,
        macroscopic_stress,
        macroscopic_gradient_rate,
        deformation_gradient_rate,
    )

    # Add an independently generated homogeneous virtual fluctuation.  Its
    # particle trace is zero, so macro power must not depend on this choice.
    virtual_fluctuation = GridFunction(fluctuation.space)
    virtual_fluctuation.Set(CF((
        sin(2 * pi * x) * cos(2 * pi * y),
        sin(2 * pi * y) * cos(2 * pi * z),
        sin(2 * pi * z) * cos(2 * pi * x),
    )))
    virtual_fluctuation.vec.data = (
        Projector(fluctuation.space.FreeDofs(), True)
        * virtual_fluctuation.vec
    )
    alternative_power, _, alternative_error = hill_mandel_error(
        mesh,
        stress,
        macroscopic_stress,
        macroscopic_gradient_rate,
        deformation_gradient_rate + Grad(virtual_fluctuation),
    )
    virtual_field_error = abs(alternative_power - microscopic_power) / max(
        1.0, abs(microscopic_power)
    )

    boundary_stress = boundary_first_piola(mesh, stress)
    boundary_power = float(np.sum(boundary_stress * macroscopic_gradient_rate))
    boundary_error = abs(microscopic_power - boundary_power) / max(
        1.0, abs(microscopic_power), abs(boundary_power)
    )

    stress_asymmetry = np.linalg.norm(
        macroscopic_stress - macroscopic_stress.T
    ) / max(1.0, np.linalg.norm(macroscopic_stress))
    normal_stresses = np.diag(macroscopic_stress)
    cubic_normal_variation = np.max(
        np.abs(normal_stresses - normal_stresses.mean())
    ) / max(1.0, np.max(np.abs(normal_stresses)))
    shear_scale = np.max(
        np.abs(macroscopic_stress - np.diag(normal_stresses))
    ) / max(1.0, np.max(np.abs(normal_stresses)))

    print(f"\nmesh elements:                    {mesh.ne}")
    print("homogenization: FE volume + particle reaction work")
    print("homogenized first-Piola stress:")
    print(macroscopic_stress)
    print(f"microscopic virtual power:        {microscopic_power:.12e}")
    print(f"macroscopic virtual power:        {macroscopic_power:.12e}")
    print(f"Hill-Mandel relative error:       {relative_error:.3e}")
    print(f"alternative virtual-field error:  {alternative_error:.3e}")
    print(f"virtual-field independence error: {virtual_field_error:.3e}")
    print(f"constant-stress recovery error:   {constant_stress_error:.3e}")
    print(f"raw-boundary error (diagnostic):  {boundary_error:.3e}")
    print(f"macrostress asymmetry:            {stress_asymmetry:.3e}")
    print(f"cubic normal-stress variation:    {cubic_normal_variation:.3e}")
    print(f"relative shear stress:            {shear_scale:.3e}")
    print(f"final RVE residual:               {history[-1]:.3e}")

    if diagnostic:
        print("DIAGNOSTIC ONLY: acceptance assertions were not evaluated.")
        return

    assert relative_error < 1.0e-6, "Hill-Mandel work equality failed"
    assert alternative_error < 1.0e-6, "alternative virtual-work equality failed"
    assert virtual_field_error < 1.0e-6, "power depends on free virtual fluctuations"
    assert constant_stress_error < 1.0e-10, (
        "constant-stress recovery has incorrect reaction sign or normalization"
    )
    assert stress_asymmetry < 1.0e-6, "hydrostatic macrostress is asymmetric"
    assert cubic_normal_variation < 1.0e-5, (
        "hydrostatic normal stresses violate cubic symmetry"
    )
    assert shear_scale < 1.0e-5, "hydrostatic loading produced shear stress"
    assert history[-1] < 1.0e-9, "RVE equilibrium is not converged"
    print("P2-V5 PASS: homogenized stress satisfies Hill-Mandel equality.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxh", type=float, default=0.22)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Report refinement diagnostics without claiming verification passed.",
    )
    main(**vars(parser.parse_args()))
