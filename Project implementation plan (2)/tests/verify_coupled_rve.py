"""P2-V4 -- Coupled periodic matrix and exact rigid-particle verification.

Run:
    python -m tests.verify_coupled_rve
"""
import numpy as np
from ngsolve import (
    BND,
    CF,
    Integrate,
    InnerProduct,
    Mesh,
    Norm,
    sqrt,
    specialcf,
    x,
    y,
    z,
)

from src.rve import PARTICLE_BOUNDARY, make_periodic_particle_cell_mesh
from src.rve_solver import solve_periodic_rigid_particle


def _vector_value(coefficient, mesh, point):
    return np.asarray(coefficient(mesh(*point)), dtype=float)


def _surface_resultants(mesh, stress, center):
    normal = specialcf.normal(3)
    traction = stress * normal
    arm = CF((x - center[0], y - center[1], z - center[2]))
    moment = CF((
        arm[1] * traction[2] - arm[2] * traction[1],
        arm[2] * traction[0] - arm[0] * traction[2],
        arm[0] * traction[1] - arm[1] * traction[0],
    ))
    particle_boundary = mesh.Boundaries(PARTICLE_BOUNDARY)
    force = np.array([
        Integrate(
            traction[component],
            mesh,
            BND,
            definedon=particle_boundary,
        )
        for component in range(3)
    ])
    resultant_moment = np.array([
        Integrate(
            moment[component],
            mesh,
            BND,
            definedon=particle_boundary,
        )
        for component in range(3)
    ])
    traction_scale = Integrate(
        sqrt(InnerProduct(traction, traction)),
        mesh,
        BND,
        definedon=particle_boundary,
    )
    return force, resultant_moment, traction_scale


def main(
    radius=0.2,
    stretch=1.005,
    mu=1.0,
    bulk_modulus=1.0e4,
):
    center = np.array((0.5, 0.5, 0.5))
    mesh = make_periodic_particle_cell_mesh(
        radius=radius,
        center=center,
        maxh=0.22,
    )
    macro_gradient = CF(
        (
            stretch, 0.0, 0.0,
            0.0, stretch, 0.0,
            0.0, 0.0, stretch,
        ),
        dims=(3, 3),
    )
    particle_translation = (stretch - 1.0) * center
    solution, history, deformation_gradient, stress = (
        solve_periodic_rigid_particle(
            mesh,
            macro_gradient,
            particle_displacement=CF(tuple(particle_translation)),
            mu=mu,
            bulk_modulus=bulk_modulus,
            displacement_order=2,
            verbose=True,
        )
    )
    fluctuation, _ = solution.components

    outer_pairs = (
        ((0.0, 0.31, 0.67), (1.0, 0.31, 0.67)),
        ((0.28, 0.0, 0.61), (0.28, 1.0, 0.61)),
        ((0.73, 0.36, 0.0), (0.73, 0.36, 1.0)),
    )
    periodic_jump = max(
        np.max(
            np.abs(
                _vector_value(fluctuation, mesh, slave)
                - _vector_value(fluctuation, mesh, master)
            )
        )
        for master, slave in outer_pairs
    )

    reference_position = CF((x, y, z))
    rigid_trace_error = (
        (stretch - 1.0) * reference_position
        + fluctuation
        - CF(tuple(particle_translation))
    )
    particle_boundary = mesh.Boundaries(PARTICLE_BOUNDARY)
    particle_area = Integrate(1, mesh, BND, definedon=particle_boundary)
    rigid_error = sqrt(
        Integrate(
            InnerProduct(rigid_trace_error, rigid_trace_error),
            mesh,
            BND,
            definedon=particle_boundary,
        )
        / particle_area
    )

    force, moment, traction_scale = _surface_resultants(
        mesh,
        stress,
        center,
    )
    force_error = np.linalg.norm(force) / max(1.0, traction_scale)
    moment_error = np.linalg.norm(moment) / max(1.0, radius * traction_scale)
    center_gradient = np.asarray(
        deformation_gradient(mesh(0.5 + radius + 0.02, 0.5, 0.5))
    ).reshape(3, 3)
    minimum_sampled_jacobian = np.linalg.det(center_gradient)
    final_residual = history[-1]

    print(f"\nmesh elements:                    {mesh.ne}")
    print(f"bulk/shear ratio:                 {bulk_modulus / mu:.3e}")
    print(f"maximum periodic jump:            {periodic_jump:.3e}")
    print(f"particle rigid-trace L2 error:     {rigid_error:.3e}")
    print(f"relative particle force:          {force_error:.3e}")
    print(f"relative particle moment:         {moment_error:.3e}")
    print(f"near-particle sampled J:          {minimum_sampled_jacobian:.6e}")
    print(f"final RVE residual:               {final_residual:.3e}")

    assert periodic_jump < 1.0e-9, "RVE fluctuation is not periodic"
    assert rigid_error < 1.0e-9, "particle surface violates rigid motion"
    assert force_error < 1.0e-6, "particle has nonzero resultant force"
    assert moment_error < 1.0e-8, "particle has nonzero resultant moment"
    assert minimum_sampled_jacobian > 0.0, "matrix deformation is inadmissible"
    assert final_residual < 1.0e-9, "coupled RVE Newton solve did not converge"
    print("P2-V4 PASS: periodic matrix and rigid particle are in equilibrium.")


if __name__ == "__main__":
    main()
