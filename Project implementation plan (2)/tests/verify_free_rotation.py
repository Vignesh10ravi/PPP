"""P2-V7 -- Verify exact free rigid-particle rotation under simple shear.

Run:
    python -m tests.verify_free_rotation
"""
import argparse

import numpy as np
from ngsolve import BND, CF, Integrate, InnerProduct, sqrt, x, y, z

from src.free_particle import solve_free_rotating_particle
from src.rve import PARTICLE_BOUNDARY, make_periodic_particle_cell_mesh


def main(maxh=0.22, shear=0.05):
    center = np.array((0.5, 0.5, 0.5))
    radius = 0.2
    macro_gradient = np.array((
        (1.0, shear, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ))
    mesh = make_periodic_particle_cell_mesh(
        radius=radius,
        center=center,
        maxh=maxh,
    )
    result = solve_free_rotating_particle(
        mesh,
        macro_gradient,
        center=center,
        verbose=True,
    )
    fluctuation, _ = result.solution.components

    reference_position = CF((x, y, z))
    reference_arm = reference_position - CF(tuple(center))
    expected_particle_displacement = (
        CF(tuple((macro_gradient - np.eye(3)) @ center))
        + (CF(tuple(result.rotation.ravel()), dims=(3, 3)) - CF(
            tuple(np.eye(3).ravel()), dims=(3, 3)
        )) * reference_arm
    )
    total_displacement = (
        (CF(tuple(macro_gradient.ravel()), dims=(3, 3))
         - CF(tuple(np.eye(3).ravel()), dims=(3, 3))) * reference_position
        + fluctuation
    )
    particle_boundary = mesh.Boundaries(PARTICLE_BOUNDARY)
    area = Integrate(1, mesh, BND, definedon=particle_boundary)
    rigid_trace_error = sqrt(
        Integrate(
            InnerProduct(
                total_displacement - expected_particle_displacement,
                total_displacement - expected_particle_displacement,
            ),
            mesh,
            BND,
            definedon=particle_boundary,
        ) / area
    )

    orthogonality_error = np.linalg.norm(
        result.rotation.T @ result.rotation - np.eye(3)
    )
    determinant_error = abs(np.linalg.det(result.rotation) - 1.0)
    hessian_symmetry = np.linalg.norm(
        result.condensed_hessian - result.condensed_hessian.T
    ) / max(1.0, np.linalg.norm(result.condensed_hessian))
    initial_residual = result.rotational_residuals[0]
    final_residual = result.rotational_residuals[-1]
    reduction = final_residual / max(initial_residual, 1.0e-30)
    rotation_angle = np.arctan2(result.rotation[1, 0], result.rotation[0, 0])

    print(f"\nmesh elements:                     {mesh.ne}")
    print(f"rotation iterations:               {len(result.rotational_residuals)}")
    print(f"initial rotational residual:       {initial_residual:.3e}")
    print(f"final rotational residual:         {final_residual:.3e}")
    print(f"rotational residual ratio:         {reduction:.3e}")
    print(f"particle z-rotation angle:          {rotation_angle:.6e}")
    print(f"rotation orthogonality error:       {orthogonality_error:.3e}")
    print(f"rotation determinant error:         {determinant_error:.3e}")
    print(f"particle rigid-trace L2 error:      {rigid_trace_error:.3e}")
    print(f"rotational Hessian symmetry error:  {hessian_symmetry:.3e}")
    print(
        f"max condensed linear residual:      "
        f"{max(result.condensed_linear_residuals):.3e}"
    )
    print(f"final mixed residual:               {result.newton_history[-1]:.3e}")

    assert len(result.rotational_residuals) >= 2, "rotation was not exercised"
    assert final_residual < 1.0e-9, "particle moment equilibrium failed"
    assert reduction < 1.0e-5, "rotational Newton did not reduce the moment"
    assert abs(rotation_angle) > 1.0e-3, "shear produced no particle rotation"
    assert orthogonality_error < 1.0e-12, "rotation matrix is not orthogonal"
    assert determinant_error < 1.0e-12, "rotation matrix is not proper"
    assert rigid_trace_error < 1.0e-9, "particle trace is not exactly rigid"
    assert hessian_symmetry < 1.0e-10, "condensed rotational Hessian is asymmetric"
    assert max(result.condensed_linear_residuals) < 1.0e-9, (
        "condensed rotational sensitivity is not equilibrated"
    )
    assert result.newton_history[-1] < 1.0e-9, "mixed RVE is not converged"
    print("P2-V7 PASS: free finite particle rotation is moment-equilibrated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxh", type=float, default=0.22)
    parser.add_argument("--shear", type=float, default=0.05)
    main(**vars(parser.parse_args()))
