"""P2-V3 -- Verify the near-incompressible mixed matrix formulation.

Run:
    python -m tests.verify_mixed_matrix
"""
import numpy as np
from netgen.occ import Box, OCCGeometry, Pnt
from ngsolve import CF, Grad, Id, Mesh, x, y, z

from src.mixed_matrix import solve_mixed_matrix


def _sample_deformation_gradients(mesh, displacement, points):
    deformation_gradient = Id(3) + Grad(displacement)
    return np.asarray([
        deformation_gradient(mesh(*point)) for point in points
    ]).reshape(-1, 3, 3)


def main(mu=1.0, bulk_modulus=1.0e4):
    geometry = OCCGeometry(Box(Pnt(0, 0, 0), Pnt(1, 1, 1)))
    mesh = Mesh(geometry.GenerateMesh(maxh=0.3))

    # Mixed distortional loading with a small controlled volume change.
    target_jacobian = 1.001
    stretch_x = 1.08
    stretch_y = 0.94
    stretch_z = target_jacobian / (stretch_x * stretch_y)
    target_gradient = np.array((
        (stretch_x, 0.06, 0.00),
        (0.00, stretch_y, 0.04),
        (0.00, 0.00, stretch_z),
    ))
    boundary_displacement = CF((
        (stretch_x - 1.0) * x + 0.06 * y,
        (stretch_y - 1.0) * y + 0.04 * z,
        (stretch_z - 1.0) * z,
    ))

    solution, history = solve_mixed_matrix(
        mesh,
        boundary_displacement,
        mu=mu,
        bulk_modulus=bulk_modulus,
        displacement_order=2,
        verbose=True,
    )
    displacement, pressure = solution.components
    sample_points = (
        (0.21, 0.34, 0.43),
        (0.63, 0.27, 0.72),
        (0.38, 0.76, 0.19),
    )
    sampled_gradients = _sample_deformation_gradients(
        mesh,
        displacement,
        sample_points,
    )

    gradient_error = np.max(np.abs(sampled_gradients - target_gradient))
    sampled_jacobians = np.linalg.det(sampled_gradients)
    jacobian_error = np.max(np.abs(sampled_jacobians - target_jacobian))

    sampled_pressures = np.array([
        float(pressure(mesh(*point))) for point in sample_points
    ])
    expected_pressure = bulk_modulus * (target_jacobian - 1.0)
    pressure_error = np.max(np.abs(sampled_pressures - expected_pressure))
    pressure_variation = np.max(
        np.abs(sampled_pressures - sampled_pressures.mean())
    )
    final_residual = history[-1]

    print(f"\nmesh elements:                    {mesh.ne}")
    print(f"bulk/shear ratio:                 {bulk_modulus / mu:.3e}")
    print(f"target J:                         {target_jacobian:.6f}")
    print(f"expected pressure:                {expected_pressure:.6e}")
    print(f"max deformation-gradient error:   {gradient_error:.3e}")
    print(f"max Jacobian error:               {jacobian_error:.3e}")
    print(f"max pressure error:               {pressure_error:.3e}")
    print(f"pressure variation:               {pressure_variation:.3e}")
    print(f"final mixed residual:             {final_residual:.3e}")

    assert bulk_modulus / mu >= 1.0e4, "test is not near-incompressible"
    assert gradient_error < 1.0e-10, "mixed patch deformation is not affine"
    assert jacobian_error < 1.0e-10, "mixed patch has the wrong volume change"
    assert pressure_error < 1.0e-8, "pressure does not satisfy its field equation"
    assert pressure_variation < 1.0e-9, "pressure is not uniform"
    assert final_residual < 1.0e-10, "mixed Newton solve did not converge"
    print("P2-V3 PASS: the near-incompressible Taylor-Hood patch is exact.")


if __name__ == "__main__":
    main()
