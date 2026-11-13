"""P2-V2 -- Verify rigid-particle geometry and finite rigid kinematics.

Run:
    python -m tests.verify_rigid_particle
"""
import numpy as np
from ngsolve import Integrate

from src.rve import (
    PARTICLE_BOUNDARY,
    make_periodic_displacement_space,
    make_periodic_particle_cell_mesh,
    rigid_increment_constraint_matrix,
    rigid_increment_map,
    rigid_particle_displacement,
    rotation_matrix,
)


def _particle_points(center, radius):
    directions = np.array((
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, -1.0),
    ))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return center + radius * directions


def _maximum_distance_error(reference, deformed):
    reference_distances = np.linalg.norm(
        reference[:, None, :] - reference[None, :, :],
        axis=2,
    )
    deformed_distances = np.linalg.norm(
        deformed[:, None, :] - deformed[None, :, :],
        axis=2,
    )
    return np.max(np.abs(deformed_distances - reference_distances))


def main(radius=0.2, maxh=0.18):
    center = np.array((0.5, 0.5, 0.5))
    mesh = make_periodic_particle_cell_mesh(
        radius=radius,
        center=center,
        maxh=maxh,
    )
    boundaries = set(mesh.GetBoundaries())
    assert PARTICLE_BOUNDARY in boundaries, "particle interface is not named"

    periodic = make_periodic_displacement_space(mesh, order=1)
    active_dofs = sum(bool(is_free) for is_free in periodic.FreeDofs())
    assert active_dofs > 0, "periodic matrix space has no active dofs"

    matrix_volume = Integrate(1, mesh)
    exact_matrix_volume = 1.0 - 4.0 * np.pi * radius ** 3 / 3.0
    volume_error = abs(matrix_volume - exact_matrix_volume) / exact_matrix_volume

    points = _particle_points(center, radius)
    translation = np.array((0.07, -0.04, 0.02))
    rotation_vector = np.array((0.31, -0.18, 0.24))
    rotation = rotation_matrix(rotation_vector)
    displacement = rigid_particle_displacement(
        points,
        translation,
        rotation_vector,
        center,
    )
    deformed = points + displacement

    orthogonality_error = np.linalg.norm(rotation.T @ rotation - np.eye(3))
    determinant_error = abs(np.linalg.det(rotation) - 1.0)
    distance_error = _maximum_distance_error(points, deformed)

    rigid_map = rigid_increment_map(points, rotation_vector, center)
    constraint = rigid_increment_constraint_matrix(
        points,
        rotation_vector,
        center,
    )
    map_rank = np.linalg.matrix_rank(rigid_map)
    constraint_rank = np.linalg.matrix_rank(constraint)
    increment = np.array((0.03, -0.02, 0.01, 0.12, -0.08, 0.05))
    nodal_increment = rigid_map @ increment
    constraint_residual = np.linalg.norm(
        constraint @ np.concatenate((nodal_increment, increment))
    )

    # Check the rotational columns against their defining left-multiplicative
    # finite-rotation update.
    epsilon = 1e-7
    updated_rotation = rotation_matrix(epsilon * increment[3:]) @ rotation
    rotated_arms = (points - center) @ rotation.T
    updated_arms = (points - center) @ updated_rotation.T
    finite_difference_increment = (
        increment[:3] + (updated_arms - rotated_arms) / epsilon
    ).reshape(-1)
    linearization_error = np.linalg.norm(
        finite_difference_increment - nodal_increment
    ) / max(1.0, np.linalg.norm(nodal_increment))

    print(f"\nmesh elements:                  {mesh.ne}")
    print(f"periodic active dofs:           {active_dofs}")
    print(f"relative matrix-volume error:   {volume_error:.3e}")
    print(f"rotation orthogonality error:   {orthogonality_error:.3e}")
    print(f"rotation determinant error:     {determinant_error:.3e}")
    print(f"maximum distance error:         {distance_error:.3e}")
    print(f"rigid-map rank:                 {map_rank}")
    print(f"constraint rank:                {constraint_rank}")
    print(f"constraint residual:            {constraint_residual:.3e}")
    print(f"increment linearization error:  {linearization_error:.3e}")

    assert volume_error < 5e-2, "particle-cell volume is inconsistent"
    assert orthogonality_error < 1e-12, "rotation is not orthogonal"
    assert determinant_error < 1e-12, "rotation is not proper"
    assert distance_error < 1e-12, "finite particle motion is not rigid"
    assert map_rank == 6, "particle does not have six independent rigid modes"
    assert constraint_rank == 3 * len(points), "constraint equations are dependent"
    assert constraint_residual < 1e-12, "rigid increment violates its constraint"
    assert linearization_error < 1e-7, "rigid increment map is inconsistent"
    print("P2-V2 PASS: particle geometry and exact rigid kinematics are valid.")


if __name__ == "__main__":
    main()
