"""Periodic-cell infrastructure for Phase 2 RVE homogenization.

The macroscopic deformation is imposed separately.  The finite-element space
created here represents only the periodic displacement fluctuation, so opposite
faces must share degrees of freedom in all three coordinate directions.
"""
import numpy as np
from netgen.occ import (
    Box,
    IdentificationType,
    OCCGeometry,
    Pnt,
    Sphere,
    X,
    Y,
    Z,
)
from ngsolve import Mesh, Periodic, VectorH1


PERIODIC_IDENTIFICATIONS = ("periodic_x", "periodic_y", "periodic_z")
PARTICLE_BOUNDARY = "particle"


def _identify_periodic_faces(cell):
    directions = (
        (X, "xmin", "xmax", PERIODIC_IDENTIFICATIONS[0]),
        (Y, "ymin", "ymax", PERIODIC_IDENTIFICATIONS[1]),
        (Z, "zmin", "zmax", PERIODIC_IDENTIFICATIONS[2]),
    )
    for direction, master_name, slave_name, identification_name in directions:
        master = cell.faces.Min(direction)
        slave = cell.faces.Max(direction)
        master.name = master_name
        slave.name = slave_name
        master.Identify(
            slave,
            identification_name,
            IdentificationType.PERIODIC,
        )


def make_periodic_unit_cell_mesh(maxh=0.25):
    """Generate a unit-cube mesh with three translational identifications."""
    if maxh <= 0:
        raise ValueError("maxh must be positive")

    cell = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    _identify_periodic_faces(cell)
    return Mesh(OCCGeometry(cell).GenerateMesh(maxh=maxh))


def make_periodic_particle_cell_mesh(
    radius=0.2,
    center=(0.5, 0.5, 0.5),
    maxh=0.2,
):
    """Generate a periodic matrix cell with a spherical rigid-particle cavity.

    Only the deformable matrix is meshed.  The particle motion is represented
    by six generalized rigid-body coordinates and coupled to the named cavity
    boundary by :func:`rigid_increment_constraint_matrix`.
    """
    center = np.asarray(center, dtype=float)
    if center.shape != (3,):
        raise ValueError("center must contain three coordinates")
    if maxh <= 0:
        raise ValueError("maxh must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if np.any(center - radius <= 0) or np.any(center + radius >= 1):
        raise ValueError("particle must lie strictly inside the unit cell")

    box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    particle = Sphere(Pnt(*center), radius)
    particle.faces.name = PARTICLE_BOUNDARY
    cell = box - particle
    cell.mat("matrix")
    _identify_periodic_faces(cell)
    return Mesh(OCCGeometry(cell).GenerateMesh(maxh=maxh))


def make_periodic_displacement_space(mesh, order=2):
    """Return the periodic vector space for the RVE fluctuation field."""
    if order < 1:
        raise ValueError("order must be at least one")
    return Periodic(VectorH1(mesh, order=order))


def _skew(vector):
    """Return the matrix K such that K @ value == vector x value."""
    x_value, y_value, z_value = vector
    return np.array((
        (0.0, -z_value, y_value),
        (z_value, 0.0, -x_value),
        (-y_value, x_value, 0.0),
    ))


def rotation_matrix(rotation_vector):
    """Convert an axis-angle rotation vector to a proper orthogonal matrix."""
    rotation_vector = np.asarray(rotation_vector, dtype=float)
    if rotation_vector.shape != (3,):
        raise ValueError("rotation_vector must contain three values")

    angle = np.linalg.norm(rotation_vector)
    generator = _skew(rotation_vector)
    if angle < 1e-8:
        first_coefficient = 1.0 - angle ** 2 / 6.0
        second_coefficient = 0.5 - angle ** 2 / 24.0
    else:
        first_coefficient = np.sin(angle) / angle
        second_coefficient = (1.0 - np.cos(angle)) / angle ** 2
    return (
        np.eye(3)
        + first_coefficient * generator
        + second_coefficient * generator @ generator
    )


def rigid_particle_displacement(
    points,
    translation,
    rotation_vector,
    center=(0.5, 0.5, 0.5),
):
    """Evaluate the exact finite rigid-particle displacement at material points."""
    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)
    translation = np.asarray(translation, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if center.shape != (3,) or translation.shape != (3,):
        raise ValueError("center and translation must contain three values")

    arms = points - center
    rotation = rotation_matrix(rotation_vector)
    return translation + arms @ (rotation - np.eye(3)).T


def rigid_increment_map(
    points,
    rotation_vector=(0.0, 0.0, 0.0),
    center=(0.5, 0.5, 0.5),
):
    """Map six rigid increments to particle-boundary displacement increments.

    Generalized increments are ordered as
    ``(dt_x, dt_y, dt_z, dtheta_x, dtheta_y, dtheta_z)``.  Rotation increments
    are left-multiplicative, giving ``du = dt + dtheta x (R @ (X - center))``.
    """
    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if center.shape != (3,):
        raise ValueError("center must contain three values")

    rotated_arms = (points - center) @ rotation_matrix(rotation_vector).T
    result = np.zeros((3 * len(points), 6))
    for index, arm in enumerate(rotated_arms):
        rows = slice(3 * index, 3 * index + 3)
        result[rows, :3] = np.eye(3)
        result[rows, 3:] = -_skew(arm)
    return result


def rigid_increment_constraint_matrix(
    points,
    rotation_vector=(0.0, 0.0, 0.0),
    center=(0.5, 0.5, 0.5),
):
    """Return C for the exact algebraic constraint C @ [du, dq] = 0."""
    rigid_map = rigid_increment_map(points, rotation_vector, center)
    return np.hstack((np.eye(rigid_map.shape[0]), -rigid_map))
