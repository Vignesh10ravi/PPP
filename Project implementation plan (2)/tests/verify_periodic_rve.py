"""P2-V1 -- Verify the triply periodic RVE mesh and displacement space.

Run:
    python -m tests.verify_periodic_rve
"""
import numpy as np
from ngsolve import CF, GridFunction, VectorH1, cos, pi, sin, x, y, z

from src.rve import make_periodic_displacement_space, make_periodic_unit_cell_mesh


def _active_dofs(fes):
    return sum(bool(is_free) for is_free in fes.FreeDofs())


def _value(gfu, mesh, point):
    return np.asarray(gfu(mesh(*point)), dtype=float)


def main(maxh=0.25, order=2):
    mesh = make_periodic_unit_cell_mesh(maxh=maxh)
    unconstrained = VectorH1(mesh, order=order)
    periodic = make_periodic_displacement_space(mesh, order=order)

    unconstrained_dofs = _active_dofs(unconstrained)
    periodic_dofs = _active_dofs(periodic)
    assert periodic_dofs < unconstrained_dofs, (
        "periodic identification did not eliminate any active degrees of freedom"
    )

    # A smooth, exactly periodic field exercises every displacement component
    # and each of the three independent face identifications.
    periodic_field = CF((
        sin(2 * pi * x) + 0.2 * cos(2 * pi * y),
        cos(2 * pi * y) + 0.3 * sin(2 * pi * z),
        sin(2 * pi * z) + 0.4 * cos(2 * pi * x),
    ))
    gfu = GridFunction(periodic)
    gfu.Set(periodic_field)

    interior_coordinates = ((0.23, 0.37), (0.41, 0.68), (0.72, 0.19))
    face_pairs = (
        ((0.0, *interior_coordinates[0]), (1.0, *interior_coordinates[0])),
        (
            (interior_coordinates[1][0], 0.0, interior_coordinates[1][1]),
            (interior_coordinates[1][0], 1.0, interior_coordinates[1][1]),
        ),
        ((*interior_coordinates[2], 0.0), (*interior_coordinates[2], 1.0)),
    )
    jumps = [
        np.max(np.abs(_value(gfu, mesh, slave) - _value(gfu, mesh, master)))
        for master, slave in face_pairs
    ]
    max_jump = max(jumps)

    print(f"\nmesh elements:              {mesh.ne}")
    print(f"unconstrained active dofs:  {unconstrained_dofs}")
    print(f"periodic active dofs:       {periodic_dofs}")
    print(f"eliminated active dofs:     {unconstrained_dofs - periodic_dofs}")
    print(f"max opposite-face jump:     {max_jump:.3e}")

    assert max_jump < 1e-10, "periodic traces do not match on opposite faces"
    print("P2-V1 PASS: the unit-cell displacement space is triply periodic.")


if __name__ == "__main__":
    main()
