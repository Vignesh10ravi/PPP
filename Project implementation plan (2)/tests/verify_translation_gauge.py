"""P2-V8 -- Verify the periodic RVE translation gauge.

Uniform particle translation is not an independent energetic coordinate:
adding a constant to the particle displacement adds the same constant to the
periodic fluctuation, while F, pressure, stress, and energy remain unchanged.
The gauge is fixed by making the particle center follow the macroscopic map.

Run:
    python -m tests.verify_translation_gauge
"""
import argparse

import numpy as np
from ngsolve import CF, Grad, Id

from src.homogenization import homogenized_first_piola
from src.rve import make_periodic_particle_cell_mesh
from src.rve_solver import solve_periodic_rigid_particle


def _matrix_cf(matrix):
    return CF(tuple(np.asarray(matrix).ravel()), dims=(3, 3))


def _value(coefficient, mesh, point):
    return np.asarray(coefficient(mesh(*point)), dtype=float)


def main(maxh=0.22):
    center = np.array((0.5, 0.5, 0.5))
    macro_gradient = np.array((
        (1.02, 0.05, -0.01),
        (0.00, 0.98, 0.03),
        (0.01, 0.00, 1.00),
    ))
    base_translation = (macro_gradient - np.eye(3)) @ center
    gauge_shift = np.array((2.0e-3, -1.5e-3, 1.0e-3))
    mesh = make_periodic_particle_cell_mesh(maxh=maxh)
    macro_cf = _matrix_cf(macro_gradient)

    base, base_history, _, base_stress = solve_periodic_rigid_particle(
        mesh,
        macro_cf,
        particle_displacement=CF(tuple(base_translation)),
        verbose=False,
    )
    shifted, shifted_history, _, shifted_stress = solve_periodic_rigid_particle(
        mesh,
        macro_cf,
        particle_displacement=CF(tuple(base_translation + gauge_shift)),
        verbose=False,
    )
    base_fluctuation, base_pressure = base.components
    shifted_fluctuation, shifted_pressure = shifted.components

    sample_points = (
        (0.18, 0.27, 0.34),
        (0.76, 0.31, 0.63),
        (0.42, 0.82, 0.21),
        (0.67, 0.73, 0.79),
    )
    displacement_shift_error = max(
        np.linalg.norm(
            _value(shifted_fluctuation - base_fluctuation, mesh, point)
            - gauge_shift
        )
        for point in sample_points
    )
    base_gradient = Id(3) + Grad(base_fluctuation)
    shifted_gradient = Id(3) + Grad(shifted_fluctuation)
    gradient_error = max(
        np.linalg.norm(
            _value(shifted_gradient - base_gradient, mesh, point)
        )
        for point in sample_points
    )
    pressure_error = max(
        abs(float((shifted_pressure - base_pressure)(mesh(*point))))
        for point in sample_points
    )
    stress_error = max(
        np.linalg.norm(
            _value(shifted_stress - base_stress, mesh, point)
        )
        for point in sample_points
    )

    base_macro_stress = homogenized_first_piola(
        mesh, base_stress, base_fluctuation.space, center=center,
    )
    shifted_macro_stress = homogenized_first_piola(
        mesh, shifted_stress, shifted_fluctuation.space, center=center,
    )
    macro_stress_error = np.linalg.norm(
        shifted_macro_stress - base_macro_stress
    ) / max(1.0, np.linalg.norm(base_macro_stress))

    print(f"\nmesh elements:                     {mesh.ne}")
    print(f"imposed translation-gauge shift:   {np.linalg.norm(gauge_shift):.3e}")
    print(f"fluctuation shift error:            {displacement_shift_error:.3e}")
    print(f"deformation-gradient change:        {gradient_error:.3e}")
    print(f"pressure change:                    {pressure_error:.3e}")
    print(f"pointwise stress change:             {stress_error:.3e}")
    print(f"homogenized stress relative change: {macro_stress_error:.3e}")
    print(f"base mixed residual:                {base_history[-1]:.3e}")
    print(f"shifted mixed residual:             {shifted_history[-1]:.3e}")

    assert displacement_shift_error < 1.0e-10, (
        "particle translation did not produce a uniform fluctuation shift"
    )
    assert gradient_error < 1.0e-10, "translation gauge changed deformation"
    assert pressure_error < 1.0e-9, "translation gauge changed pressure"
    assert stress_error < 1.0e-8, "translation gauge changed pointwise stress"
    assert macro_stress_error < 1.0e-10, (
        "translation gauge changed homogenized stress"
    )
    assert max(base_history[-1], shifted_history[-1]) < 1.0e-9, (
        "gauge comparison RVE did not converge"
    )
    print("P2-V8 PASS: particle translation is correctly treated as an RVE gauge.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxh", type=float, default=0.22)
    main(**vars(parser.parse_args()))
