import numpy as np
from ngsolve import (
    CF, Det, Grad, GridFunction, Id, IntegrationRule, ET,
    VectorH1, H1, Periodic, x, y, z, dx, Integrate
)

from src.rve import (
    PARTICLE_BOUNDARY,
    make_periodic_particle_cell_mesh,
)
from src.rve_dataset import (
    DatasetConfig,
    generate_deformation_states,
    particle_radius,
)
from src.mixed_matrix import make_mixed_space

config = DatasetConfig(
    volume_fractions=(0.02, 0.05, 0.10),
    n_deformations=100,
    seed=1729,
    max_strain=0.20,
    determinant_floor=0.50,
    max_condition_number=3.0,
    split_fractions=(0.70, 0.15, 0.15),
    maxh=0.22,
    mu=1.0,
    bulk_modulus=10000.0,
    displacement_order=2,
    integration_order=10,
)

F = np.asarray(generate_deformation_states(config)[47], dtype=float)
print("Macro det(F):", np.linalg.det(F))
print("Macro F:\n", F)

mesh = make_periodic_particle_cell_mesh(
    radius=particle_radius(0.02),
    maxh=config.maxh,
)

fes = make_mixed_space(
    mesh,
    displacement_order=config.displacement_order,
    periodic=True,
    dirichlet=PARTICLE_BOUNDARY,
)

solution = GridFunction(fes)
reference_position = CF((x, y, z))
macro_gradient = CF(tuple(F.ravel()), dims=(3, 3))
macro_displacement = (macro_gradient - Id(3)) * reference_position

# Reproduce the initial displacement boundary condition used by the solver.
particle_displacement = CF((0.0, 0.0, 0.0))
particle_fluctuation = particle_displacement - macro_displacement
solution.components[0].Set(
    particle_fluctuation,
    definedon=mesh.Boundaries(PARTICLE_BOUNDARY),
)

fluctuation, pressure = solution.components
local_F = macro_gradient + Grad(fluctuation)
J = Det(local_F)

print("Solution vector finite:",
      np.all(np.isfinite(solution.vec.FV().NumPy())))

measure = dx(intrules={
    ET.TET: IntegrationRule(ET.TET, order=config.integration_order),
})

# These are integrated diagnostics, not pointwise minima.
for label, expr in (
    ("J", J),
    ("J squared", J * J),
):
    try:
        value = Integrate(expr * measure, mesh)
        print(f"Integral of {label}: {value}")
    except Exception as exc:
        print(f"Could not integrate {label}: {type(exc).__name__}: {exc}")

# Compare the initial state with the uniform macro-gradient field.
uniform_J = np.linalg.det(F)
print("Uniform macro J:", uniform_J)
print("Diagnostic complete.")
