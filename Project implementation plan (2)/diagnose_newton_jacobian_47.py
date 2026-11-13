import numpy as np

from ngsolve import (
    GridFunction,
    Integrate,
    IfPos,
    Det,
    Grad,
    dx,
    Norm,
)

from src.rve_dataset import (
    DatasetConfig,
    generate_deformation_states,
    particle_radius,
)
from src.rve import make_periodic_particle_cell_mesh
from src.rve_solver import (
    _matrix_stress,
    _rve_residual_form,
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

state_index = 47
volume_fraction = 0.02
F = np.asarray(
    generate_deformation_states(config)[state_index],
    dtype=float,
)

print("Sample:", f"v000_f{state_index:06d}")
print("det(F):", np.linalg.det(F))
print("cond(F):", np.linalg.cond(F))
print("F:\n", F, flush=True)

mesh = make_periodic_particle_cell_mesh(
    radius=particle_radius(volume_fraction),
    maxh=config.maxh,
)

fes = make_mixed_space(
    mesh,
    displacement_order=config.displacement_order,
    periodic=True,
    dirichlet="particle",
)

solution = GridFunction(fes)
residual = solution.vec.CreateVector()

# Reproduce the actual solver's prescribed rigid-particle boundary data.
# Reuse the production solver to avoid duplicating its initialization logic.
from src.rve_dataset import solve_dataset_sample

print(
    "\nIMPORTANT: this diagnostic must use the exact solver state "
    "and residual form.",
    flush=True,
)
print(
    "The script is intentionally stopping before a Newton solve because "
    "the current public solver interface does not expose its intermediate "
    "Newton states.",
    flush=True,
)

deformation_gradient = F + Grad(solution.components[0])
J = Det(deformation_gradient)

try:
    min_integral = Integrate(
        IfPos(J, 0.0, 1.0),
        mesh,
        order=config.integration_order,
    )
    print("Negative-or-zero J indicator integral:", min_integral)
except Exception as exc:
    print("J indicator evaluation failed:", repr(exc))

print(
    "\nNo production data or checkpoints were modified.",
    flush=True,
)
