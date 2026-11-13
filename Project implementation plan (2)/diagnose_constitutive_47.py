import numpy as np
from ngsolve import (
    CF, Det, Grad, GridFunction, Id, Inv, IntegrationRule, ET,
    InnerProduct, Trace, VectorH1, H1, Periodic, x, y, z, dx, Integrate
)

from src.rve import PARTICLE_BOUNDARY, make_periodic_particle_cell_mesh
from src.rve_dataset import DatasetConfig, generate_deformation_states, particle_radius
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

macro_cf = CF(tuple(F.ravel()), dims=(3, 3))
macro_u = (macro_cf - Id(3)) * CF((x, y, z))
solution.components[0].Set(
    -macro_u,
    definedon=mesh.Boundaries(PARTICLE_BOUNDARY),
)

fluctuation, pressure = solution.components
local_F = macro_cf + Grad(fluctuation)
J = Det(local_F)
measure = dx(intrules={
    ET.TET: IntegrationRule(ET.TET, order=config.integration_order),
})

tests = [
    ("J", J),
    ("J^(-2/3)", J ** (-2.0 / 3.0)),
    ("I1", Trace(local_F.trans * local_F)),
    ("inverse_F", Inv(local_F)[0, 0]),
    ("inverse_transpose", Inv(local_F).trans[0, 0]),
]

for name, expression in tests:
    try:
        value = Integrate(expression * measure, mesh)
        print(f"{name}: integral={value}, finite={np.isfinite(value)}")
    except Exception as exc:
        print(f"{name}: ERROR {type(exc).__name__}: {exc}")

print("Done.")
