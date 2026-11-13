import numpy as np
from ngsolve import (
    CF, Det, Grad, GridFunction, Id, IntegrationRule, ET,
    dx, x, y, z, Integrate,
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

macro_cf = CF(tuple(F.ravel()), dims=(3, 3))
macro_u = (macro_cf - Id(3)) * CF((x, y, z))
measure = dx(intrules={
    ET.TET: IntegrationRule(ET.TET, order=config.integration_order),
})

def inspect(label, impose_particle_boundary):
    solution = GridFunction(fes)

    if impose_particle_boundary:
        solution.components[0].Set(
            -macro_u,
            definedon=mesh.Boundaries(PARTICLE_BOUNDARY),
        )

    local_F = macro_cf + Grad(solution.components[0])
    J = Det(local_F)

    j_values = np.asarray(
        Integrate(J * measure, mesh, element_wise=True),
        dtype=float,
    ).ravel()

    power_values = np.asarray(
        Integrate(J ** (-2.0 / 3.0) * measure, mesh, element_wise=True),
        dtype=float,
    ).ravel()

    print(f"\n{label}")
    print("  finite J integrals:", np.count_nonzero(np.isfinite(j_values)))
    print("  negative J element integrals:", np.count_nonzero(j_values < 0))
    print("  non-finite power integrals:", np.count_nonzero(~np.isfinite(power_values)))
    print("  minimum J element integral:", np.nanmin(j_values))
    print("  maximum J element integral:", np.nanmax(j_values))

inspect("Zero fluctuation", False)
inspect("Particle boundary imposed", True)
