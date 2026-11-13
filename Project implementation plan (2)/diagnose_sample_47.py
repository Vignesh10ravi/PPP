import numpy as np

from src.rve_dataset import (
    DatasetConfig,
    generate_deformation_states,
    particle_radius,
    solve_dataset_sample,
)
from src.rve import make_periodic_particle_cell_mesh

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

gradients = generate_deformation_states(config)
F = np.asarray(gradients[state_index], dtype=float)

print("=" * 65)
print("TARGETED RVE SAMPLE DIAGNOSTIC")
print("=" * 65)
print(f"Sample ID:       v000_f{state_index:06d}")
print(f"Volume fraction: {volume_fraction}")
print(f"Mesh maxh:       {config.maxh}")
print(f"det(F):          {np.linalg.det(F):.12f}")
print(f"cond(F):         {np.linalg.cond(F):.12f}")
print("F:")
print(F)
print("=" * 65, flush=True)

print("Building periodic particle mesh...", flush=True)
mesh = make_periodic_particle_cell_mesh(
    radius=particle_radius(volume_fraction),
    maxh=config.maxh,
)

print("Solving this sample...", flush=True)

result = solve_dataset_sample(
    mesh=mesh,
    deformation_gradient=F,
    volume_fraction=volume_fraction,
    config=config,
)

print("\nSAMPLE SOLVED SUCCESSFULLY")
for key in (
    "energy",
    "determinant",
    "condition_number",
    "mixed_residual",
    "rotation_residual",
    "condensed_residual",
    "tangent_symmetry_error",
):
    if key in result:
        print(f"{key}: {np.asarray(result[key])}")

print("\nAll result fields are finite:",
      all(np.all(np.isfinite(value))
          for value in result.values()
          if np.issubdtype(np.asarray(value).dtype, np.number)))
