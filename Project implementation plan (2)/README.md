# pICNN → NGSolve : Phase 1 (Proof of Concept)

PyTorch↔NGSolve UMAT bridge with a polyconvex Input-Convex Neural Network that
reproduces compressible Neo-Hookean, verified by exact-tangent and
quadratic-convergence checks. This is the committee-requested proof of concept
that must pass **before** any RVE data generation (Phase 2).

## Layout
```
src/
  neohooke.py      compressible Neo-Hookean reference + Phase-1 data
  picnn.py         polyconvex (p)ICNN + stress-free/zero-energy normalization
  constitutive.py  exact P and A (3x3x3x3) via autograd; 9x9 flattening
  train.py         Sobolev training (energy + stress), checkpoint I/O
  ngsolve_umat.py  IntegrationRuleSpace bridge + custom Newton-Raphson
  rve.py           periodic cell and exact finite rigid-particle kinematics
  mixed_matrix.py  Taylor-Hood near-incompressible matrix formulation
  rve_solver.py    coupled periodic mixed-matrix/rigid-particle solve
  homogenization.py variational macrostress and Hill-Mandel work
  rve_tangent.py    exact condensed prescribed-particle macroscopic tangent
  free_particle.py  free finite particle rotation by exact condensation
  rve_dataset.py    deterministic checkpointed RVE dataset generation
  generate_rve_dataset.py Phase-2 dataset command-line entry point
tests/
  verify_model_interface.py  V1: deformation/feature API contract
  verify_reference.py  V2: zero energy and stress at F = I
  verify_neohooke_response.py  V2b: NN vs analytical Neo-Hookean response
  verify_tangent.py  V3: autograd tangent vs central difference
  verify_element.py  V4: single-element quadratic Newton convergence
  verify_patch.py  V5: multi-element affine patch test
  verify_fe_comparison.py  V6: analytical vs neural FE benchmark
  verify_periodic_rve.py   P2-V1: periodic mesh/space verification
  verify_rigid_particle.py P2-V2: rigid-particle geometry and constraints
  verify_mixed_matrix.py   P2-V3: mixed matrix affine patch verification
  verify_coupled_rve.py    P2-V4: coupled periodic rigid-particle equilibrium
  verify_hill_mandel.py    P2-V5: homogenized stress and macrohomogeneity
  verify_homogenized_tangent.py P2-V6: condensed 9x9 tangent verification
  verify_free_rotation.py P2-V7: free particle moment equilibrium
  verify_translation_gauge.py P2-V8: periodic translation-gauge verification
  verify_free_rotation_tangent.py P2-V9: tangent with rotation condensation
  verify_rve_dataset.py P2-V10: dataset sampling/split/serialization contract
```

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run (in order — each is a checkpoint)
```bash
python -m tests.verify_model_interface # V1  (torch only)
python -m tests.verify_reference       # V2  (torch only)
python -m tests.verify_tangent        # V3  (torch only)
python -m src.train --epochs 1500 --ckpt checkpoints/phase1.pt
python -m tests.verify_neohooke_response # trained-response validation
python -m tests.verify_element        # V4  (needs ngsolve)
python -m tests.verify_patch          # V5  (needs ngsolve)
python -m tests.verify_fe_comparison  # V6  (needs ngsolve)
```

Resume a checkpoint for additional epochs with a new learning rate:
```bash
python -m src.train --resume checkpoints/phase1.pt --epochs 2000 \
  --lr 1e-3 --ckpt checkpoints/phase1_refined.pt
```

## What each verification proves
- **V1** — the public material API consumes deformation gradients through
  `model.energy(F)`, while precomputed features are consumed by
  `model.net(features)`; stress and tangent shapes are consistent.
- **V2** — reference normalization gives both `Psi(I) = 0` and `P(I) = 0` to
  floating-point precision.
- **V2b** — on unseen deformations, the trained network's energy, first-Piola
  stress, and full tangent agree with the analytical Neo-Hookean material.
- **V3** — the autograd tangent A = d²Ψ/dF² is exact, independent of training.
  The finite-difference error is V-shaped, bottoming ~1e-8 near eps=1e-6;
  machine precision is unreachable by FD *by construction* (this is the answer
  to the slide-8 question about the perturbation parameter).
- **V4** — the NN material, plugged into NGSolve via `IntegrationRuleSpace`,
  drives a custom Newton-Raphson loop to **quadratic** residual convergence
  (~1e-1 → 1e-3 → 1e-8), proving the assembled global stiffness uses a
  consistent tangent.
- **V5** — a mixed stretch/shear affine field remains uniform over a
  multi-element mesh and has zero assembled residual on all free interior
  degrees of freedom.
- **V6** — matched analytical and neural nonlinear FE problems agree in their
  free displacements, integration-point deformation gradients, and reaction
  vectors.

## Design notes (corrections vs. the original slides)
- Reference normalization subtracts the **stress** term too:
  `Ψ̂(F) = Ψ_raw(F) − Ψ_raw(I) − P_raw(I):(F−I)` → truly stress-free at F=I.
- Tangents are **autograd**, never finite differences.
- Total-Lagrange P/F framework → full **9×9** tangent (not the 6×6 minor-
  symmetric Voigt, which is only valid for the 2nd-Piola/Green-Lagrange tangent).
- Polyconvexity via convex features [I1, I2, J, −lnJ] + non-negative weights.

## Phase 2 (after Phase 1 is green)
Periodic RVE with `Periodic()` FE spaces on a periodic mesh; rigid inclusions
via rigid-body dof constraints (not E→∞ penalty); mixed u–p for the near-
incompressible PDMS matrix; Hill–Mandel averaging with a **consistent**
(statically condensed) macroscopic tangent; retrain as a *partial* ICNN with
volume fraction as the non-convex context input.

Phase 2 is split into independently verifiable gates. The first gate checks
the periodic-cell infrastructure before particle and constitutive constraints
are introduced:
```bash
python -m tests.verify_periodic_rve  # P2-V1
python -m tests.verify_rigid_particle  # P2-V2
python -m tests.verify_mixed_matrix  # P2-V3
python -m tests.verify_coupled_rve  # P2-V4
python -m tests.verify_hill_mandel  # P2-V5
python -m tests.verify_homogenized_tangent  # P2-V6
python -m tests.verify_free_rotation  # P2-V7
python -m tests.verify_translation_gauge  # P2-V8
python -m tests.verify_free_rotation_tangent  # P2-V9
python -m tests.verify_rve_dataset  # P2-V10
```

P2-V5 uses the matrix-volume stress average **plus assembled FE particle
reaction work**, with matching order-10 quadrature in the RVE solver and
homogenization. It checks an independently constructed virtual field and
invariance to an additional free virtual fluctuation. Raw outer-boundary
stress integration is reported only as a stress-recovery diagnostic, not as
an exactly equilibrated FE reaction. This gate covers the prescribed
nonrotating particle whose center follows the macroscopic deformation; it
does not yet establish free-particle rotational equilibrium under general
loading or the condensed macroscopic tangent.

P2-V6 symbolically differentiates the mixed discrete energy with nine global
macroscopic coordinates and eliminates the free fluctuation and pressure DOFs
through a Schur complement. The full first-Piola tangent uses row-major
`(i,J) -> i*3+J` indexing; it is not a 6x6 Voigt tangent. Verification checks
major symmetry without symmetrizing the result, all nine independent
stress sensitivities, linearized equilibrium, and second-order nonlinear
Taylor remainders. Perturbed nonlinear states are used only for verification,
never to construct the tangent. Particle orientation is still prescribed;
free-particle rotations and their geometric tangent terms remain separate work.

P2-V7 adds those three finite rotation coordinates for general loading. The
particle moment and rotational Hessian come from first and second derivatives
of the discrete energy, with mixed fluctuation/pressure DOFs eliminated by a
Schur complement. Rotations update multiplicatively with Rodrigues' formula;
no stiffness penalty or finite-difference moment is used.

P2-V8 verifies why only the three rotations are condensed. Under periodic
fluctuation boundary conditions, uniform particle translation is the global
translation gauge: changing it shifts the entire fluctuation by a constant
without changing deformation, pressure, stress, or energy. A nominal 6x6
rigid-coordinate Hessian is therefore singular by construction. The gauge is
fixed by mapping the particle center with the macroscopic deformation; adding
a translation penalty or attempting to invert those three null modes would be
incorrect.

P2-V9 augments the nine macroscopic coordinates with the three physical
left-multiplicative rotation increments and condenses rotation, fluctuation,
and pressure together. Translation remains gauge-fixed. The resulting full
9x9 tangent includes rotational geometric terms and is checked for energy-
gradient consistency, major symmetry, all nine stress sensitivities, and
linearized force--moment equilibrium.

P2-V10 assigns each macroscopic deformation state to one split and evaluates
that complete group at every requested volume fraction, preventing the same
deformation from leaking across train, validation, and test data. Every RVE
sample is atomically checkpointed before final assembly. Generate a production
dataset with:
```bash
python -m src.generate_rve_dataset \
  --output data/rve_phase2.npz \
  --volume-fractions 0.02 0.05 0.10 \
  --n-deformations 100 --seed 1729
```
The compressed NPZ stores volume fraction, macroscopic deformation gradient,
homogenized energy, first-Piola stress, full row-major 9x9 tangent, particle
rotation, split label, and convergence/admissibility diagnostics. The adjacent
JSON manifest records the exact configuration, schema, software version,
array shapes, and split counts.
