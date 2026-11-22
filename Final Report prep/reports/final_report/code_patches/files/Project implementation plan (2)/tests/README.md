# Verification tests (report Chapter 7)

Run everything with `./run_tests.sh quick|full|audit` from the folder that contains
`src/`, `tests/`, `benchmarks/` (see the header of `run_tests.sh` for paths to the
archived checkpoints/datasets; on the cluster use `MODE=full qsub -v MODE run_tests.pbs`).
Every test prints its metrics and ends with `... PASS ...`; a failed check raises an
`AssertionError` with an explanation. Inputs are generated inside the test unless listed.

| ID | Command | Inputs | Aim | Expected result | Time |
|---|---|---|---|---|---|
| V0a-e | `python -m tests.verify_fe_basics --ckpt checkpoints/phase1_validated.pt --rve-ckpt <frozen.pt>` | both checkpoints (optional) | partition of unity, linear-field reproduction, Jacobian volume/area, 6 rigid-body zero eigenvalues (tet/hex, P1/P2), 18 zero modes with a one-point hex rule | errors ~1e-14; 6 zero modes for analytic law and RVE pICNN; legacy Phase 1 net reported (3 zero, 3 negative) | 20 s |
| V1 | `python -m tests.verify_model_interface` | - | API shapes, dtypes, finiteness | shapes (4), (4,3,3), (4,3,3,3,3), float64 | 3 s |
| V2 | `python -m tests.verify_reference` | - | Psi(I)=0, P(I)=0 | exactly 0 | 3 s |
| V2b | `python -m tests.verify_neohooke_response --ckpt checkpoints/phase1_validated.pt` | Phase 1 ckpt | trained net vs analytic neo-Hooke, 256 states | 5.367 / 3.274 / 3.865 % < 10 / 10 / 25 % | 2 s |
| V3 | `python -m tests.verify_tangent` | - | AD tangent vs central differences | V-curve, min error < 1e-6 near eps = 1e-6 | 4 s |
| V4 | `python -m tests.verify_element --ckpt checkpoints/phase1_validated.pt` | Phase 1 ckpt | Newton convergence of the FE solve | 4 iterations, order -> 1.9, residual 4e-11 | 3 s |
| V5 | `python -m tests.verify_patch --ckpt checkpoints/phase1_validated.pt` | Phase 1 ckpt | affine patch test, 206 irregular tets | F error 4e-16, free residual 3e-16 | 5 s |
| V6 | `python -m tests.verify_fe_comparison --ckpt checkpoints/phase1_validated.pt` | Phase 1 ckpt | neural vs analytic FE solution | errors 2.0 / 0.35 / 13.4 % < 15 % | 5 s |
| V7/V8 | `python -m tests.verify_stress_control --ckpt checkpoints/phase1_validated.pt` | Phase 1 ckpt (optional) | single hex: full deformation control (stress + tangent vs FD) and uniaxial stress control, St. Venant-Kirchhoff then neo-Hooke | stretches = analytic to 1e-15, P = diag(t,0,0), quadratic Newton per load step | 30 s |
| P2-V1 | `python -m tests.verify_periodic_rve` | - | periodic fluctuation space | opposite-face jump ~2e-16 | 1 s |
| P2-V2 | `python -m tests.verify_rigid_particle` | - | exact rigid kinematics | rank 6, residual ~1e-17 | 1 s |
| P2-V3 | `python -m tests.verify_mixed_matrix` | - | Taylor-Hood patch, kappa/mu = 1e4 | F, J, p exact to ~1e-13 | 1 s |
| P2-V4 | `python -m tests.verify_coupled_rve` | - | coupled periodic RVE equilibrium | residual ~2e-14 | 20 s |
| P2-V5 | `python -m tests.verify_hill_mandel` | - | Hill-Mandel macro-homogeneity | relative error ~1e-15 | 20 s |
| P2-V6 | `python -m tests.verify_homogenized_tangent` | - | condensed tangent: symmetry, Taylor order | orders 2.000, 2.000 | 3.5 min |
| P2-V7 | `python -m tests.verify_free_rotation` | - | free rotation under simple shear | 3 iterations, R in SO(3) | 3.5 min |
| P2-V8 | `python -m tests.verify_translation_gauge` | - | translation-gauge invariance | changes ~1e-15 | 1 min |
| P2-V9 | `python -m benchmarks.rve_constitutive_checks --dataset <pilot.npz> --output p2v9.json --sample-ids v001_f000047` | pilot dataset | free-rotation tangent vs re-solved central differences (9 directions) | status pass, error ~5e-12 | ~1 h |
| P2-V10 | `python -m tests.verify_rve_dataset` | - | deterministic dataset, leak-free splits | 0 leaks | 1 s |
| S-V1 | `python -m benchmarks.rve_surrogate_constitutive_checks --dataset <phase7.npz> --manifest <phase7.manifest.json> --checkpoint <frozen.pt> --training-dataset <training.npz> --report sv1.json --states-per-det-bin 2 --fd-step 1e-5` | frozen ckpt, datasets | surrogate energy gradient, tangent, objectivity, symmetry | status pass, all errors <= 5e-8 | 6 s |
| M-V0 | `python -m benchmarks.phase6_macro_fe_demo --checkpoint <frozen.pt> --output mv0.json --stretch 0.01 --load-steps 2 --order 1 --integration-order 4 --merit-function energy --tangent-check-at-load-step 2 --tangent-check-iteration 0 --tangent-check-only` | frozen ckpt | macro stiffness vs directional differences of the residual | directional error ~6e-10, symmetry ~1e-16 | 10 s |
