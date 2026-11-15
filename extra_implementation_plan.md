# Five-Day Parallel Implementation and V&V Plan

## Goal

Continue the main simulation and its outstanding validation while adding a bounded inverse-design proof of concept, a fresh-holdout runtime/accuracy benchmark, a volume-fraction sensitivity map, and two lightweight independent additions: a frozen-model load-path sweep and a report/figure packager. Keep model evaluations inside the trained domain, `v_f in [0.02, 0.10]`; use direct RVE checks where stated.

This plan assumes access to the final Phase 6/7 checkpoint and its dataset manifests. The repository's local `checkpoints/` folder currently contains Phase 1 and smoke weights only, so locate the frozen later checkpoint before running a real design. Do not train on or tune against the fresh validation cases.

**Progress:** The Phase 6/7 checkpoint, training dataset, and manifest are present and their SHA-256 values match. The bounded optimizer and pICNN derivative self-check passed. The 5% simple-shear attempt is preserved in `artifacts/phase6_phase7_final/results/rve_inverse_design.json`; its automatically generated target was outside the surrogate's predicted range and the direct-RVE check missed the 5% goal. The 5% uniaxial-stretch run completed on the HPC cluster with status `pass`: the model selected `v_f=0.0586515`, and the direct RVE response was within 0.237% of the midpoint target, below the 5% limit. The target was derived from direct-RVE endpoint responses, so this is a verified single-case proof of concept, not a result against an independently specified target.

The fresh holdout was generated and passed its 90-sample dataset audit. The pICNN-only load-path sweep completed with status `pass`: 165 points across five modes and three volume fractions; maximum work-conjugacy scaled error was `4.5987e-06` (tolerance `1e-05`), and maximum tangent major-symmetry error was `1.5176e-15` (tolerance `1e-08`). This sweep is surrogate-only and does not establish direct-RVE accuracy.

The corrected pICNN-versus-direct-RVE benchmark completed 30 held-out cases. Energy and stress NRMSEs on the training scales were `4.3047e-04` and `8.7895e-04`; median measured speedup was `57,844.7x` for repeated solver work and `2,725.1x` after amortizing mesh/reference setup. These are specific to the selected HPC node, 30-case subset, and measured energy/full-stress workload.

The volume-fraction map completed nine direct-RVE comparisons and passed its derivative check (maximum scaled error `7.3724e-07`, tolerance `1e-05`). Its raw energy RMSE was `0.0641` and raw stress-component RMSE `1.0175`; these are in the report's model units. The map status means the run/derivative checks passed and direct solves completed; no predeclared surrogate-to-RVE accuracy threshold was applied. Matplotlib was installed in the `ppp-ngsolve` environment and the report packager generated `inverse_design_curve.png`, `loading_path_curves.png`, and `volume_fraction_map.png`, alongside the combined summary JSON/CSV. The three-target inverse-design sweep (25%, 50%, and 75% of the direct-RVE endpoint response span) completed with status `pass` for all three targets. The largest direct target error was `0.2367%`, the largest surrogate-to-direct-RVE error was `0.2373%`, and the maximum scaled derivative error was `3.2094e-07` (tolerance `1e-05`). Target errors were `0.044%`, `0.237%`, and `0.071%`, respectively. The targets are simulation-derived rather than independent experimental targets. The simple-shear sweep's targets (`0.05456` to `0.05902` for `P01`) all fell below the surrogate-predicted range (`0.07632` to `0.10737`). The optimizer returned the lower bound `v_f=0.02` for all three. Across the map, surrogate `P01` rose from `0.07632` to `0.10737` as `v_f` increased from `0.02` to `0.10`, while direct RVE rose from `0.05233` to `0.06125`. At `v_f=0.02`, the surrogate was 45.85% above direct RVE. The 25% target happened to be within 4.09% of the direct-RVE boundary response, but it was not a surrogate-supported inverse-design success. The maximum direct target error was `11.34%`, despite a maximum scaled derivative error of `5.43e-09`. A training-only feature-coverage audit found the tested simple-shear states in a dense region of the model's invariant-feature space (nearest-neighbor distances at the 2nd–13.5th percentiles of training leave-one-out distances); this is descriptive evidence, not a coverage pass/fail threshold. Compression passed all three targets; its maximum direct target error was `2.056%`, maximum surrogate-to-direct-RVE error `2.099%`, and maximum scaled derivative error `2.270e-07`. None of these tasks changes the frozen checkpoint.

## Parallel workstreams

| Workstream | Work | Evidence to produce |
|---|---|---|
| Main simulation | Run a fresh grouped RVE holdout for the final checkpoint. Continue the macro-FE loading path and quantify reaction/energy changes across mesh sizes. Track the second-order solver issue separately so it does not consume the extra-work budget. | Frozen checkpoint/data hashes; held-out energy, stress, and tangent errors; macro-FE residual and reaction/energy-versus-mesh table; unresolved solver failures recorded. |
| Extra implementation | Complete the bounded inverse-design case; benchmark against a fresh holdout; map volume-fraction sensitivity; sweep additional loading paths; package results. Do not change the frozen model. | Verified inverse-design report; independent holdout errors and paired runtime measurements; map and derivative checks; load-path curves and consistency checks; combined final tables/figures. |

Use separate output folders and manifests for the two workstreams. Fresh holdout cases remain validation-only; the optimizer may use the frozen model but must not use their direct-RVE labels.

## Primary extra: one verified inverse-design case

### Implementation

1. Use a loading state already covered by the model's validation domain and one measurable response; the current default is the first-Piola shear stress `P01` at 5% simple shear.
2. Run direct RVE cases at the two volume-fraction endpoints. By default, define the target as the midpoint of those endpoint responses; an explicit target is also accepted only when it lies between them.
3. Search `v_f in [0.02, 0.10]` with the frozen pICNN. Save the target rule/value, load state, bounds, checkpoint hash, optimizer settings, and predicted optimum.

### Verification

- Compare the pICNN gradient of the objective with a central-difference diagnostic at several in-domain volume fractions.
- Confirm the grid-bracketed search is deterministic for fixed inputs, respects the bounds, and checks every sampled local minimum before choosing the best candidate.
- Check that the chosen deformation is orientation-preserving and does not exceed the validated deformation range.

### Validation

- Run the full RVE solver at both endpoint volume fractions, the proposed optimum, and two nearby volume fractions. Do not add these cases to training before scoring the design.
- Report the direct-RVE target error and compare the local response trend with the surrogate prediction.
- Use a predeclared target-error goal of 5% as a pass target; report the measured value even if it misses. Describe the result as a proof of concept, not a general material-design guarantee.

## Five-day sequence

| Day | Main simulation | Extra implementation |
|---|---|---|
| 1 | Freeze the final checkpoint, grouped split, and independent holdout cases. Launch fresh RVE holdout jobs. Record the macro-FE mesh/order/load settings. | Locate the checkpoint and manifests; choose a validated loading state and response; define the target rule from direct-RVE endpoint results. |
| 2 | Evaluate the final model on the fresh holdout. Continue the current macro-FE run and capture reaction, energy, residual, and runtime data. | Implement the bounded search and derivative/reproducibility checks. |
| 3 | Run the planned mesh or load-path comparisons; record second-order solver behavior if it is attempted. | Verify the inverse-design optimum with direct RVE; generate and audit the separate fresh holdout. |
| 4 | Summarize holdout errors and macro-FE response differences. Avoid changing the frozen test split. | Run the sensitivity map, matched runtime/accuracy benchmark, and independent pICNN load-path sweep. |
| 5 | Assemble the main-simulation results and list any remaining convergence gap. | Package the available outputs into combined tables/figures and summarize measured accuracy, speedup, derivative checks, and direct-RVE results. |

## Two supporting add-ons

### 1. Surrogate-versus-RVE accuracy and runtime benchmark

Generate a fresh grouped holdout with the final checkpoint frozen. Use 10 non-reference test deformation groups at all three volume fractions (30 matched cases). Measure direct-RVE and pICNN energy/stress evaluation time under recorded hardware and thread settings; report repeated-run medians, paired speedup, and held-out energy/stress errors. The benchmark intentionally compares the same energy/stress outputs; it does not time the more expensive tangent solve.

- **Verification:** Same inputs, units, volume fractions, and tensor ordering; deterministic outputs; timing includes a warm-up and repeated runs.
- **Validation:** Report errors on the frozen independent holdout, not training data. State the speedup only for the measured setup.
- **Why it fits:** The current report says the direct-RVE/surrogate speedup has not yet been measured, and this uses the existing solver and model.

### 2. Volume-fraction response and sensitivity map

Use the frozen pICNN to map energy, selected stresses, and the tangent Frobenius norm across `v_f = 0.02, 0.05, 0.10` for three representative loading modes: 5% uniaxial stretch, 5% simple shear, and 5% uniaxial compression. Compute the model's response sensitivity to `v_f`, then check all nine points against direct RVE results.

- **Verification:** Compare the pICNN's `v_f` derivative with a finite-difference diagnostic and check that results are repeatable.
- **Validation:** Compare selected map points with direct RVE labels and state clearly where the map interpolates or approaches the edge of the trained range.
- **Why it fits:** It adds a material-design result without introducing new geometry, a new constitutive law, or an ensemble-training pipeline.

## Additional independent tasks

### 3. Frozen-model multi-mode loading-path sweep

Evaluate the frozen pICNN at 11 amplitudes from 0% to 10% for uniaxial tension, uniaxial compression, simple shear, equibiaxial tension, and volumetric expansion, at all three in-domain volume fractions. Report energy, first-Piola stress, directional tangent, and determinant/condition number. Check the work-conjugate identity `dPsi/da = P : dF/da` against finite differences and check major symmetry of the 9x9 tangent.

- **Scope:** Surrogate-only exploratory curves. They do not replace direct-RVE validation or constitute a new independent test set.
- **Cost:** One CPU core; no NGSolve solves and no holdout results are required.

### 4. Result packager

Combine whichever inverse-design, map, benchmark, and loading-path JSON reports have completed into one summary JSON/CSV and thesis-ready PNG figures. The packager now also accepts repeated `--target-sweep MODE=PATH` inputs, recomputes surrogate-feasibility status from the stored target ranges, and creates a cross-mode inverse-design figure. It only reads reports and can be rerun when later jobs finish.

### 5. Multi-target inverse-design sweep

At 5% uniaxial stretch, define three target responses at 25%, 50%, and 75% of the direct-RVE response span between `v_f=0.02` and `v_f=0.10`. Optimize each target with the frozen pICNN, directly solve each selected volume fraction, and report target error plus surrogate-to-RVE error. This extends the one-target proof of concept without using the independent holdout labels or changing the model.

**Completed:** All three targets passed the 5% direct target-error criterion. The selected volume fractions were `0.041978`, `0.058651`, and `0.080898`; the largest direct target error was `0.2367%`, the largest surrogate-to-RVE error was `0.2373%`, and the maximum scaled derivative error was `3.2094e-07`.

### 6. Multi-mode inverse-design robustness sweep

Repeat the three-target endpoint-span design for 5% simple shear (`P01`) and 5% uniaxial compression (`P00`) with the same frozen checkpoint, volume-fraction bounds, optimizer, direct-RVE settings, and 5% target-error criterion. This adds two loading states to the completed uniaxial-stretch sweep. Targets remain derived from direct-RVE endpoint responses, not from external measurements. Preserve separate JSON/CSV reports per loading mode and report any target miss without changing the model.

- **Verification:** Run the same pICNN derivative check over the domain and at the selected volume fractions for each mode.
- **Validation:** Directly solve both endpoint volume fractions and each selected candidate; compare direct-RVE target error and surrogate-to-RVE error for all three targets.
- **Why it fits:** It reuses the completed optimizer and validated loading states from the map and loading-path sweep; it requires no new training or holdout use.

**Results:** The simple-shear case produced a partial result (1/3 target passed; maximum direct target error `11.34%`; maximum surrogate-to-RVE error `45.85%`). The derivative check passed, which confirms the pICNN gradient implementation but not shear accuracy. The compression-only retry passed all three targets; maximum direct target error was `2.056%`, maximum surrogate-to-RVE error was `2.099%`, and maximum scaled derivative error was `2.270e-07`.

**Shear interpretation:** The three `P01` targets (`0.05456` to `0.05902`) were all outside the surrogate's predicted response range (`0.07632` to `0.10737`). All selected candidates therefore saturated at `v_f=0.02`. One direct-RVE boundary evaluation happened to meet its target within 5%, but the surrogate prediction at that point was 45.85% from the direct response. Treat this mode as outside the demonstrated inverse-design capability.

### 7. Surrogate-feasibility gate for inverse-design reports

Update the target-sweep report so a design passes only when the target is inside the surrogate-predicted response range, the surrogate predicted target error meets the selected limit, and the direct-RVE target error meets that limit. Continue reporting the direct-RVE target result separately, so a boundary hit cannot be mistaken for a surrogate-supported design. This is a reporting and decision safeguard; it does not alter the frozen checkpoint or require rerunning the completed reports.

- **Implementation:** The target-sweep script now reports `direct_rve_target_status` separately from overall `status`, and distinguishes surrogate-unreachable targets.
- **Validation:** Existing simple-shear values show all three targets are outside the model response range; under the new rule, none count as a verified inverse-design pass. The stretch and compression direct-RVE results are unchanged; their predicted-range flags can be checked from their saved reports when applying the revised status rule.

### 8. Training-only shear feature-coverage audit

Compare the 5% simple-shear query state and nearby amplitudes against deformation states in the frozen training split, using the same invariant features supplied to the pICNN. Report nearest-neighbor distances relative to leave-one-out distances within the training set. Do not read or use stress/energy labels and do not select a pass/fail distance threshold.

**Completed:** The audit used 909 training rows across 303 unique deformation states and volume fractions `0.02`, `0.05`, and `0.10`; the dataset SHA-256 matches the manifest (`8553db63af311e0a7e987c5c3f520b49ce0f6160a880cc0314f8d36864513336`). At shear amplitudes `0.025`, `0.05`, and `0.075`, nearest standardized invariant-feature distances fell at the 13.5th, 8.3rd, and 2.0th percentiles of the training leave-one-out nearest-distance distribution. This suggests the tested shear states are not sparse in invariant-feature space; it does not explain the stress error or establish surrogate accuracy. JSON and CSV outputs are in `artifacts/phase6_phase7_final/results/shear_training_coverage.{json,csv}`.

### 9. Component-wise stress-error audit

Reprocess the saved 30-case holdout benchmark by all nine first-Piola stress components, both in aggregate and by volume fraction. Report raw errors, error relative to each component's holdout-label RMS, and error relative to the global training stress scale. This checks whether a global stress NRMSE obscures a large relative error in a low-magnitude component such as shear. It uses the saved benchmark tensors only and requires no additional RVE solves.

**Completed:** The audit covered the saved 30-case holdout. Component-relative NRMSE ranged from `0.053%` to `0.221%`; `P01` was `0.122%`. Direct-RVE-versus-holdout component RMSEs were around `1e-12`, confirming the rerun reproduces the stored labels. These broad holdout metrics support stress accuracy on those selected cases. They do not establish the relative accuracy needed for the low-magnitude simple-shear inverse-design targets (`P01` around `0.055`), where an absolute discrepancy of about `0.024` produced a 45.85% relative error. The componentwise audit JSON/CSV were written to `rve_extra_results_summary_final_20261114/stress_component_audit.{json,csv}` on the HPC filesystem.

### 10. Direct-RVE simple-shear path validation

Add direct-RVE checks at 2.5% and 7.5% simple shear for `v_f=0.02, 0.05, 0.10`. Reuse the three completed 5% points from the volume-fraction map, producing a nine-point amplitude-by-volume-fraction comparison of pICNN energy and stress against direct RVE. Report both absolute and local relative P01 errors; do not set a pass/fail cutoff after seeing the results.

**Completed:** The six new direct-RVE solves plus the three reused 5% baseline points all converged. The frozen pICNN overpredicts `P01` by 45.9%, 55.3%, and 75.3% at `v_f=0.02, 0.05, 0.10`, respectively; these relative errors remain nearly constant from 2.5% through 7.5% shear. Maximum `P01` absolute error was `0.06914`, maximum stress-component RMSE `0.03517`, and maximum full-stress Frobenius relative error `81.05%`. This is a systematic, volume-fraction-dependent shear-response bias, not an isolated 5% point. The frozen checkpoint does not support the tested simple-shear response targets. Report: `/scratch4/vr80beqo/rve_extra_shear_path_validation_20261114/rve_shear_path_validation.json`.

### 11. Cross-scale tangent validation for implicit FE use

Compare the frozen pICNN's autograd-derived material tangent with the condensed free-rotation RVE tangent at 0.5% uniaxial stretch, 1% uniaxial stretch, and 5% simple shear, all at `v_f=0.05`. Record stress and energy agreement, 9x9 tangent relative error and symmetry, condensed residuals, and RVE equilibrium diagnostics. Interpret these metrics alongside the macro-FE Newton histories at matching mesh and load settings. This tests whether the surrogate tangent represents the microscale response and whether tangent discrepancies are associated with the macro solver's convergence behavior; the selected states alone cannot establish causality.

**Completed:** The direct RVE states equilibrated with mixed residuals from `1.38e-14` to `2.45e-11` and rotation residuals below `6.6e-12`. At 0.5% and 1% uniaxial stretch, energy, full-stress, and full-tangent relative errors were about 1.1–1.3%. The uniaxial `P01` relative-error value of 100% is ill-conditioned because the reference component is near zero; absolute errors were only `4.565e-6` and `9.097e-6`.

At 5% simple shear, energy error was `55.309%` and full-stress error `58.132%`. Although the full 9x9 tangent Frobenius relative error was only `1.030%`, the active shear component `dP01/dF01` was `1.73122` for the pICNN versus `1.11488` for direct RVE, a `55.283%` component error. The aggregate tangent norm therefore masked a large shear-mode error. This agrees with the nine-point shear-path validation. The stretch states show no similarly large constitutive discrepancy that by itself explains macro-FE convergence difficulties; this comparison does not rule out errors at other local states or discretization/solver causes. Report: `/scratch4/vr80beqo/rve_extra_picnn_tangent_validation_20261114/rve_picnn_tangent_validation.json`.

### 12. Acoustic-tensor stability screening

Use the saved pICNN and direct-RVE 9x9 tangents to form the acoustic tensor `Q_ik(N)=A_iJkL N_J N_L` over a deterministic sphere-direction sample. Report the smallest sampled eigenvalue, its reference direction, and polarization for each tangent/state. This is a stability screen over sampled directions; a positive sampled minimum is not a proof of strong ellipticity over the continuous sphere. The postprocessor uses the completed tangent report and performs no retraining or additional RVE solves.

**Completed:** The deterministic directional screen sampled 655 propagation directions for each of six tangents. All pICNN and direct-RVE tangents had positive minimum sampled eigenvalues. The pICNN minima were 1.6517–1.7806 for the three states, while the direct-RVE minima were 1.1060–1.1140. This supports sampled strong ellipticity at those states; finite directional sampling is not a proof over the continuous sphere. The higher pICNN minima at the tested states are consistent with its stiffer shear tangent, but do not establish a causal relation. Report: `/scratch4/vr80beqo/rve_extra_picnn_tangent_validation_20261114/acoustic_tensor_audit.json`.

### 13. Periodic-RVE material-isotropy and cell-orientation audit

For each of six fixed deformation states, compare direct-RVE energy and stress after three deterministic proper right rotations of the same macroscopic gradient. For an isotropic material, the expected relations are `W(FQ)=W(F)` and `P(FQ)=P(F)Q`. Retain the periodic cell orientation and solver settings, and record equilibrium and microscopic-J diagnostics for every solve. The purpose is to quantify orientation sensitivity introduced by the finite periodic cell and its cubic arrangement; do not impose a post hoc pass threshold or interpret finite-cell differences as failure of the solver's spatial objectivity.

**Completed:** Six selected Phase 7 deformation/volume-fraction records were evaluated in the base orientation and under three deterministic non-cubic right rotations (24 direct-RVE state evaluations total). All 18 rotated comparisons completed. Maximum relative energy difference was `5.5602e-05` (0.00556%) and maximum relative stress difference was `2.8035e-05` (0.00280%). No post hoc pass threshold was applied. The small differences support the isotropy assumption for the selected states and rotations, while this finite sample does not prove isotropy over all deformation states or all rotations. Report: `/scratch4/vr80beqo/rve_extra_material_isotropy_20261115/rve_material_isotropy_check.json`; CSV: `/scratch4/vr80beqo/rve_extra_material_isotropy_20261115/rve_material_isotropy_check.csv`.

### 14. Small-strain effective-modulus benchmark against spherical micromechanics

At `F=I` and `v_f=0.02, 0.05, 0.10`, project the stored 9x9 direct-RVE tangent and frozen pICNN Hessian onto one volumetric and five orthonormal deviatoric strain modes. Report effective bulk/shear moduli, equivalent Young's modulus and Poisson ratio, tangent symmetry, volumetric/deviatoric coupling, and deviatoric anisotropy. Compare the direct RVE values with the rigid-sphere Mori-Tanaka estimate, which is the corresponding Hashin-Shtrikman lower reference for a stiffer inclusion under standard isotropic assumptions. The matrix parameters are `K=10000`, `G=1`; record them with the results. The selected reference-state pICNN labels are in its training dataset, so that comparison is in-sample. This analytical comparison checks the RVE homogenization trend independently of surrogate fitting; it is not a nonlinear or held-out validation.

**Results:** The direct RVE moduli increase with particle fraction: `K=10185, 10469, 10988` and `G=1.048, 1.124, 1.267` at `v_f=0.02, 0.05, 0.10`. The isotropic Mori-Tanaka/Hashin-Shtrikman lower reference is slightly above the direct RVE projection: by 0.19–1.11% for `K` and 0.25–0.83% for `G`. Since the finite periodic cell is not exactly isotropic (direct-RVE deviatoric anisotropy rises from 0.21% to 4.12%), treat this as a comparison with an isotropic reference, not as a clean bound violation. The pICNN bulk moduli differ from the direct RVE by about +0.45%, -1.03%, and -1.05%, while its shear moduli are higher by about 45.6%, 54.0%, and 69.5%. The pICNN tangent is isotropic to numerical precision by construction. This provides an independent small-strain signal of the shear stiffness bias; it does not perform or select the shear adaptation.

The selected rows are `test` split records, so they were excluded from fitting. The Phase 7 test set was inspected in earlier project work, so this is not a fresh independent validation. The final regenerated JSON records this split-specific scope. The theoretical references are Mori and Tanaka (1973), DOI `10.1016/0001-6160(73)90064-3`, and Hashin and Shtrikman (1963), DOI `10.1016/0022-5096(63)90060-7`.

## HPC submission order

1. Submit `rve_extra_holdout.pbs`; require its dataset audit to pass.
2. Submit `rve_loading_path_sweep.pbs` independently; it requests one CPU core and does not need the holdout.
3. Submit `rve_extra_volume_fraction_map.pbs` when cluster capacity allows. It writes a JSON report, CSV table, and run log.
4. After the holdout dataset and manifest exist, submit `rve_extra_surrogate_benchmark.pbs`. It writes the paired timing/accuracy report and per-case CSV.
5. Copy completed JSON reports to one folder and run `python -m benchmarks.summarize_extra_results` to build the combined table and figures.
6. Submit `rve_extra_inverse_design_target_sweep.pbs` as a separate short RVE validation job; it writes per-target results to JSON/CSV.
7. Submit `rve_extra_inverse_design_loading_modes.pbs` to run the same three-target verification for simple shear and uniaxial compression; it writes one JSON/CSV pair per mode.
8. The first loading-mode job completed the simple-shear portion but stopped before compression due to a malformed deformation-gradient argument list. The corrected compression-only retry has completed and passed all three targets.
9. **Completed:** `rve_extra_shear_path_validation.pbs` added six direct-RVE points at 2.5% and 7.5% shear, reusing the completed 5% map points.
10. **Completed:** `rve_extra_picnn_tangent_validation.pbs` compared the frozen surrogate tangent with the direct free-rotation RVE tangent at three representative states. The saved full tangent matrices support componentwise post-processing; future tangent audits should report active shear components as well as aggregate norms.

The benchmark uses the checkpoint SHA recorded above. Do not start it with the older Phase 7 evaluation dataset; the fresh holdout is generated specifically to avoid reusing that previously inspected test set.

## Collected result snapshot

| Study | Measured outcome | Interpretation |
|---|---|---|
| Inverse design | `v_f=0.0586515`; direct response `528.1448` versus midpoint target `529.3978`; relative error `0.237%` | Passes the 5% target criterion for one uniaxial-stretch case. The target was generated from endpoint RVE responses. |
| Fresh holdout benchmark | 30 cases; energy/stress NRMSE on training scales `4.30e-4`/`8.79e-4`; median amortized speedup `2,725x` | Low surrogate errors on the selected fresh test subset; report speedup only for the measured setup and outputs. |
| Volume-fraction map | 9 direct RVE cases; max derivative error `7.37e-7`; energy/stress raw RMSE `0.0641`/`1.0175` | Derivative check passed. Raw errors have no pass/fail cutoff here; compare them to training response scales before making an accuracy claim. |
| Loading-path sweep | 165 points; work-conjugacy error `4.60e-6`; tangent symmetry error `1.52e-15` | Internal surrogate consistency only; no direct-RVE labels were used. |
| Three-target inverse-design sweep | 3/3 targets passed; target errors `0.044%`, `0.237%`, `0.071%`; max derivative error `3.21e-7` | 5% uniaxial stretch; targets were fractions of the direct-RVE endpoint span, not independent experimental targets. |
| Multi-mode inverse design | Stretch: 3/3 surrogate-supported passes; compression: 3/3 passes; shear: 0/3 surrogate-supported targets | All shear targets were below the predicted response range. One direct boundary response happened to be within 4.09% of target, but the surrogate/RVE discrepancy was 45.85%. |

## Go/no-go decisions

- If the final checkpoint or manifest cannot be found on Day 1, do not substitute Phase 1 weights for the RVE surrogate; switch to the runtime benchmark or map only after recovering the intended model.
- If fresh holdout errors are poor, do not claim a validated inverse design. Use the holdout findings to define a validity map instead.
- If the direct RVE at the proposed optimum fails to converge, report the candidate as unvalidated and complete the runtime benchmark as the extra contribution.
- Do not attempt the full ensemble + active-learning + inverse-design package in this five-day window. A single well-verified extension is the target.
