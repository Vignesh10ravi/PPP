# Project Implementation Plan: Hyperelastic RVE Surrogate

This repository archives the code and computational results for a project on a
physics-informed neural-network surrogate for the homogenized response of a
hyperelastic composite. The workflow combines direct finite-element RVE
simulations, energy-based surrogate training, constitutive checks, and a
small-scale macro-FE demonstration.

## Repository map

- `Project implementation plan (2)/` — Python source, benchmarks, verification
  tests, pinned environment requirements, and test runner.
- `Project implementation plan (2)/test_logs/` — archived local verification
  outputs; each mode's `summary.txt` lists the checks it includes.
- `rve_*/` — run-specific datasets, manifests, checkpoints, reports, and logs.
- `cluster_outputs/scratch4/` — additional cluster results. Fifty byte-identical
  copies of root `rve_*` folders were removed. Forty unique folders and one
  differing same-name folder remain here.
- `Final Report prep/reports/final_report/` — v13 LaTeX report source, including
  figures, figure-source data, bibliography, and build copy.
- `Final Report prep/PPP_final_report_Vignesh_Ravi_v13.pdf` — v13 report PDF.
- `code_patches/` — original code/test addition bundle and patch; canonical
  source files are in `Project implementation plan (2)/` and `.github/`.
- `extra_implementation_plan.md` — notes on additional implementation work.

The report source and submitted PDF are the v13 versions. The PDF is also copied
to `Final Report prep/reports/final_report/build/main.pdf` so both repository
copies match.

## Run the verification suite

Use Linux or WSL with the dependencies in
`Project implementation plan (2)/requirements-pinned.txt` installed. From the
repository root:

```bash
cd "Project implementation plan (2)"
bash run_tests.sh quick
```

Other modes are `full`, `audit`, `fe`, `rve`, `surrogate`, `extended`, and
`p2-v9`. See `tests/README.md` for the checks and expected results. The suite
uses archived datasets and checkpoints in the repository layout; the runner
supports `PPP_*` environment variables when those artifacts are stored
elsewhere. New logs are written under `Project implementation plan (2)/test_logs/`.

The GitHub Actions workflow is in `.github/workflows/tests.yml`.

## Archived results

Datasets and checkpoints are stored alongside run-specific manifests and
reports in the `rve_*` folders. Use the corresponding manifest and audit log
when interpreting a dataset. The archive contains both successful and
diagnostic runs, so check each run's report status before citing its results.
