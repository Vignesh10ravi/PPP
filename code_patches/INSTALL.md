# New and corrected code — how to install and run

`files/` mirrors your GitHub repository (`Vignesh10ravi/PPP`):

```
files/
  .github/workflows/tests.yml              -> repository root (CI on GitHub)
  Project implementation plan (2)/
    run_tests.sh, run_tests.pbs            -> new: one-command test runner (Ubuntu / PBS)
    requirements-pinned.txt                -> new: exact versions of the cluster runs
    tests/verify_fe_basics.py              -> new: V0a-V0e (element basics, rigid modes, hourglass)
    tests/verify_stress_control.py         -> new: V7 (full deformation control), V8 (stress control)
    tests/README.md                        -> new: command / inputs / expected result per test
    tests/verify_element.py                -> fixed: no hard-coded path, --ckpt option, Newton table
    src/ngsolve_umat.py                    -> assemble_system(), timers, tractions + component BCs
    src/picnn.py, src/train.py             -> fixed: legacy Phase 1 checkpoints load again
    benchmarks/profile_bridge.py           -> new: computation-time study
```

## Option A: copy the files (simplest)

From the folder that contains this `INSTALL.md`:

```bash
REPO=~/PPP_repo            # your clone of github.com/Vignesh10ravi/PPP
cp -r "files/." "$REPO/"
```

## Option B: apply the patch

```bash
cd "$REPO/Project implementation plan (2)"
patch -p1 --dry-run < /path/to/report_preparation.patch   # check first
patch -p1 < /path/to/report_preparation.patch
mkdir -p ../.github/workflows && cp /path/to/files/.github/workflows/tests.yml ../.github/workflows/
```

## Run on Ubuntu (your computer / WSL)

```bash
cd "$REPO/Project implementation plan (2)"
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-pinned.txt
bash run_tests.sh quick      # ~5 min: V0-V8, P2-V1..V5, P2-V10, S-V1
bash run_tests.sh full       # ~15 min: + P2-V6..V8, M-V0
bash run_tests.sh audit      # ~1-1.5 h: + P2-V9
```

The runner expects the archived artifacts in the repository layout
(`../rve_phase7_augmented_finetune12/...` etc.). If they are elsewhere, set e.g.

```bash
export PPP_RVE_CKPT=/path/to/partial_icnn_phase6_phase7_finetuned.pt
export PPP_PHASE7_NPZ=/path/to/rve_phase7_independent.npz
export PPP_PHASE7_MANIFEST=/path/to/rve_phase7_independent.manifest.json
export PPP_TRAIN_NPZ=/path/to/rve_phase6_phase7_training.npz
export PPP_PILOT_NPZ=/path/to/rve_phase2.npz
```

Logs and `summary.txt` go to `test_logs/<date>_<mode>/`; the exit code is 0 only if all tests pass.

## Run on the TU Freiberg cluster

```bash
# from your machine
scp -r "files/Project implementation plan (2)/." "$CLUSTER_USER@$CLUSTER_HOST:~/PPP/Project implementation plan (2)/"
# on the login node
cd "$HOME/PPP/Project implementation plan (2)"
MODE=full  qsub -v MODE run_tests.pbs     # or MODE=audit for P2-V9 as well
qstat -u "$USER"
cat /scratch/$USER/ppp_tests_full_*/summary.txt
```

`run_tests.pbs` uses your `ppp-ngsolve` venv, the `teachingq` queue and the
`/scratch4/vr80beqo/...` artifact paths of the production runs; adjust them if needed.

## Continuous integration

After committing `.github/workflows/tests.yml` and pushing, GitHub runs the quick suite on
every push (Actions tab). The checkpoints used by the tests must be in the repository (they are).

All of the above was executed in an independent Linux environment
(Python 3.13, NGSolve 6.2.2608, PyTorch 2.15): quick and full modes pass; P2-V9 passed in 59.6 min.
