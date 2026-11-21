#!/usr/bin/env bash
# Run the PPP verification suite (Chapter 7 of the report) on Ubuntu or on the cluster.
#
#   ./run_tests.sh quick      # all short checks, run sequentially       (~5 min)
#   ./run_tests.sh full       # quick + extended checks                  (~15 min)
#   ./run_tests.sh audit      # full + P2-V9                             (~1-1.5 h)
#   ./run_tests.sh fe         # V0-V8 only
#   ./run_tests.sh rve        # P2-V1..V5 and P2-V10 only
#   ./run_tests.sh surrogate # S-V1 only
#   ./run_tests.sh extended   # P2-V6..V8 and M-V0 only
#   ./run_tests.sh p2-v9      # long RVE constitutive audit only
#
# Run from the folder that contains src/, tests/ and benchmarks/.
# Paths to archived artifacts can be overridden by environment variables (defaults
# follow the GitHub repository layout, one level above this folder):
#   PPP_PHASE1_CKPT, PPP_RVE_CKPT, PPP_PHASE7_NPZ, PPP_PHASE7_MANIFEST,
#   PPP_TRAIN_NPZ, PPP_PILOT_NPZ, PPP_LOG_DIR
set -uo pipefail
MODE="${1:-quick}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
REPO="$(cd "$ROOT/.." && pwd)"

RUN_FE=0 RUN_RVE=0 RUN_SURROGATE=0 RUN_EXTENDED=0 RUN_P2_V9=0
case "$MODE" in
  quick) RUN_FE=1 RUN_RVE=1 RUN_SURROGATE=1 ;;
  full) RUN_FE=1 RUN_RVE=1 RUN_SURROGATE=1 RUN_EXTENDED=1 ;;
  audit) RUN_FE=1 RUN_RVE=1 RUN_SURROGATE=1 RUN_EXTENDED=1 RUN_P2_V9=1 ;;
  fe) RUN_FE=1 ;;
  rve) RUN_RVE=1 ;;
  surrogate) RUN_SURROGATE=1 ;;
  extended) RUN_EXTENDED=1 ;;
  p2-v9) RUN_P2_V9=1 ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Use quick|full|audit|fe|rve|surrogate|extended|p2-v9" >&2
    exit 2 ;;
esac

PHASE1_CKPT="${PPP_PHASE1_CKPT:-checkpoints/phase1_validated.pt}"
RVE_CKPT="${PPP_RVE_CKPT:-$REPO/rve_phase7_augmented_finetune12/training/partial_icnn_phase6_phase7_finetuned.pt}"
PHASE7_NPZ="${PPP_PHASE7_NPZ:-$REPO/rve_phase7_independent_20261107/data/rve_phase7_independent.npz}"
PHASE7_MANIFEST="${PPP_PHASE7_MANIFEST:-$REPO/rve_phase7_independent_20261107/data/rve_phase7_independent.manifest.json}"
TRAIN_NPZ="${PPP_TRAIN_NPZ:-$REPO/rve_phase7_augmented_training_20261109/data/rve_phase6_phase7_training.npz}"
PILOT_NPZ="${PPP_PILOT_NPZ:-$REPO/rve_phase2_solver_revision_oR3pd5/data/rve_phase2.npz}"
LOG_DIR="${PPP_LOG_DIR:-test_logs/$(date +%Y%m%d_%H%M%S)_$MODE}"
mkdir -p "$LOG_DIR"

export NGS_NUM_THREADS="${NGS_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

declare -a RESULTS=()
FAILED=0

run() {  # run <ID> <log name> <command...>
  local id="$1" name="$2"; shift 2
  local log="$LOG_DIR/$name.log" t0=$SECONDS
  printf '%-7s %-38s ... ' "$id" "$name"
  if "$@" > "$log" 2>&1; then
    printf 'PASS  (%ds)\n' $((SECONDS - t0)); RESULTS+=("$id PASS $name")
  else
    printf 'FAIL  (%ds)  -> %s\n' $((SECONDS - t0)) "$log"; RESULTS+=("$id FAIL $name"); FAILED=1
  fi
}

need() { [ -f "$1" ] || { echo "missing file: $1 (set the corresponding PPP_* variable)"; exit 2; }; }
if [ "$RUN_FE" = 1 ]; then need "$PHASE1_CKPT"; need "$RVE_CKPT"; fi
if [ "$RUN_SURROGATE" = 1 ] || [ "$RUN_EXTENDED" = 1 ]; then need "$RVE_CKPT"; fi
if [ "$RUN_SURROGATE" = 1 ]; then
  need "$PHASE7_NPZ"; need "$PHASE7_MANIFEST"; need "$TRAIN_NPZ"
fi

echo "PPP verification suite, mode=$MODE, logs in $LOG_DIR"
python -c "import sys, torch, ngsolve, numpy; print('python', sys.version.split()[0], '| torch', torch.__version__, '| ngsolve', ngsolve.__version__, '| numpy', numpy.__version__)" | tee "$LOG_DIR/environment.txt"

if [ "$RUN_FE" = 1 ]; then
  # ---- finite-element basics and Phase 1 bridge ------------------------------
  run V0     verify_fe_basics         python -m tests.verify_fe_basics --ckpt "$PHASE1_CKPT" --rve-ckpt "$RVE_CKPT"
  run V1     verify_model_interface   python -m tests.verify_model_interface
  run V2     verify_reference         python -m tests.verify_reference
  run V2b    verify_neohooke_response python -m tests.verify_neohooke_response --ckpt "$PHASE1_CKPT"
  run V3     verify_tangent           python -m tests.verify_tangent
  run V4     verify_element           python -m tests.verify_element --ckpt "$PHASE1_CKPT"
  run V5     verify_patch             python -m tests.verify_patch --ckpt "$PHASE1_CKPT"
  run V6     verify_fe_comparison     python -m tests.verify_fe_comparison --ckpt "$PHASE1_CKPT"
  run V7/V8  verify_stress_control    python -m tests.verify_stress_control --ckpt "$PHASE1_CKPT"
fi

if [ "$RUN_RVE" = 1 ]; then
  # ---- Phase 2 RVE ------------------------------------------------------------
  run P2-V1  verify_periodic_rve      python -m tests.verify_periodic_rve
  run P2-V2  verify_rigid_particle    python -m tests.verify_rigid_particle
  run P2-V3  verify_mixed_matrix      python -m tests.verify_mixed_matrix
  run P2-V4  verify_coupled_rve       python -m tests.verify_coupled_rve
  run P2-V5  verify_hill_mandel       python -m tests.verify_hill_mandel
  run P2-V10 verify_rve_dataset       python -m tests.verify_rve_dataset
fi

if [ "$RUN_SURROGATE" = 1 ]; then
  # ---- surrogate --------------------------------------------------------------
  run S-V1 surrogate_constitutive_checks python -m benchmarks.rve_surrogate_constitutive_checks \
      --dataset "$PHASE7_NPZ" --manifest "$PHASE7_MANIFEST" --checkpoint "$RVE_CKPT" \
      --training-dataset "$TRAIN_NPZ" --report "$LOG_DIR/s_v1_constitutive_checks.json" \
      --states-per-det-bin 2 --fd-step 1e-5 --torch-threads "$TORCH_NUM_THREADS"
fi

if [ "$RUN_EXTENDED" = 1 ]; then
  run P2-V6 verify_homogenized_tangent python -m tests.verify_homogenized_tangent
  run P2-V7 verify_free_rotation       python -m tests.verify_free_rotation
  run P2-V8 verify_translation_gauge   python -m tests.verify_translation_gauge
  run M-V0  macro_tangent_check        python -m benchmarks.phase6_macro_fe_demo \
      --checkpoint "$RVE_CKPT" --output "$LOG_DIR/m_v0_report.json" --volume-fraction 0.05 \
      --stretch 0.01 --maxh 0.25 --order 1 --integration-order 4 --load-steps 2 --max-iter 50 \
      --merit-function energy --load-predictor secant \
      --tangent-check-at-load-step 2 --tangent-check-iteration 0 --tangent-check-only
fi

if [ "$RUN_P2_V9" = 1 ]; then
  need "$PILOT_NPZ"
  run P2-V9 rve_constitutive_checks python -m benchmarks.rve_constitutive_checks \
      --dataset "$PILOT_NPZ" --output "$LOG_DIR/p2_v9_rve_constitutive_checks.json" \
      --sample-ids v001_f000047
fi

echo; echo "Summary ($MODE):"; printf '  %s\n' "${RESULTS[@]}" | tee "$LOG_DIR/summary.txt"
exit $FAILED
