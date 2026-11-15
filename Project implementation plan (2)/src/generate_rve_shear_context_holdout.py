"""Generate direct-RVE simple-shear states for surrogate evaluation."""

import argparse
from pathlib import Path

import numpy as np

from .generate_rve_dataset_parallel import generate_parallel_dataset
from .rve_dataset import DatasetConfig


DEFAULT_HOLDOUT_AMPLITUDES = (0.035, 0.065, 0.085, 0.10)
PREVIOUSLY_USED_AMPLITUDES = {
    0.01, 0.02, 0.025, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07,
    0.075, 0.08, 0.09, 0.095,
}


def _states(amplitudes):
    gradients = [np.eye(3, dtype=float)]
    gradients.extend(
        np.array([[1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for gamma in amplitudes
    )
    return np.asarray(gradients, dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--adaptation-seed", type=int, default=20261115)
    parser.add_argument("--maxh", type=float, default=0.22)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--bulk-modulus", type=float, default=1.0e4)
    parser.add_argument("--displacement-order", type=int, default=2)
    parser.add_argument("--integration-order", type=int, default=10)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--amplitudes", type=float, nargs="+", default=DEFAULT_HOLDOUT_AMPLITUDES
    )
    parser.add_argument(
        "--exclude-amplitudes", type=float, nargs="*", default=(),
        help="additional amplitudes to reject because they were used in prior studies",
    )
    args = parser.parse_args()

    amplitudes = tuple(args.amplitudes)
    if len(set(amplitudes)) != len(amplitudes) or any(a <= 0.0 for a in amplitudes):
        raise ValueError("amplitudes must be unique positive values")
    excluded_amplitudes = PREVIOUSLY_USED_AMPLITUDES.union(args.exclude_amplitudes)
    overlap = excluded_amplitudes.intersection(amplitudes)
    if overlap:
        raise ValueError(f"holdout amplitudes overlap prior data: {sorted(overlap)}")

    gradients = _states(amplitudes)
    config = DatasetConfig(
        volume_fractions=(0.02, 0.05, 0.10),
        n_deformations=len(gradients),
        seed=args.seed,
        split_fractions=(0.70, 0.15, 0.15),
        max_strain=max(0.10, max(amplitudes)),
        determinant_floor=0.50,
        max_condition_number=3.0,
        maxh=args.maxh,
        mu=args.mu,
        bulk_modulus=args.bulk_modulus,
        displacement_order=args.displacement_order,
        integration_order=args.integration_order,
    )
    splits = np.full(len(gradients), "test", dtype="<U10")
    design = {
        "load_mode": "positive simple shear, F[0,1]=gamma",
        "holdout_amplitudes": list(amplitudes),
        "excluded_previous_amplitudes": sorted(excluded_amplitudes),
        "includes_identity_reference": True,
        "microstructure_seed": args.seed,
        "matches_adaptation_dataset_microstructure_seed": (
            args.seed == args.adaptation_seed
        ),
        "evaluation_only": True,
        "purpose": (
            "new independent final direct-RVE evaluation of a shear surrogate; "
            "all states are held out from fitting and model selection"
        ),
        "split_unit": "deformation state; all states assigned to test",
    }
    output = generate_parallel_dataset(
        config=config,
        output=args.output,
        work_directory=args.work_directory,
        workers=args.workers,
        resume=True,
        deformation_states=gradients,
        split_assignments=splits,
        deformation_design=design,
        allow_empty_splits=True,
    )
    manifest_path = output.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise SystemExit(
            "Shear holdout generation is incomplete; inspect the .failures.json "
            "report and resume with the same paths and settings."
        )
    print(f"Independent shear holdout: {output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
