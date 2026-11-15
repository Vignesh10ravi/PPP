"""Generate grouped direct-RVE simple-shear data for a separate fine-tune.

Train, model-selection validation, and final-test amplitudes are disjoint and
each deformation state is repeated across all requested volume fractions.
Previously reported 2.5%, 5%, and 7.5% shear points are excluded from every
split and remain baseline evidence only.
"""

import argparse
from pathlib import Path

import numpy as np

from .generate_rve_dataset_parallel import generate_parallel_dataset
from .rve_dataset import DatasetConfig


DEFAULT_AMPLITUDES = {
    "train": (0.01, 0.02, 0.04, 0.06, 0.08, 0.09),
    "validation": (0.03, 0.07),
    "test": (0.045, 0.055, 0.095),
}
BASELINE_AMPLITUDES = (0.025, 0.05, 0.075)


def _states_and_splits(amplitudes_by_split):
    gradients = []
    split_labels = []
    flattened = []
    for split_name in ("train", "validation", "test"):
        values = tuple(float(value) for value in amplitudes_by_split[split_name])
        if not values:
            raise ValueError(f"{split_name} amplitudes cannot be empty")
        if any(not np.isfinite(value) or value <= 0.0 or value > 0.10 for value in values):
            raise ValueError("simple-shear amplitudes must lie in (0, 0.10]")
        flattened.extend(values)
        for amplitude in values:
            gradient = np.eye(3, dtype=float)
            gradient[0, 1] = amplitude
            gradients.append(gradient)
            split_labels.append(split_name)

    rounded = np.round(np.asarray(flattened, dtype=float), decimals=12)
    if len(np.unique(rounded)) != len(rounded):
        raise ValueError("shear amplitudes must be unique across splits")
    if any(
        np.isclose(value, baseline, rtol=0.0, atol=1.0e-12)
        for value in flattened
        for baseline in BASELINE_AMPLITUDES
    ):
        raise ValueError(
            "the existing 2.5%, 5%, and 7.5% baseline states must stay out "
            "of the fine-tuning splits"
        )
    return np.asarray(gradients), np.asarray(split_labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument(
        "--train-amplitudes", type=float, nargs="+", default=DEFAULT_AMPLITUDES["train"]
    )
    parser.add_argument(
        "--validation-amplitudes", type=float, nargs="+",
        default=DEFAULT_AMPLITUDES["validation"],
    )
    parser.add_argument(
        "--test-amplitudes", type=float, nargs="+", default=DEFAULT_AMPLITUDES["test"]
    )
    parser.add_argument(
        "--volume-fractions", type=float, nargs="+", default=(0.02, 0.05, 0.10)
    )
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--maxh", type=float, default=0.22)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--bulk-modulus", type=float, default=1.0e4)
    parser.add_argument("--displacement-order", type=int, default=2)
    parser.add_argument("--integration-order", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    amplitudes_by_split = {
        "train": args.train_amplitudes,
        "validation": args.validation_amplitudes,
        "test": args.test_amplitudes,
    }
    gradients, split_labels = _states_and_splits(amplitudes_by_split)
    config = DatasetConfig(
        volume_fractions=tuple(args.volume_fractions),
        n_deformations=len(gradients),
        seed=args.seed,
        split_fractions=(0.55, 0.18, 0.27),
        max_strain=0.10,
        determinant_floor=0.50,
        max_condition_number=3.0,
        maxh=args.maxh,
        mu=args.mu,
        bulk_modulus=args.bulk_modulus,
        displacement_order=args.displacement_order,
        integration_order=args.integration_order,
    )
    design = {
        "load_mode": "positive simple shear, F[0,1]=gamma",
        "amplitudes_by_split": {
            key: [float(value) for value in values]
            for key, values in amplitudes_by_split.items()
        },
        "excluded_existing_baseline_amplitudes": list(BASELINE_AMPLITUDES),
        "split_unit": "deformation state; each state is repeated at every volume fraction",
        "purpose": "separate shear-aware fine-tuning; final-test states are excluded from fitting",
    }
    output = generate_parallel_dataset(
        config=config,
        output=args.output,
        work_directory=args.work_directory,
        workers=args.workers,
        resume=True,
        deformation_states=gradients,
        split_assignments=split_labels,
        deformation_design=design,
    )
    manifest_path = output.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise SystemExit(
            "Shear dataset is incomplete; inspect the .failures.json report and "
            "resume with the same paths and amplitudes."
        )
    print(f"Shear adaptation data: {output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
