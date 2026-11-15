"""Command-line entry point for deterministic Phase-2 RVE data generation."""
import argparse

from .rve_dataset import DatasetConfig, generate_rve_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/rve_phase2.npz")
    parser.add_argument("--work-directory")
    parser.add_argument("--volume-fractions", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    parser.add_argument("--n-deformations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-strain", type=float, default=0.20)
    parser.add_argument("--determinant-floor", type=float, default=0.50)
    parser.add_argument("--max-condition-number", type=float, default=3.0)
    parser.add_argument("--maxh", type=float, default=0.22)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--bulk-modulus", type=float, default=1.0e4)
    parser.add_argument(
        "--microscopic-jacobian-floor",
        type=float,
        default=1.0e-8,
        help="reject samples whose minimum quadrature-point microscopic J is at or below this value",
    )
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()

    config = DatasetConfig(
        volume_fractions=tuple(arguments.volume_fractions),
        n_deformations=arguments.n_deformations,
        seed=arguments.seed,
        max_strain=arguments.max_strain,
        determinant_floor=arguments.determinant_floor,
        max_condition_number=arguments.max_condition_number,
        maxh=arguments.maxh,
        mu=arguments.mu,
        bulk_modulus=arguments.bulk_modulus,
        microscopic_jacobian_floor=arguments.microscopic_jacobian_floor,
    )
    _, manifest = generate_rve_dataset(
        config,
        output=arguments.output,
        work_directory=arguments.work_directory,
        resume=not arguments.no_resume,
    )
    print(
        f"P2 dataset complete: {manifest['n_samples']} samples -> "
        f"{arguments.output}"
    )


if __name__ == "__main__":
    main()
