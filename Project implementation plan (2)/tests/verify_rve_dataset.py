"""P2-V10 -- Verify deterministic RVE dataset sampling and serialization.

Run:
    python -m tests.verify_rve_dataset

This contract test does not launch a production RVE campaign.
"""
from dataclasses import replace
import tempfile
from pathlib import Path

import numpy as np

from src.rve_dataset import (
    DatasetConfig,
    _atomic_savez,
    assign_deformation_splits,
    generate_deformation_states,
    particle_radius,
)


def main():
    config = DatasetConfig(
        volume_fractions=(0.02, 0.08),
        n_deformations=12,
        seed=314159,
        max_strain=0.15,
    )
    states_a = generate_deformation_states(config)
    states_b = generate_deformation_states(config)
    splits_a = assign_deformation_splits(
        config.n_deformations, config.seed, config.split_fractions
    )
    splits_b = assign_deformation_splits(
        config.n_deformations, config.seed, config.split_fractions
    )

    deterministic_error = np.max(np.abs(states_a - states_b))
    determinants = np.linalg.det(states_a)
    condition_numbers = np.linalg.cond(states_a)
    split_counts = {
        name: int(np.count_nonzero(splits_a == name))
        for name in ("train", "validation", "test")
    }
    sample_splits = np.repeat(splits_a[None, :], len(config.volume_fractions), axis=0)
    leakage_count = sum(
        len(np.unique(sample_splits[:, state_index])) != 1
        for state_index in range(config.n_deformations)
    )
    radii = np.array([particle_radius(value) for value in config.volume_fractions])

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "sample.npz"
        _atomic_savez(
            output,
            deformation_gradient=states_a[0],
            tangent=np.eye(9),
            split=np.asarray(splits_a[0]),
        )
        with np.load(output, allow_pickle=False) as saved:
            serialization_error = np.max(
                np.abs(saved["deformation_gradient"] - states_a[0])
            )
            tangent_shape = saved["tangent"].shape

    changed = replace(config, seed=config.seed + 1)
    different_seed_change = np.max(
        np.abs(generate_deformation_states(changed) - states_a)
    )

    print(f"\ndeformation states:                {len(states_a)}")
    print(f"volume fractions:                 {config.volume_fractions}")
    print(f"particle radii:                   {radii}")
    print(f"deterministic regeneration error: {deterministic_error:.3e}")
    print(f"minimum determinant:              {determinants.min():.6f}")
    print(f"maximum condition number:         {condition_numbers.max():.6f}")
    print(f"split counts:                     {split_counts}")
    print(f"cross-volume split leaks:         {leakage_count}")
    print(f"different-seed state change:      {different_seed_change:.3e}")
    print(f"serialization error:              {serialization_error:.3e}")
    print(f"serialized tangent shape:         {tangent_shape}")

    assert np.array_equal(states_a[0], np.eye(3)), "reference state is missing"
    assert deterministic_error == 0.0, "sampling is not deterministic"
    assert determinants.min() > config.determinant_floor, "inadmissible determinant"
    assert condition_numbers.max() <= config.max_condition_number, (
        "inadmissible condition number"
    )
    assert all(count > 0 for count in split_counts.values()), "empty data split"
    assert np.array_equal(splits_a, splits_b), "split assignment is not deterministic"
    assert leakage_count == 0, "deformation group leaks across data splits"
    assert different_seed_change > 1.0e-3, "seed does not alter sampled states"
    assert np.all((radii > 0) & (radii < 0.5)), "invalid particle radius"
    assert serialization_error == 0.0, "NPZ serialization changed data"
    assert tangent_shape == (9, 9), "tangent serialization has wrong shape"
    print("P2-V10 PASS: deterministic dataset contract and splits are valid.")


if __name__ == "__main__":
    main()
