"""Retry one failed RVE checkpoint using the solver's load continuation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src import free_particle, rve_dataset
from src.rve import make_periodic_particle_cell_mesh, rotation_matrix
from src.rve_dataset import DatasetConfig, particle_radius


def _continuation_retry_solver(
    mesh,
    macro_gradient,
    center=(0.5, 0.5, 0.5),
    initial_rotation=None,
    mu=1.0,
    bulk_modulus=1.0e4,
    displacement_order=2,
    rotation_tol=1.0e-9,
    max_rotation_iterations=8,
    integration_order=10,
    verbose=True,
):
    """Use the normal free-rotation solve, falling back on line-search failure."""
    macro_gradient = np.asarray(macro_gradient, dtype=float)
    center = np.asarray(center, dtype=float)
    if macro_gradient.shape != (3, 3) or np.linalg.det(macro_gradient) <= 0.0:
        raise ValueError("macro_gradient must be an orientation-preserving 3x3 matrix")
    if center.shape != (3,):
        raise ValueError("center must contain three coordinates")

    rotation = (
        np.eye(3)
        if initial_rotation is None
        else np.asarray(initial_rotation, dtype=float).copy()
    )
    if rotation.shape != (3, 3):
        raise ValueError("initial_rotation must have shape (3, 3)")

    history = []
    final_state = None
    previous_solution = None
    for iteration in range(max_rotation_iterations):
        particle_displacement = free_particle._particle_displacement(
            macro_gradient, rotation, center
        )
        try:
            result = free_particle.solve_periodic_rigid_particle(
                mesh,
                free_particle._matrix_cf(macro_gradient),
                particle_displacement=particle_displacement,
                mu=mu,
                bulk_modulus=bulk_modulus,
                displacement_order=displacement_order,
                maxit=200,
                integration_order=integration_order,
                verbose=False,
                initial_solution=previous_solution,
            )
        except FloatingPointError as exc:
            if "Non-finite Newton residual at iteration 0" not in str(exc):
                raise
            print(
                "[TARGETED RETRY] Non-finite initial residual; using load continuation.",
                flush=True,
            )
            result = free_particle._solve_periodic_with_load_continuation(
                mesh,
                macro_gradient,
                rotation,
                center,
                mu,
                bulk_modulus,
                displacement_order,
                integration_order,
            )
        except RuntimeError as exc:
            retryable = (
                "Newton line search could not find" in str(exc)
                or "Newton did not converge within maxit=" in str(exc)
            )
            if not retryable:
                raise
            print(
                "[TARGETED RETRY] Direct Newton solve failed; "
                "retrying this rotation with load continuation: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            result = free_particle._solve_periodic_with_load_continuation(
                mesh,
                macro_gradient,
                rotation,
                center,
                mu,
                bulk_modulus,
                displacement_order,
                integration_order,
            )

        solution, newton_history, deformation_gradient, stress = result
        previous_solution = solution
        gradient, hessian, linear_residuals = free_particle._rotational_newton_system(
            mesh,
            solution,
            macro_gradient,
            rotation,
            center,
            mu,
            bulk_modulus,
            integration_order,
            equilibrium_tol=1.0e-8,
        )
        residual = np.linalg.norm(gradient)
        history.append(residual)
        final_state = (
            solution,
            newton_history,
            deformation_gradient,
            stress,
            gradient,
            hessian,
            linear_residuals,
        )
        if verbose:
            print(f"Rotation Newton {iteration:2d}: |moment| = {residual:.3e}")
        if residual < rotation_tol:
            break
        increment = np.linalg.solve(hessian, -gradient)
        rotation = rotation_matrix(increment) @ rotation

    if final_state is None:
        raise RuntimeError("rotation solve did not execute")
    (
        solution,
        newton_history,
        deformation_gradient,
        stress,
        gradient,
        hessian,
        linear_residuals,
    ) = final_state
    return free_particle.FreeRotationResult(
        solution=solution,
        rotation=rotation,
        stress=stress,
        deformation_gradient=deformation_gradient,
        newton_history=tuple(newton_history),
        rotational_residuals=np.asarray(history),
        condensed_hessian=hessian,
        condensed_linear_residuals=linear_residuals,
    )


def _parse_sample_id(sample_id):
    try:
        volume_index = int(sample_id.split("_", 1)[0][1:])
        state_index = int(sample_id.rsplit("f", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid sample ID: {sample_id}") from exc
    return volume_index, state_index


def retry_checkpoint(work_directory, sample_id):
    work_directory = Path(work_directory).resolve()
    run_path = work_directory / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"run metadata not found: {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    config = DatasetConfig(**run["config"])
    config.validate()

    volume_index, state_index = _parse_sample_id(sample_id)
    if not 0 <= volume_index < len(config.volume_fractions):
        raise ValueError(f"volume index is outside the configured range: {sample_id}")
    if not 0 <= state_index < config.n_deformations:
        raise ValueError(f"state index is outside the configured range: {sample_id}")

    samples_directory = work_directory / "samples"
    checkpoint = samples_directory / f"{sample_id}.npz"
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            rve_dataset.validate_solved_sample(
                {name: saved[name] for name in saved.files}, config
            )
        print(f"Checkpoint already exists and is valid: {checkpoint}", flush=True)
        return checkpoint

    gradient = None
    for peer_volume_index in range(len(config.volume_fractions)):
        if peer_volume_index == volume_index:
            continue
        peer_id = f"v{peer_volume_index:03d}_f{state_index:06d}"
        peer_path = samples_directory / f"{peer_id}.npz"
        if not peer_path.is_file():
            continue
        with np.load(peer_path, allow_pickle=False) as peer:
            peer_gradient = np.asarray(peer["deformation_gradient"], dtype=float)
        if gradient is not None and not np.allclose(
            gradient, peer_gradient, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(f"peer checkpoints disagree on deformation state {state_index}")
        gradient = peer_gradient

    if gradient is None:
        raise FileNotFoundError(
            f"no completed peer checkpoint supplies the gradient for state {state_index}"
        )

    volume_fraction = float(config.volume_fractions[volume_index])
    print(
        f"Retrying {sample_id}: vf={volume_fraction:.4f}, "
        f"detF={np.linalg.det(gradient):.8f}, condF={np.linalg.cond(gradient):.6f}",
        flush=True,
    )

    mesh = make_periodic_particle_cell_mesh(
        radius=particle_radius(volume_fraction),
        maxh=config.maxh,
    )
    rve_dataset.solve_free_rotating_particle = _continuation_retry_solver
    sample = rve_dataset.solve_dataset_sample(
        mesh,
        gradient,
        volume_fraction,
        config,
    )
    rve_dataset.validate_solved_sample(sample, config)
    rve_dataset._atomic_savez(checkpoint, **sample)

    helper_path = Path(__file__).resolve()
    helper_sha256 = hashlib.sha256(helper_path.read_bytes()).hexdigest()
    provenance = {
        "sample_id": sample_id,
        "method": "existing adaptive load-continuation path after direct Newton line-search failure",
        "source_revision_sha256": run.get("source_revision_sha256"),
        "retry_helper_sha256": helper_sha256,
        "determinant": float(np.linalg.det(gradient)),
        "condition_number": float(np.linalg.cond(gradient)),
        "mixed_residual": float(sample["mixed_residual"]),
        "rotation_residual": float(sample["rotation_residual"]),
        "condensed_residual": float(sample["condensed_residual"]),
        "minimum_microscopic_jacobian": float(
            sample["minimum_microscopic_jacobian"]
        ),
    }
    provenance_path = work_directory.parent / f"{sample_id}.retry_provenance.json"
    rve_dataset._atomic_json(provenance_path, provenance)
    print(f"Checkpoint saved: {checkpoint}", flush=True)
    print(f"Retry provenance saved: {provenance_path}", flush=True)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    args = parser.parse_args()
    retry_checkpoint(args.work_directory, args.sample_id)


if __name__ == "__main__":
    main()
