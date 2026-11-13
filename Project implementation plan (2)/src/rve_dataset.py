"""Deterministic free-rotation RVE dataset generation.

Each deformation state belongs to exactly one split and is evaluated at every
requested particle volume fraction. Samples are checkpointed individually
before the final compressed NPZ is assembled, so interrupted runs can resume
without repeating completed RVE solves.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import ngsolve
import numpy as np
from ngsolve import Det, GridFunction, InnerProduct, Integrate
from ngsolve.comp import IntegrationRuleSpace

from .free_particle import solve_free_rotating_particle
from .rve import make_periodic_particle_cell_mesh
from .rve_tangent import condensed_free_rotation_tangent


SCHEMA_VERSION = 2
SPLIT_NAMES = ("train", "validation", "test")
TANGENT_FLATTENING = "row-major: (i,J) -> 3*i + J"
_DATASET_SOURCE_FILES = (
    "generate_rve_dataset.py",
    "generate_rve_dataset_parallel.py",
    "rve_dataset.py",
    "free_particle.py",
    "rve_solver.py",
    "rve_tangent.py",
    "rve.py",
    "mixed_matrix.py",
    "homogenization.py",
)


@dataclass(frozen=True)
class DatasetConfig:
    volume_fractions: tuple = (0.02, 0.05, 0.10)
    n_deformations: int = 100
    n_compression_deformations: int = 0
    compression_determinant_range: tuple = (0.52, 0.70)
    n_expansion_deformations: int = 0
    expansion_determinant_range: tuple = (1.50, 1.80)
    seed: int = 1729
    max_strain: float = 0.20
    determinant_floor: float = 0.50
    max_condition_number: float = 3.0
    split_fractions: tuple = (0.70, 0.15, 0.15)
    maxh: float = 0.22
    mu: float = 1.0
    bulk_modulus: float = 1.0e4
    displacement_order: int = 2
    integration_order: int = 10
    microscopic_jacobian_floor: float = 1.0e-8

    def validate(self):
        volume_fractions = np.asarray(self.volume_fractions, dtype=float)
        split_fractions = np.asarray(self.split_fractions, dtype=float)
        compression_range = np.asarray(
            self.compression_determinant_range, dtype=float
        )
        expansion_range = np.asarray(
            self.expansion_determinant_range, dtype=float
        )
        if self.n_deformations < 3:
            raise ValueError("n_deformations must be at least three")
        if not isinstance(self.n_compression_deformations, (int, np.integer)) or self.n_compression_deformations < 0:
            raise ValueError(
                "n_compression_deformations must be a nonnegative integer"
            )
        if not isinstance(self.n_expansion_deformations, (int, np.integer)) or self.n_expansion_deformations < 0:
            raise ValueError(
                "n_expansion_deformations must be a nonnegative integer"
            )
        if self.n_compression_deformations + self.n_expansion_deformations >= self.n_deformations:
            raise ValueError(
                "the targeted compression and expansion counts must sum to "
                "less than n_deformations"
            )
        if (
            compression_range.shape != (2,)
            or not np.all(np.isfinite(compression_range))
            or compression_range[0] <= 0.0
            or compression_range[1] <= compression_range[0]
        ):
            raise ValueError(
                "compression_determinant_range must be a finite increasing pair"
            )
        if (
            self.n_compression_deformations > 0
            and compression_range[0] <= self.determinant_floor
        ):
            raise ValueError(
                "the lower compression determinant must exceed determinant_floor"
            )
        if (
            expansion_range.shape != (2,)
            or not np.all(np.isfinite(expansion_range))
            or expansion_range[0] <= 1.0
            or expansion_range[1] <= expansion_range[0]
        ):
            raise ValueError(
                "expansion_determinant_range must be a finite increasing pair "
                "above one"
            )
        if (
            volume_fractions.ndim != 1
            or len(volume_fractions) == 0
            or np.any(volume_fractions <= 0)
            or np.any(volume_fractions >= np.pi / 6.0)
        ):
            raise ValueError("volume fractions must lie in (0, pi/6)")
        if len(np.unique(volume_fractions)) != len(volume_fractions):
            raise ValueError("volume fractions must be unique")
        if (
            split_fractions.shape != (3,)
            or np.any(split_fractions <= 0)
            or not np.isclose(split_fractions.sum(), 1.0)
        ):
            raise ValueError("split_fractions must be three positive values summing to one")
        if self.max_strain <= 0:
            raise ValueError("max_strain must be positive")
        if self.determinant_floor <= 0:
            raise ValueError("determinant_floor must be positive")
        if self.max_condition_number <= 1:
            raise ValueError("max_condition_number must exceed one")
        if self.maxh <= 0 or self.mu <= 0 or self.bulk_modulus <= 0:
            raise ValueError("mesh size and material moduli must be positive")
        if self.displacement_order < 2 or self.integration_order < 1:
            raise ValueError("invalid finite-element or integration order")
        if (
            not np.isfinite(self.microscopic_jacobian_floor)
            or self.microscopic_jacobian_floor <= 0
        ):
            raise ValueError("microscopic_jacobian_floor must be finite and positive")


def particle_radius(volume_fraction):
    """Radius of one sphere in a unit-volume cell."""
    volume_fraction = float(volume_fraction)
    if not 0 < volume_fraction < np.pi / 6.0:
        raise ValueError("volume_fraction must lie in (0, pi/6)")
    return (3.0 * volume_fraction / (4.0 * np.pi)) ** (1.0 / 3.0)


def _generate_targeted_deformation_states(generator, count, determinant_range, config):
    """Generate distorted states with determinants stratified across a range."""
    if count == 0:
        return []

    determinant_low, determinant_high = determinant_range
    target_edges = np.linspace(determinant_low, determinant_high, count + 1)
    target_determinants = generator.uniform(target_edges[:-1], target_edges[1:])
    generator.shuffle(target_determinants)
    states = []
    for target_determinant in target_determinants:
        for _ in range(1000):
            base_gradient = np.eye(3) + generator.uniform(
                -config.max_strain,
                config.max_strain,
                size=(3, 3),
            )
            base_determinant = np.linalg.det(base_gradient)
            if (
                not np.isfinite(base_determinant)
                or base_determinant <= 0.0
                or np.linalg.cond(base_gradient) > config.max_condition_number
            ):
                continue

            gradient = base_gradient * (
                target_determinant / base_determinant
            ) ** (1.0 / 3.0)
            determinant = np.linalg.det(gradient)
            if (
                not np.all(np.isfinite(gradient))
                or determinant <= config.determinant_floor
                or np.linalg.cond(gradient) > config.max_condition_number
            ):
                continue
            states.append(gradient)
            break
        else:
            raise RuntimeError(
                "could not generate an admissible deformation at target "
                f"determinant {target_determinant:.6f}"
            )
    return states


def generate_deformation_states(config):
    """Return deterministic gradients with optional targeted tails in det(F)."""
    config.validate()
    generator = np.random.default_rng(config.seed)
    states = [np.eye(3)]
    attempts = 0
    n_standard = (
        config.n_deformations
        - config.n_compression_deformations
        - config.n_expansion_deformations
    )
    max_attempts = 1000 * config.n_deformations
    while len(states) < n_standard and attempts < max_attempts:
        attempts += 1
        gradient = np.eye(3) + generator.uniform(
            -config.max_strain,
            config.max_strain,
            size=(3, 3),
        )
        determinant = np.linalg.det(gradient)
        if determinant <= config.determinant_floor:
            continue
        if np.linalg.cond(gradient) > config.max_condition_number:
            continue
        states.append(gradient)

    states.extend(
        _generate_targeted_deformation_states(
            generator,
            config.n_compression_deformations,
            config.compression_determinant_range,
            config,
        )
    )
    states.extend(
        _generate_targeted_deformation_states(
            generator,
            config.n_expansion_deformations,
            config.expansion_determinant_range,
            config,
        )
    )

    if len(states) != config.n_deformations:
        raise RuntimeError("could not generate enough admissible deformation states")
    return np.stack(states)


def assign_deformation_splits(n_deformations, seed, fractions=(0.70, 0.15, 0.15)):
    """Assign whole deformation groups to deterministic leakage-free splits."""
    if n_deformations < 3:
        raise ValueError("at least three deformation groups are required")
    fractions = np.asarray(fractions, dtype=float)
    if (
        fractions.shape != (3,)
        or np.any(fractions <= 0)
        or not np.isclose(fractions.sum(), 1.0)
    ):
        raise ValueError("fractions must be three positive values summing to one")

    counts = np.ones(3, dtype=int)
    remaining = n_deformations - 3
    ideal = fractions * remaining
    counts += np.floor(ideal).astype(int)
    leftover = int(remaining - np.floor(ideal).sum())
    for index in np.argsort(-(ideal - np.floor(ideal)))[:leftover]:
        counts[index] += 1

    labels = np.concatenate([
        np.full(count, name, dtype="<U10")
        for name, count in zip(SPLIT_NAMES, counts)
    ])
    permutation = np.random.default_rng(seed + 1).permutation(n_deformations)
    assignments = np.empty(n_deformations, dtype="<U10")
    assignments[permutation] = labels
    return assignments


def _polar_rotation(deformation_gradient):
    left, _, right_transpose = np.linalg.svd(deformation_gradient)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transpose
    return rotation


def _homogenized_energy(
    mesh,
    deformation_gradient,
    pressure,
    mu,
    bulk_modulus,
    integration_order,
):
    jacobian = Det(deformation_gradient)
    first_invariant = InnerProduct(deformation_gradient, deformation_gradient)
    density = (
        0.5 * mu * (jacobian ** (-2.0 / 3.0) * first_invariant - 3.0)
        + pressure * (jacobian - 1.0)
        - pressure ** 2 / (2.0 * bulk_modulus)
    )
    return float(Integrate(density, mesh, order=integration_order))


def microscopic_jacobian_extrema(mesh, deformation_gradient, integration_order):
    """Sample the final microscopic Jacobian at the configured quadrature points."""
    quadrature_space = IntegrationRuleSpace(mesh, order=integration_order)
    sampled_jacobian = GridFunction(quadrature_space)
    sampled_jacobian.Interpolate(Det(deformation_gradient))
    values = np.asarray(sampled_jacobian.vec.FV().NumPy(), dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise FloatingPointError(
            "microscopic Jacobian quadrature samples are empty or nonfinite"
        )
    return float(values.min()), float(values.max())


def validate_solved_sample(sample, config):
    """Reject incomplete, inadmissible, or unconverged sample records."""
    required = {
        "minimum_microscopic_jacobian",
        "maximum_microscopic_jacobian",
        "mixed_iterations",
        "rotation_iterations",
        "minimum_particle_clearance",
        "solver_status",
        "mixed_residual",
        "rotation_residual",
        "condensed_residual",
    }
    missing = required.difference(sample)
    if missing:
        raise ValueError(f"sample is missing required fields: {sorted(missing)}")

    if str(np.asarray(sample["solver_status"]).item()) != "converged":
        raise RuntimeError("solver_status is not converged")
    for name, value in sample.items():
        if name == "solver_status":
            continue
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"sample field {name} is nonfinite")

    minimum_j = float(sample["minimum_microscopic_jacobian"])
    maximum_j = float(sample["maximum_microscopic_jacobian"])
    if minimum_j <= config.microscopic_jacobian_floor:
        raise RuntimeError(
            "minimum microscopic Jacobian "
            f"{minimum_j:.6e} is not above the configured floor "
            f"{config.microscopic_jacobian_floor:.6e}"
        )
    if maximum_j < minimum_j:
        raise ValueError("maximum microscopic Jacobian is below its minimum")
    if float(sample["minimum_particle_clearance"]) <= 0.0:
        raise RuntimeError("particle has nonpositive clearance from the cell boundary")
    for name in ("mixed_iterations", "rotation_iterations"):
        if int(sample[name]) < 0:
            raise ValueError(f"{name} cannot be negative")
    for name in ("mixed_residual", "rotation_residual", "condensed_residual"):
        if float(sample[name]) >= 1.0e-8:
            raise RuntimeError(f"{name} does not meet the 1e-8 convergence threshold")


def solve_dataset_sample(mesh, deformation_gradient, volume_fraction, config):
    """Solve one RVE state and return arrays ready for serialization."""
    initial_rotation = _polar_rotation(deformation_gradient)
    free_state = solve_free_rotating_particle(
        mesh,
        deformation_gradient,
        initial_rotation=initial_rotation,
        mu=config.mu,
        bulk_modulus=config.bulk_modulus,
        displacement_order=config.displacement_order,
        integration_order=config.integration_order,
        verbose=False,
    )
    tangent_result = condensed_free_rotation_tangent(
    mesh,
    free_state.solution,
    _matrix_cf(deformation_gradient),
    free_state.rotation,
    mu=config.mu,
    bulk_modulus=config.bulk_modulus,
    integration_order=config.integration_order,
    )
    _, pressure = free_state.solution.components
    energy = _homogenized_energy(
        mesh,
        free_state.deformation_gradient,
        pressure,
        config.mu,
        config.bulk_modulus,
        config.integration_order,
    )
    minimum_j, maximum_j = microscopic_jacobian_extrema(
        mesh,
        free_state.deformation_gradient,
        config.integration_order,
    )
    minimum_clearance = 0.5 - particle_radius(volume_fraction)
    sample = {
        "volume_fraction": np.asarray(volume_fraction),
        "deformation_gradient": np.asarray(deformation_gradient),
        "energy": np.asarray(energy),
        "stress": tangent_result.stress,
        "tangent": tangent_result.tangent,
        "rotation": free_state.rotation,
        "determinant": np.asarray(np.linalg.det(deformation_gradient)),
        "condition_number": np.asarray(np.linalg.cond(deformation_gradient)),
        "mixed_residual": np.asarray(free_state.newton_history[-1]),
        "rotation_residual": np.asarray(free_state.rotational_residuals[-1]),
        "condensed_residual": np.asarray(max(tangent_result.linear_residuals)),
        "tangent_symmetry_error": np.asarray(
            np.linalg.norm(tangent_result.tangent - tangent_result.tangent.T)
            / max(1.0, np.linalg.norm(tangent_result.tangent))
        ),
        "minimum_microscopic_jacobian": np.asarray(minimum_j),
        "maximum_microscopic_jacobian": np.asarray(maximum_j),
        "mixed_iterations": np.asarray(len(free_state.newton_history) - 1),
        "rotation_iterations": np.asarray(
            max(0, len(free_state.rotational_residuals) - 1)
        ),
        "minimum_particle_clearance": np.asarray(minimum_clearance),
        "solver_status": np.asarray("converged"),
    }
    validate_solved_sample(sample, config)
    return sample


def _matrix_cf(matrix):
    from ngsolve import CF
    return CF(tuple(np.asarray(matrix).ravel()), dims=(3, 3))


def _atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        np.savez_compressed(temporary, **arrays)
    temporary_path.replace(path)


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _config_payload(config):
    payload = asdict(config)
    payload["volume_fractions"] = list(config.volume_fractions)
    payload["split_fractions"] = list(config.split_fractions)
    payload["compression_determinant_range"] = list(
        config.compression_determinant_range
    )
    payload["expansion_determinant_range"] = list(
        config.expansion_determinant_range
    )
    return payload


def _configuration_sha256(config):
    encoded = json.dumps(
        _config_payload(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_revision_sha256():
    source_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _DATASET_SOURCE_FILES:
        path = source_root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_rve_dataset(config, output, work_directory=None, resume=True):
    """Generate, checkpoint, validate, and assemble one compressed dataset."""
    config.validate()
    output = Path(output)
    work_directory = (
        Path(work_directory)
        if work_directory is not None
        else output.with_suffix("").with_name(output.stem + "_work")
    )
    sample_directory = work_directory / "samples"
    sample_directory.mkdir(parents=True, exist_ok=True)

    gradients = generate_deformation_states(config)
    splits = assign_deformation_splits(
        config.n_deformations,
        config.seed,
        config.split_fractions,
    )
    state_hash = hashlib.sha256(gradients.tobytes()).hexdigest()
    run_payload = {
        "schema_version": SCHEMA_VERSION,
        "config": _config_payload(config),
        "configuration_sha256": _configuration_sha256(config),
        "source_revision_sha256": _source_revision_sha256(),
        "deformation_sha256": state_hash,
    }
    run_file = work_directory / "run.json"
    if run_file.exists():
        existing = json.loads(run_file.read_text(encoding="utf-8"))
        if existing != run_payload:
            raise ValueError("work directory belongs to a different dataset configuration")
    else:
        _atomic_json(run_file, run_payload)

    total = config.n_deformations * len(config.volume_fractions)
    completed = 0
    records = []
    for volume_index, volume_fraction in enumerate(config.volume_fractions):
        radius = particle_radius(volume_fraction)
        mesh = make_periodic_particle_cell_mesh(radius=radius, maxh=config.maxh)
        for state_index, gradient in enumerate(gradients):
            sample_id = f"v{volume_index:03d}_f{state_index:06d}"
            checkpoint = sample_directory / f"{sample_id}.npz"
            sample = None
            if checkpoint.exists() and resume:
                with np.load(checkpoint, allow_pickle=False) as saved:
                    sample = {name: saved[name] for name in saved.files}
                if (
                    not np.array_equal(sample["deformation_gradient"], gradient)
                    or not np.isclose(sample["volume_fraction"], volume_fraction)
                ):
                    raise ValueError(f"checkpoint {checkpoint} does not match its sample")
                try:
                    validate_solved_sample(sample, config)
                except (ValueError, RuntimeError, FloatingPointError):
                    sample = None
            if sample is None:
                sample = solve_dataset_sample(
                    mesh,
                    gradient,
                    volume_fraction,
                    config,
                )
                _atomic_savez(checkpoint, **sample)
            sample["state_index"] = np.asarray(state_index)
            sample["split"] = np.asarray(splits[state_index])
            sample["sample_id"] = np.asarray(sample_id)
            records.append(sample)
            completed += 1
            print(
                f"[{completed:5d}/{total}] {sample_id} "
                f"vf={volume_fraction:.4f} split={splits[state_index]}",
                flush=True,
            )

    field_names = tuple(records[0])
    dataset = {
        name: np.stack([record[name] for record in records])
        for name in field_names
    }
    _atomic_savez(output, **dataset)
    manifest = {
        **run_payload,
        "dataset_sha256": _file_sha256(output),
        "tangent_flattening": TANGENT_FLATTENING,
        "ngsolve_version": ngsolve.__version__,
        "n_samples": total,
        "fields": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in dataset.items()
        },
        "split_counts": {
            name: int(np.count_nonzero(dataset["split"] == name))
            for name in SPLIT_NAMES
        },
    }
    _atomic_json(output.with_suffix(".json"), manifest)
    return dataset, manifest
