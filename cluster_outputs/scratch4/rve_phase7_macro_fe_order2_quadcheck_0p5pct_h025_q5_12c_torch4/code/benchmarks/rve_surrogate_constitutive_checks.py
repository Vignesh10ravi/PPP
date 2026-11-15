"""Check constitutive consistency and objectivity of a trained RVE surrogate."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.func import hessian, jacrev

from src.picnn import HyperelasticNN


DETERMINANT_BINS = (
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.85),
    (0.85, 1.00),
    (1.00, 1.25),
    (1.25, float("inf")),
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_error(actual, expected):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    return float(
        np.linalg.norm(actual - expected)
        / max(1.0, float(np.linalg.norm(actual)), float(np.linalg.norm(expected)))
    )


def _rotation(dtype, device):
    axis = np.asarray((1.0, 2.0, -1.0), dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    angle = 0.37
    matrix = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    return torch.as_tensor(matrix, dtype=dtype, device=device)


def _state_functions(model, vf_tensor, dtype, device):
    context = vf_tensor.reshape(1, 1)
    reference = model.reference(dtype, device, context)

    def energy_matrix(gradient):
        return model.energy(
            gradient.unsqueeze(0), context, ref=reference
        ).squeeze()

    def energy_flat(flat_gradient):
        return energy_matrix(flat_gradient.reshape(3, 3))

    return energy_matrix, energy_flat


def _evaluate_state(model, gradient_np, volume_fraction, dtype, device):
    gradient = torch.as_tensor(gradient_np, dtype=dtype, device=device)
    vf_tensor = torch.as_tensor(volume_fraction, dtype=dtype, device=device)
    energy_matrix, energy_flat = _state_functions(
        model, vf_tensor, dtype, device
    )
    energy = energy_matrix(gradient)
    stress = jacrev(energy_matrix)(gradient)
    tangent = hessian(energy_flat)(gradient.reshape(-1))
    return energy, stress, tangent


def _select_test_states(data, states_per_bin):
    split = data["split"].astype(str)
    state_indices = np.asarray(data["state_index"], dtype=int)
    gradients = np.asarray(data["deformation_gradient"], dtype=float)
    determinants = np.linalg.det(gradients)
    eligible = np.flatnonzero((split == "test") & (state_indices != 0))
    state_rows = {}
    for row in eligible:
        state_rows.setdefault(int(state_indices[row]), int(row))

    chosen_states = []
    bin_selection = {}
    for lower, upper in DETERMINANT_BINS:
        candidate_states = [
            state
            for state, row in state_rows.items()
            if lower <= determinants[row] < upper
        ]
        candidate_states.sort(key=lambda state: (determinants[state_rows[state]], state))
        if candidate_states:
            positions = np.linspace(
                0,
                len(candidate_states) - 1,
                min(states_per_bin, len(candidate_states)),
                dtype=int,
            )
            selected = [candidate_states[position] for position in positions]
            chosen_states.extend(selected)
            bin_selection[f"[{lower:.2f}, {upper:.2f})"] = selected
        else:
            bin_selection[f"[{lower:.2f}, {upper:.2f})"] = []

    chosen_states = sorted(set(chosen_states))
    selected_rows = np.asarray(
        [
            row
            for row in eligible
            if int(state_indices[row]) in set(chosen_states)
        ],
        dtype=int,
    )
    if not len(selected_rows):
        raise ValueError("no independent test records were selected for the checks")
    return selected_rows, chosen_states, bin_selection


def _check_sample(model, sample_id, state_index, vf, gradient_np, fd_step, dtype, device, rotation):
    energy, stress, tangent = _evaluate_state(
        model, gradient_np, vf, dtype, device
    )
    energy_np = float(energy.detach().cpu())
    stress_np = stress.detach().cpu().numpy()
    tangent_np = tangent.detach().cpu().numpy()

    gradient = torch.as_tensor(gradient_np, dtype=dtype, device=device)
    vf_tensor = torch.as_tensor(vf, dtype=dtype, device=device)
    energy_matrix, _ = _state_functions(model, vf_tensor, dtype, device)
    energy_gradient_fd = np.empty((3, 3), dtype=float)
    tangent_fd = np.empty((9, 9), dtype=float)
    for column in range(9):
        direction = torch.zeros((3, 3), dtype=dtype, device=device)
        direction.reshape(-1)[column] = fd_step
        stress_plus = jacrev(energy_matrix)(gradient + direction)
        stress_minus = jacrev(energy_matrix)(gradient - direction)
        tangent_fd[:, column] = (
            (stress_plus - stress_minus).reshape(-1) / (2.0 * fd_step)
        ).detach().cpu().numpy()

        energy_plus = energy_matrix(gradient + direction)
        energy_minus = energy_matrix(gradient - direction)
        energy_gradient_fd.reshape(-1)[column] = float(
            ((energy_plus - energy_minus) / (2.0 * fd_step)).detach().cpu()
        )

    energy_gradient_error = _relative_error(energy_gradient_fd, stress_np)
    tangent_directional_error = _relative_error(tangent_fd, tangent_np)
    tangent_symmetry_error = float(
        np.linalg.norm(tangent_np - tangent_np.T)
        / max(1.0, float(np.linalg.norm(tangent_np)))
    )

    rotated_gradient_np = rotation.detach().cpu().numpy() @ gradient_np
    rotated_energy, rotated_stress, rotated_tangent = _evaluate_state(
        model, rotated_gradient_np, vf, dtype, device
    )
    rotation_np = rotation.detach().cpu().numpy()
    transform = torch.kron(rotation, torch.eye(3, dtype=dtype, device=device))
    tangent_expected = transform @ tangent @ transform.T
    objectivity_energy_error = abs(
        float(rotated_energy.detach().cpu()) - energy_np
    ) / max(1.0, abs(energy_np), abs(float(rotated_energy.detach().cpu())))
    objectivity_stress_error = _relative_error(
        rotated_stress.detach().cpu().numpy(), rotation_np @ stress_np
    )
    objectivity_tangent_error = _relative_error(
        rotated_tangent.detach().cpu().numpy(),
        tangent_expected.detach().cpu().numpy(),
    )

    metrics = {
        "energy_gradient_relative_error": energy_gradient_error,
        "tangent_directional_relative_error": tangent_directional_error,
        "tangent_symmetry_error": tangent_symmetry_error,
        "objectivity_energy_relative_error": float(objectivity_energy_error),
        "objectivity_stress_relative_error": objectivity_stress_error,
        "objectivity_tangent_relative_error": objectivity_tangent_error,
    }
    thresholds = {
        "energy_gradient_relative_error": 1.0e-6,
        "tangent_directional_relative_error": 1.0e-5,
        "tangent_symmetry_error": 1.0e-10,
        "objectivity_energy_relative_error": 1.0e-10,
        "objectivity_stress_relative_error": 1.0e-10,
        "objectivity_tangent_relative_error": 1.0e-9,
    }
    checks = {name: value <= thresholds[name] for name, value in metrics.items()}
    return {
        "sample_id": str(sample_id),
        "state_index": int(state_index),
        "volume_fraction": float(vf),
        "determinant": float(np.linalg.det(gradient_np)),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "metrics": metrics,
    }


def audit_surrogate(
    dataset_path,
    manifest_path,
    checkpoint_path,
    training_dataset_path,
    report_path,
    states_per_bin=2,
    fd_step=1.0e-5,
    torch_threads=4,
):
    if states_per_bin < 1:
        raise ValueError("states_per_bin must be at least one")
    if not np.isfinite(fd_step) or fd_step <= 0.0:
        raise ValueError("fd_step must be finite and positive")
    dataset_path = Path(dataset_path)
    manifest_path = Path(manifest_path)
    checkpoint_path = Path(checkpoint_path)
    training_dataset_path = Path(training_dataset_path)
    report_path = Path(report_path)

    dataset_hash = _sha256_file(dataset_path)
    training_dataset_hash = _sha256_file(training_dataset_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if manifest.get("dataset_sha256") != dataset_hash:
        raise ValueError("independent dataset hash does not match its manifest")
    if checkpoint.get("dataset_sha256") != training_dataset_hash:
        raise ValueError("training dataset hash does not match the checkpoint")
    if dataset_hash == training_dataset_hash:
        raise ValueError("constitutive audit expects a separate evaluation dataset")

    with np.load(dataset_path, allow_pickle=False) as archive:
        required = {
            "sample_id", "split", "state_index", "volume_fraction",
            "deformation_gradient",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"evaluation dataset is missing fields: {sorted(missing)}")
        data = {name: np.array(archive[name]) for name in required}

    torch.set_num_threads(int(torch_threads))
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    dtype = torch.float64
    model = HyperelasticNN(n_ctx=1).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    rotation = _rotation(dtype, device)

    selected_rows, chosen_states, bin_selection = _select_test_states(
        data, states_per_bin
    )
    records = []
    for row in selected_rows:
        records.append(
            _check_sample(
                model=model,
                sample_id=data["sample_id"][row],
                state_index=data["state_index"][row],
                vf=data["volume_fraction"][row],
                gradient_np=data["deformation_gradient"][row],
                fd_step=fd_step,
                dtype=dtype,
                device=device,
                rotation=rotation,
            )
        )

    metric_names = tuple(records[0]["metrics"])
    maxima = {
        name: float(max(record["metrics"][name] for record in records))
        for name in metric_names
    }
    report = {
        "status": "pass" if all(record["status"] == "pass" for record in records) else "fail",
        "evaluated_split": "test",
        "independent_evaluation": True,
        "identity_reference_state_excluded": True,
        "dataset_sha256": dataset_hash,
        "training_dataset_sha256": training_dataset_hash,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "evaluation_manifest_configuration_sha256": manifest.get("configuration_sha256"),
        "n_test_records_checked": int(len(records)),
        "n_deformation_states_checked": int(len(chosen_states)),
        "selected_states_by_determinant_bin": bin_selection,
        "finite_difference_step": float(fd_step),
        "thresholds": {
            "energy_gradient_relative_error": 1.0e-6,
            "tangent_directional_relative_error": 1.0e-5,
            "tangent_symmetry_error": 1.0e-10,
            "objectivity_energy_relative_error": 1.0e-10,
            "objectivity_stress_relative_error": 1.0e-10,
            "objectivity_tangent_relative_error": 1.0e-9,
        },
        "maximum_errors": maxima,
        "records": records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Constitutive audit {report['status']}: {len(records)} records across "
        f"{len(chosen_states)} independent deformation states",
        flush=True,
    )
    print(f"Maximum errors: {json.dumps(maxima, sort_keys=True)}", flush=True)
    print(f"Report: {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--states-per-det-bin", type=int, default=2)
    parser.add_argument("--fd-step", type=float, default=1.0e-5)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    audit_surrogate(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        training_dataset_path=args.training_dataset,
        report_path=args.report,
        states_per_bin=args.states_per_det_bin,
        fd_step=args.fd_step,
        torch_threads=args.torch_threads,
    )


if __name__ == "__main__":
    main()
