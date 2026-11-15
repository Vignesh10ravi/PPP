"""Calibrate a frozen Phase 6/7 pICNN's simple-shear tangent response.

The original Phase 6/7 training split is replayed. Direct-RVE shear data supply
energy, P01, and dP01/dF01 objectives, with an additional VF-balanced dP01/dF01
anchor at identity and the lowest shear amplitudes. Identity tangent labels
come from a previously inspected base identity state; all identity records are
excluded from the old test regression monitor. The input checkpoint is never
modified.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .picnn import HyperelasticNN
from .train_rve import (
    _grouped_energy_stress,
    _grouped_tangent,
    _load_rve_dataset,
    _relative_tangent_loss,
    _select_tangent_validation_indices,
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_checkpoint(path, expected_sha256):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: {path}")
    checkpoint_sha256 = _sha256_file(path)
    if expected_sha256 and checkpoint_sha256 != expected_sha256:
        raise ValueError("baseline checkpoint SHA-256 differs from expected frozen hash")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("baseline checkpoint does not contain a state_dict")
    normalization = payload.get("normalization")
    required = (
        "feature_mean", "feature_scale", "context_mean", "context_scale",
        "energy_scale", "stress_scale",
    )
    if not isinstance(normalization, dict) or set(required).difference(normalization):
        raise ValueError("baseline checkpoint normalization metadata is incomplete")

    model = HyperelasticNN(n_ctx=1).to(dtype=torch.float64, device="cpu")
    model.load_state_dict(payload["state_dict"])
    model.set_normalization(
        torch.as_tensor(normalization["feature_mean"], dtype=torch.float64),
        torch.as_tensor(normalization["feature_scale"], dtype=torch.float64),
        torch.as_tensor(normalization["context_mean"], dtype=torch.float64),
        torch.as_tensor(normalization["context_scale"], dtype=torch.float64),
    )
    model.eval()
    return model, payload, checkpoint_sha256


def _to_tensors(arrays):
    return {
        "F": torch.as_tensor(arrays["deformation_gradient"], dtype=torch.float64),
        "vf": torch.as_tensor(
            arrays["volume_fraction"].reshape(-1, 1), dtype=torch.float64
        ),
        "energy": torch.as_tensor(arrays["energy"], dtype=torch.float64),
        "stress": torch.as_tensor(arrays["stress"], dtype=torch.float64),
        "tangent": torch.as_tensor(arrays["tangent"], dtype=torch.float64),
    }


def _select(tensors, indices):
    index = torch.as_tensor(indices, dtype=torch.long)
    return {name: value.index_select(0, index) for name, value in tensors.items()}


def _general_loss(model, tensors, energy_scale, stress_scale, stress_weight=1.0):
    predicted_energy, predicted_stress = _grouped_energy_stress(
        model, tensors["F"], tensors["vf"]
    )
    energy_error = predicted_energy - tensors["energy"]
    stress_error = predicted_stress - tensors["stress"]
    energy_loss = torch.mean((energy_error / energy_scale) ** 2)
    stress_loss = torch.mean((stress_error / stress_scale) ** 2)
    return energy_loss + stress_weight * stress_loss, {
        "energy_loss_normalized": float(energy_loss.detach()),
        "stress_loss_normalized": float(stress_loss.detach()),
        "energy_rmse": float(torch.sqrt(torch.mean(energy_error.detach() ** 2))),
        "stress_rmse": float(torch.sqrt(torch.mean(stress_error.detach() ** 2))),
    }


def _shear_response_loss(model, tensors, energy_scale, p01_scale):
    predicted_energy, predicted_stress = _grouped_energy_stress(
        model, tensors["F"], tensors["vf"]
    )
    energy_loss = torch.mean(
        ((predicted_energy - tensors["energy"]) / energy_scale) ** 2
    )
    p01_loss = torch.mean(
        ((predicted_stress[:, 0, 1] - tensors["stress"][:, 0, 1]) / p01_scale) ** 2
    )
    return 0.5 * (energy_loss + p01_loss), {
        "energy_relative_scale_mse": float(energy_loss.detach()),
        "P01_training_scale_mse": float(p01_loss.detach()),
    }


def _shear_tangent_loss(model, tensors, tangent_scale):
    predicted_tangent = _grouped_tangent(model, tensors["F"], tensors["vf"])
    error = predicted_tangent[:, 1, 1] - tensors["tangent"][:, 1, 1]
    loss = torch.mean((error / tangent_scale) ** 2)
    return loss, {
        "dP01_dF01_training_scale_mse": float(loss.detach()),
        "predicted_dP01_dF01_rmse": float(torch.sqrt(torch.mean(
            predicted_tangent[:, 1, 1].detach() ** 2
        ))),
        "direct_dP01_dF01_rmse": float(torch.sqrt(torch.mean(
            tensors["tangent"][:, 1, 1] ** 2
        ))),
    }


def _component_tangent_relative_loss(model, tensors):
    """Relative dP01/dF01 loss, giving each volume fraction equal weight."""
    predicted_tangent = _grouped_tangent(model, tensors["F"], tensors["vf"])
    target = tensors["tangent"][:, 1, 1]
    prediction = predicted_tangent[:, 1, 1]
    relative_error = (prediction - target) / torch.clamp(
        torch.abs(target), min=1.0e-8
    )
    volume_fractions = tensors["vf"].reshape(-1)
    per_vf = {}
    losses = []
    for volume_fraction in torch.unique(volume_fractions):
        mask = torch.isclose(
            volume_fractions, volume_fraction, rtol=0.0, atol=1.0e-12
        )
        vf_loss = torch.mean(relative_error[mask] ** 2)
        losses.append(vf_loss)
        per_vf[f"{float(volume_fraction):.2f}"] = float(vf_loss.detach())
    loss = torch.stack(losses).mean()
    return loss, {
        "dP01_dF01_relative_mse_vf_balanced": float(loss.detach()),
        "dP01_dF01_relative_mse_by_volume_fraction": per_vf,
    }


def _targeted_tangent_anchor_loss(model, identity_tensors, low_shear_tensors):
    identity_loss, identity_metrics = _component_tangent_relative_loss(
        model, identity_tensors
    )
    low_shear_loss, low_shear_metrics = _component_tangent_relative_loss(
        model, low_shear_tensors
    )
    loss = 0.5 * (identity_loss + low_shear_loss)
    return loss, {
        "identity_component_relative_mse": float(identity_loss.detach()),
        "low_shear_component_relative_mse": float(low_shear_loss.detach()),
        "identity_by_volume_fraction": identity_metrics[
            "dP01_dF01_relative_mse_by_volume_fraction"
        ],
        "low_shear_by_volume_fraction": low_shear_metrics[
            "dP01_dF01_relative_mse_by_volume_fraction"
        ],
    }


def _baseline_scales(arrays, checkpoint):
    normalization = checkpoint["normalization"]
    energy_scale = max(float(normalization["energy_scale"]), 1.0e-8)
    stress_scale = max(float(normalization["stress_scale"]), 1.0e-8)
    shear_train = arrays["split"] == "train"
    p01 = np.asarray(arrays["stress"][shear_train, 0, 1], dtype=float)
    energy = np.asarray(arrays["energy"][shear_train], dtype=float)
    tangent = np.asarray(arrays["tangent"][shear_train, 1, 1], dtype=float)
    if not len(p01):
        raise ValueError("shear data has no training states")
    return {
        "energy": energy_scale,
        "stress": stress_scale,
        "shear_energy": max(float(np.sqrt(np.mean(energy**2))), 1.0e-8),
        "shear_P01": max(float(np.sqrt(np.mean(p01**2))), 1.0e-8),
        "shear_dP01_dF01": max(float(np.sqrt(np.mean(tangent**2))), 1.0e-8),
    }


def _relative_shear_metrics(model, tensors, arrays, scales):
    with torch.enable_grad():
        response_loss, _ = _shear_response_loss(
            model, tensors, scales["shear_energy"], scales["shear_P01"]
        )
        tangent_loss, _ = _shear_tangent_loss(
            model, tensors, scales["shear_dP01_dF01"]
        )
        predicted_energy, predicted_stress = _grouped_energy_stress(
            model, tensors["F"], tensors["vf"]
        )
        predicted_tangent = _grouped_tangent(model, tensors["F"], tensors["vf"])
    p01_target = tensors["stress"][:, 0, 1]
    p01_prediction = predicted_stress[:, 0, 1]
    tangent_target = tensors["tangent"][:, 1, 1]
    tangent_prediction = predicted_tangent[:, 1, 1]
    energy_target = tensors["energy"]
    energy_relative = torch.abs(predicted_energy - energy_target) / torch.clamp(
        torch.abs(energy_target), min=1.0e-8
    )
    p01_relative = torch.abs(p01_prediction - p01_target) / torch.clamp(
        torch.abs(p01_target), min=1.0e-8
    )
    tangent_relative = torch.abs(tangent_prediction - tangent_target) / torch.clamp(
        torch.abs(tangent_target), min=1.0e-8
    )
    rows = []
    for index in range(len(arrays["split"])):
        F = np.asarray(arrays["deformation_gradient"][index], dtype=float)
        rows.append({
            "shear_amplitude": float(F[0, 1]),
            "volume_fraction": float(arrays["volume_fraction"][index]),
            "energy_direct_rve": float(energy_target[index]),
            "energy_pICNN": float(predicted_energy[index].detach()),
            "energy_relative_error": float(energy_relative[index].detach()),
            "P01_direct_rve": float(p01_target[index]),
            "P01_pICNN": float(p01_prediction[index].detach()),
            "P01_absolute_error": float(
                torch.abs(p01_prediction[index] - p01_target[index]).detach()
            ),
            "P01_relative_error": float(p01_relative[index].detach()),
            "dP01_dF01_direct_rve": float(tangent_target[index]),
            "dP01_dF01_pICNN": float(tangent_prediction[index].detach()),
            "dP01_dF01_relative_error": float(tangent_relative[index].detach()),
            "stress_frobenius_relative_error": float(
                torch.linalg.vector_norm(
                    predicted_stress[index] - tensors["stress"][index]
                ).detach()
                / torch.clamp(
                    torch.linalg.vector_norm(tensors["stress"][index]), min=1.0e-8
                )
            ),
        })
    summary = {
        "n_cases": len(rows),
        "energy_response_loss_on_training_scales": float(response_loss.detach()),
        "shear_tangent_loss_on_training_scale": float(tangent_loss.detach()),
        "mean_energy_relative_error": float(torch.mean(energy_relative).detach()),
        "maximum_energy_relative_error": float(torch.max(energy_relative).detach()),
        "mean_P01_relative_error": float(torch.mean(p01_relative).detach()),
        "maximum_P01_relative_error": float(torch.max(p01_relative).detach()),
        "mean_dP01_dF01_relative_error": float(torch.mean(tangent_relative).detach()),
        "maximum_dP01_dF01_relative_error": float(torch.max(tangent_relative).detach()),
        "stress_frobenius_relative_rmse": float(torch.sqrt(torch.mean(
            torch.as_tensor([
                row["stress_frobenius_relative_error"] ** 2 for row in rows
            ], dtype=torch.float64)
        ))),
    }
    return summary, rows


def _evaluate_general(model, tensors, scales):
    with torch.enable_grad():
        loss, metrics = _general_loss(
            model, tensors, scales["energy"], scales["stress"]
        )
    metrics["loss"] = float(loss.detach())
    return metrics


def _save_checkpoint(path, model, normalization, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = {
        "state_dict": copy.deepcopy(model.state_dict()),
        "normalization": normalization,
        **metadata,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _loss_value(model, tensors, scales):
    with torch.enable_grad():
        loss, _ = _general_loss(model, tensors, scales["energy"], scales["stress"])
    return float(loss.detach())


def train(args):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base, base_manifest, base_hash = _load_rve_dataset(args.base_dataset, args.base_manifest)
    shear, shear_manifest, shear_hash = _load_rve_dataset(
        args.shear_dataset, args.shear_manifest
    )
    if base_manifest.get("dataset_sha256") != base_hash:
        raise ValueError("base Phase 6/7 dataset hash mismatch")
    if shear_manifest.get("dataset_sha256") != shear_hash:
        raise ValueError("shear adaptation dataset hash mismatch")

    model, checkpoint, checkpoint_sha256 = _load_checkpoint(
        args.initialize_from, args.expected_checkpoint_sha256
    )
    checkpoint_dataset_hash = checkpoint.get("dataset_sha256")
    if checkpoint_dataset_hash and checkpoint_dataset_hash != base_hash:
        raise ValueError(
            "frozen checkpoint was not trained on the supplied Phase 6/7 dataset"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    normalization = checkpoint["normalization"]
    scales = _baseline_scales(shear, checkpoint)

    base_tensors = _to_tensors(base)
    shear_tensors = _to_tensors(shear)
    base_train_indices = np.flatnonzero(base["split"] == "train")
    base_validation_indices = np.flatnonzero(base["split"] == "validation")
    base_test_indices = np.flatnonzero(base["split"] == "test")
    shear_train_indices = np.flatnonzero(shear["split"] == "train")
    shear_validation_indices = np.flatnonzero(shear["split"] == "validation")
    shear_test_indices = np.flatnonzero(shear["split"] == "test")
    if not all((len(base_train_indices), len(base_validation_indices),
                len(base_test_indices), len(shear_train_indices),
                len(shear_validation_indices), len(shear_test_indices))):
        raise ValueError("base and shear datasets must have nonempty train/validation/test splits")

    if set(np.round(base["volume_fraction"], 12)) != set(
        np.round(shear["volume_fraction"], 12)
    ):
        raise ValueError("base and shear datasets must cover the same volume fractions")

    identity = np.all(
        np.isclose(
            base["deformation_gradient"],
            np.eye(3, dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        ),
        axis=(1, 2),
    )
    identity_anchor_indices = []
    identity_anchor_source_splits = {}
    for volume_fraction in sorted(np.unique(base["volume_fraction"])):
        rows = np.flatnonzero(
            identity
            & np.isclose(
                base["volume_fraction"], volume_fraction, rtol=0.0, atol=1.0e-12
            )
        )
        if not len(rows):
            raise ValueError(
                f"base dataset has no identity tangent anchor at VF={volume_fraction}"
            )
        split_names = set(base["split"][rows].tolist())
        if len(split_names) != 1:
            raise ValueError("duplicate identity rows are assigned to different splits")
        values = base["tangent"][rows, 1, 1]
        if not np.allclose(values, values[0], rtol=1.0e-12, atol=1.0e-12):
            raise ValueError(
                f"duplicate identity tangent labels differ at VF={volume_fraction}"
            )
        identity_anchor_indices.append(int(rows[0]))
        identity_anchor_source_splits[f"{float(volume_fraction):.2f}"] = next(
            iter(split_names)
        )
    identity_anchor_indices = np.asarray(identity_anchor_indices, dtype=int)
    # Identity labels were previously held out. They now serve only as explicit
    # calibration anchors, so remove all duplicate identity records from the
    # original-test regression monitor below.
    base_test_indices = base_test_indices[~identity[base_test_indices]]
    if not len(base_test_indices):
        raise ValueError(
            "no original test records remain after excluding identity calibration anchors"
        )

    shear_amplitudes = np.abs(shear["deformation_gradient"][:, 0, 1])
    low_shear_anchor_indices = shear_train_indices[
        shear_amplitudes[shear_train_indices]
        <= args.low_shear_max_amplitude + 1.0e-12
    ]
    if not len(low_shear_anchor_indices):
        raise ValueError(
            "no low-shear training rows fall below the configured amplitude"
        )
    for volume_fraction in np.unique(shear["volume_fraction"]):
        if not np.any(
            np.isclose(
                shear["volume_fraction"][low_shear_anchor_indices],
                volume_fraction,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError(
                f"low-shear training anchors are missing VF={volume_fraction}"
            )

    design = shear_manifest.get("deformation_design")
    if not isinstance(design, dict) or design.get("load_mode") != (
        "positive simple shear, F[0,1]=gamma"
    ):
        raise ValueError("shear dataset manifest has no recognized deformation design")
    baseline_amplitudes = {0.025, 0.05, 0.075}
    for split_name, indices in (
        ("train", shear_train_indices),
        ("validation", shear_validation_indices),
        ("test", shear_test_indices),
    ):
        amplitudes = np.round(shear["deformation_gradient"][indices, 0, 1], 12)
        expected_amplitudes = np.round(
            np.asarray(design["amplitudes_by_split"][split_name], dtype=float), 12
        )
        if set(amplitudes.tolist()) != set(expected_amplitudes.tolist()):
            raise ValueError(
                f"manifest amplitudes do not match the actual {split_name} rows"
            )
        if baseline_amplitudes.intersection(set(amplitudes.tolist())):
            raise ValueError(
                f"existing shear baseline amplitude found in new {split_name} split"
            )
    if np.any(np.abs(shear["deformation_gradient"][:, 0, 1]) > 0.10 + 1e-12):
        raise ValueError("new shear data must remain within the configured 10% path")
    pure_shear = np.repeat(np.eye(3, dtype=float)[None, :, :], len(shear["split"]), axis=0)
    pure_shear[:, 0, 1] = shear["deformation_gradient"][:, 0, 1]
    if not np.allclose(
        shear["deformation_gradient"], pure_shear, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("shear adaptation data must contain only F[0,1] simple shear")
    amplitude_split_map = {}
    for amplitude, split_name in zip(
        shear["deformation_gradient"][:, 0, 1], shear["split"]
    ):
        previous = amplitude_split_map.setdefault(round(float(amplitude), 12), split_name)
        if previous != split_name:
            raise ValueError("the same shear amplitude appears in multiple splits")

    # Original training and validation remain separate. Identity labels are
    # used as a declared calibration anchor and excluded from test monitoring.
    validation_tangent_indices = _select_tangent_validation_indices(
        base_validation_indices,
        base["state_index"],
        np.linalg.det(base["deformation_gradient"]),
        args.original_tangent_validation_states,
        args.seed,
    )

    base_train = _select(base_tensors, base_train_indices)
    base_val = _select(base_tensors, base_validation_indices)
    base_val_tangent = _select(base_tensors, validation_tangent_indices)
    shear_train = _select(shear_tensors, shear_train_indices)
    shear_val = _select(shear_tensors, shear_validation_indices)
    identity_anchor_tensors = _select(base_tensors, identity_anchor_indices)
    low_shear_anchor_tensors = _select(shear_tensors, low_shear_anchor_indices)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    base_train_reference = max(
        _loss_value(model, base_train, scales), args.loss_floor
    )
    shear_response_train_reference = max(
        float(_shear_response_loss(
            model,
            shear_train,
            scales["shear_energy"],
            scales["shear_P01"],
        )[0].detach()),
        args.loss_floor,
    )
    shear_tangent_train_reference = max(
        float(_shear_tangent_loss(
            model, shear_train, scales["shear_dP01_dF01"]
        )[0].detach()),
        args.loss_floor,
    )
    targeted_tangent_anchor_reference = max(
        float(_targeted_tangent_anchor_loss(
            model, identity_anchor_tensors, low_shear_anchor_tensors
        )[0].detach()),
        args.loss_floor,
    )
    baseline_val_orig = _loss_value(model, base_val, scales)
    baseline_val_tangent = float(_relative_tangent_loss(
        _grouped_tangent(
            model,
            base_val_tangent["F"],
            base_val_tangent["vf"],
        ),
        base_val_tangent["tangent"],
    ).detach())
    baseline_val_shear_response = float(_shear_response_loss(
        model,
        shear_val,
        scales["shear_energy"],
        scales["shear_P01"],
    )[0].detach())
    baseline_val_shear_tangent = float(_shear_tangent_loss(
        model, shear_val, scales["shear_dP01_dF01"]
    )[0].detach())
    baseline_val_targeted_tangent = float(
        _component_tangent_relative_loss(model, shear_val)[0].detach()
    )
    original_val_reference = max(
        baseline_val_orig + args.original_tangent_weight * baseline_val_tangent,
        args.loss_floor,
    )
    shear_val_reference = max(
        baseline_val_shear_response
        + args.shear_tangent_weight * baseline_val_shear_tangent
        + args.targeted_shear_tangent_weight * baseline_val_targeted_tangent,
        args.loss_floor,
    )

    def validation_snapshot(candidate):
        original_loss, original_metrics = _general_loss(
            candidate, base_val, scales["energy"], scales["stress"]
        )
        original_tangent_loss = _relative_tangent_loss(
            _grouped_tangent(
                candidate, base_val_tangent["F"], base_val_tangent["vf"]
            ),
            base_val_tangent["tangent"],
        )
        shear_response, shear_response_metrics = _shear_response_loss(
            candidate, shear_val, scales["shear_energy"], scales["shear_P01"]
        )
        shear_tangent, shear_tangent_metrics = _shear_tangent_loss(
            candidate, shear_val, scales["shear_dP01_dF01"]
        )
        targeted_tangent, targeted_tangent_metrics = (
            _component_tangent_relative_loss(candidate, shear_val)
        )
        original_objective = float(
            original_loss.detach()
            + args.original_tangent_weight * original_tangent_loss.detach()
        )
        shear_objective = float(
            shear_response.detach()
            + args.shear_tangent_weight * shear_tangent.detach()
            + args.targeted_shear_tangent_weight * targeted_tangent.detach()
        )
        score = 0.5 * (
            original_objective / original_val_reference
            + shear_objective / shear_val_reference
        )
        return {
            "selection_score": float(score),
            "original_validation_objective": original_objective,
            "original_validation_relative_to_frozen": float(
                original_objective / original_val_reference
            ),
            "original_validation": {
                **original_metrics,
                "tangent_relative_frobenius_mse": float(original_tangent_loss.detach()),
            },
            "shear_validation_objective": shear_objective,
            "shear_validation_relative_to_frozen": float(
                shear_objective / shear_val_reference
            ),
            "shear_validation": {
                **shear_response_metrics,
                **shear_tangent_metrics,
                **targeted_tangent_metrics,
                "response_loss": float(shear_response.detach()),
                "tangent_loss": float(shear_tangent.detach()),
                "targeted_component_tangent_loss": float(targeted_tangent.detach()),
            },
        }

    initial_validation = validation_snapshot(model)
    best_score = initial_validation["selection_score"]
    best_epoch = 0
    best_validation = initial_validation
    stale_checks = 0
    history = []
    output_checkpoint = Path(args.checkpoint)
    output_report = Path(args.report)
    metadata = {
        "dataset_sha256": hashlib.sha256(
            (base_hash + shear_hash).encode("ascii")
        ).hexdigest(),
        "dataset_sha256_scope": (
            "SHA-256 over the ordered source dataset hashes; the fit reads the "
            "original Phase 6/7 dataset and the grouped shear adaptation dataset"
        ),
        "initialization_checkpoint_sha256": checkpoint_sha256,
        "trainer_code_sha256": _sha256_file(Path(__file__)),
        "training_config": {
            "method": "separate shear-tangent calibration fine-tune with original-data replay",
            "epochs": args.epochs,
            "batch_size_original": args.batch_size,
            "batch_size_shear_response": args.shear_batch_size,
            "batch_size_shear_tangent": args.shear_tangent_batch_size,
            "learning_rate": args.learning_rate,
            "original_tangent_weight": args.original_tangent_weight,
            "shear_tangent_weight": args.shear_tangent_weight,
            "targeted_shear_tangent_weight": args.targeted_shear_tangent_weight,
            "low_shear_max_amplitude": args.low_shear_max_amplitude,
            "targeted_tangent_calibration": (
                "equal-weighted identity and low-shear dP01/dF01 relative MSE; "
                "each volume fraction receives equal weight"
            ),
            "identity_tangent_anchor_sample_ids": [
                str(base["sample_id"][index]) for index in identity_anchor_indices
            ],
            "identity_tangent_anchor_original_splits": identity_anchor_source_splits,
            "identity_anchor_test_rows_excluded_from_regression_monitor": True,
            "selection": (
                "equal-weighted relative change in original validation and shear "
                "validation objectives from frozen-model values"
            ),
            "loss_scales": scales,
            "seed": args.seed,
            "device": "cpu",
        },
        "source_datasets": {
            "phase6_phase7_sha256": base_hash,
            "shear_adaptation_sha256": shear_hash,
        },
        "best_epoch": 0,
        "best_validation": best_validation,
    }
    _save_checkpoint(output_checkpoint, model, normalization, metadata)

    for epoch in range(1, args.epochs + 1):
        model.train()
        original_permutation = np.random.permutation(base_train_indices)
        original_loss_sum = 0.0
        shear_loss_sum = 0.0
        step_count = 0

        for start in range(0, len(original_permutation), args.batch_size):
            batch_indices = original_permutation[start : start + args.batch_size]
            shear_batch_indices = np.random.choice(
                shear_train_indices,
                size=args.shear_batch_size,
                replace=True,
            )
            original_batch = _select(base_tensors, batch_indices)
            shear_batch = _select(shear_tensors, shear_batch_indices)
            original_loss, _ = _general_loss(
                model, original_batch, scales["energy"], scales["stress"]
            )
            shear_response_loss, _ = _shear_response_loss(
                model,
                shear_batch,
                scales["shear_energy"],
                scales["shear_P01"],
            )
            objective = (
                original_loss / base_train_reference
                + args.shear_response_weight
                * shear_response_loss
                / shear_response_train_reference
            )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            original_loss_sum += float(original_loss.detach())
            shear_loss_sum += float(shear_response_loss.detach())
            step_count += 1

        model.train()
        shear_tangent_indices = np.random.choice(
            shear_train_indices,
            size=min(args.shear_tangent_batch_size, len(shear_train_indices)),
            replace=False,
        )
        shear_tangent_batch = _select(shear_tensors, shear_tangent_indices)
        shear_tangent_loss, _ = _shear_tangent_loss(
            model, shear_tangent_batch, scales["shear_dP01_dF01"]
        )
        targeted_tangent_loss, targeted_tangent_metrics = (
            _targeted_tangent_anchor_loss(
                model, identity_anchor_tensors, low_shear_anchor_tensors
            )
        )
        optimizer.zero_grad(set_to_none=True)
        (
            args.shear_tangent_weight * shear_tangent_loss
            / shear_tangent_train_reference
            + args.targeted_shear_tangent_weight * targeted_tangent_loss
            / targeted_tangent_anchor_reference
        ).backward()
        optimizer.step()

        original_tangent_indices = np.random.choice(
            base_train_indices,
            size=min(args.original_tangent_batch_size, len(base_train_indices)),
            replace=False,
        )
        original_tangent_batch = _select(base_tensors, original_tangent_indices)
        original_tangent_loss = _relative_tangent_loss(
            _grouped_tangent(
                model,
                original_tangent_batch["F"],
                original_tangent_batch["vf"],
            ),
            original_tangent_batch["tangent"],
        )
        optimizer.zero_grad(set_to_none=True)
        (
            args.original_tangent_weight
            * original_tangent_loss
            / max(float(baseline_val_tangent), args.loss_floor)
        ).backward()
        optimizer.step()

        if epoch % args.validation_interval != 0 and epoch != args.epochs:
            continue
        model.eval()
        with torch.enable_grad():
            current_validation = validation_snapshot(model)
        history.append({
            "epoch": epoch,
            "mean_original_batch_loss": original_loss_sum / max(step_count, 1),
            "mean_shear_response_batch_loss": shear_loss_sum / max(step_count, 1),
            "shear_tangent_batch_loss": float(shear_tangent_loss.detach()),
            "targeted_tangent_anchor_loss": float(targeted_tangent_loss.detach()),
            "targeted_tangent_anchor_metrics": targeted_tangent_metrics,
            "original_tangent_batch_loss": float(original_tangent_loss.detach()),
            "validation": current_validation,
        })
        print(
            f"epoch={epoch} selection={current_validation['selection_score']:.6e} "
            f"original_ratio={current_validation['original_validation_relative_to_frozen']:.4f} "
            f"shear_ratio={current_validation['shear_validation_relative_to_frozen']:.4f}",
            flush=True,
        )
        if current_validation["selection_score"] < best_score:
            best_score = current_validation["selection_score"]
            best_epoch = epoch
            best_validation = current_validation
            stale_checks = 0
            metadata["best_epoch"] = best_epoch
            metadata["best_validation"] = best_validation
            _save_checkpoint(output_checkpoint, model, normalization, metadata)
        else:
            stale_checks += 1
            if stale_checks >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    trained_payload = torch.load(output_checkpoint, map_location="cpu", weights_only=True)
    candidate = HyperelasticNN(n_ctx=1).to(dtype=torch.float64, device="cpu")
    candidate.load_state_dict(trained_payload["state_dict"])
    candidate.set_normalization(
        torch.as_tensor(normalization["feature_mean"], dtype=torch.float64),
        torch.as_tensor(normalization["feature_scale"], dtype=torch.float64),
        torch.as_tensor(normalization["context_mean"], dtype=torch.float64),
        torch.as_tensor(normalization["context_scale"], dtype=torch.float64),
    )
    candidate.eval()
    baseline_model, _, _ = _load_checkpoint(args.initialize_from, checkpoint_sha256)

    base_test = _select(base_tensors, base_test_indices)
    baseline_original_test = _evaluate_general(baseline_model, base_test, scales)
    candidate_original_test = _evaluate_general(candidate, base_test, scales)
    shear_test_arrays = {
        key: value[shear_test_indices] for key, value in shear.items()
    }
    shear_test_tensors = _select(shear_tensors, shear_test_indices)
    baseline_shear_test, baseline_shear_rows = _relative_shear_metrics(
        baseline_model, shear_test_tensors, shear_test_arrays, scales
    )
    candidate_shear_test, candidate_shear_rows = _relative_shear_metrics(
        candidate, shear_test_tensors, shear_test_arrays, scales
    )
    for baseline_row, candidate_row in zip(baseline_shear_rows, candidate_shear_rows):
        if (
            baseline_row["shear_amplitude"] != candidate_row["shear_amplitude"]
            or baseline_row["volume_fraction"] != candidate_row["volume_fraction"]
        ):
            raise RuntimeError("baseline and candidate test row order differs")
        for key in (
            "energy_direct_rve", "P01_direct_rve", "dP01_dF01_direct_rve"
        ):
            if baseline_row[key] != candidate_row[key]:
                raise RuntimeError("baseline and candidate do not share test labels")
    paired_test_rows = []
    for baseline_row, candidate_row in zip(baseline_shear_rows, candidate_shear_rows):
        paired_test_rows.append({
            "shear_amplitude": candidate_row["shear_amplitude"],
            "volume_fraction": candidate_row["volume_fraction"],
            **{f"baseline_{key}": value for key, value in baseline_row.items()
               if key not in {"shear_amplitude", "volume_fraction"}},
            **{f"candidate_{key}": value for key, value in candidate_row.items()
               if key not in {"shear_amplitude", "volume_fraction"}},
        })

    final_report = {
        "status": "complete",
        "scope": (
            "Separate shear-tangent calibration candidate initialized from the "
            "frozen Phase 6/7 checkpoint. The original checkpoint is unchanged. "
            "Original training data are replayed and original validation monitors "
            "retention. Identity dP01/dF01 anchors from a previously inspected "
            "base identity state are now calibration data; all identity records "
            "are excluded from the base-test regression monitor. The prior shear "
            "test has already been inspected and is development data only. A new "
            "independent shear holdout is needed for a final claim."
        ),
        "baseline_checkpoint_sha256": checkpoint_sha256,
        "candidate_checkpoint": str(output_checkpoint.resolve()),
        "candidate_checkpoint_sha256": _sha256_file(output_checkpoint),
        "base_dataset_sha256": base_hash,
        "shear_adaptation_dataset_sha256": shear_hash,
        "training_config": metadata["training_config"],
        "identity_tangent_calibration": {
            "sample_ids": [
                str(base["sample_id"][index]) for index in identity_anchor_indices
            ],
            "original_splits": identity_anchor_source_splits,
            "excluded_identity_rows_from_original_test_monitor": int(
                np.count_nonzero(identity & (base["split"] == "test"))
            ),
            "low_shear_training_amplitudes": sorted(
                set(np.round(shear["deformation_gradient"][low_shear_anchor_indices, 0, 1], 12).tolist())
            ),
            "volume_fractions": sorted(
                set(np.round(base["volume_fraction"][identity_anchor_indices], 12).tolist())
            ),
            "normalization": "per-record relative error, averaged equally across volume fractions",
        },
        "split_counts": {
            "original_phase6_phase7": {
                name: int(np.count_nonzero(base["split"] == name))
                for name in ("train", "validation", "test")
            },
            "new_shear": {
                name: int(np.count_nonzero(shear["split"] == name))
                for name in ("train", "validation", "test")
            },
        },
        "shear_design": shear_manifest.get("deformation_design"),
        "best_epoch": best_epoch,
        "candidate_improved_validation": bool(best_epoch > 0),
        "initial_validation": initial_validation,
        "best_validation": best_validation,
        "original_phase7_test_regression_monitor": {
            "frozen_checkpoint": baseline_original_test,
            "shear_adapted_candidate": candidate_original_test,
            "interpretation_note": (
                "This pre-existing test split is a regression monitor only, was "
                "previously inspected, and excludes identity records reused as "
                "calibration anchors. It is not a fresh final test."
            ),
            "n_samples_excluding_identity_calibration_anchors": int(len(base_test_indices)),
        },
        "previously_inspected_shear_development_set": {
            "frozen_checkpoint": baseline_shear_test,
            "shear_adapted_candidate": candidate_shear_test,
            "points": [
                {"frozen_checkpoint": left, "shear_adapted_candidate": right}
                for left, right in zip(baseline_shear_rows, candidate_shear_rows)
            ],
        },
        "history": history,
    }
    _atomic_json(output_report, final_report)
    csv_path = Path(args.test_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(paired_test_rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        import csv

        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(paired_test_rows)
    print(json.dumps({
        "status": final_report["status"],
        "best_epoch": best_epoch,
        "candidate_improved_validation": bool(best_epoch > 0),
        "best_selection_score": best_score,
        "original_test_frozen_stress_rmse": baseline_original_test["stress_rmse"],
        "original_test_candidate_stress_rmse": candidate_original_test["stress_rmse"],
        "shear_test_frozen_mean_P01_error": baseline_shear_test[
            "mean_P01_relative_error"
        ],
        "shear_test_candidate_mean_P01_error": candidate_shear_test[
            "mean_P01_relative_error"
        ],
        "candidate_checkpoint": str(output_checkpoint.resolve()),
        "report": str(output_report.resolve()),
        "csv": str(csv_path.resolve()),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--shear-dataset", type=Path, required=True)
    parser.add_argument("--shear-manifest", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shear-batch-size", type=int, default=16)
    parser.add_argument("--shear-tangent-batch-size", type=int, default=8)
    parser.add_argument("--original-tangent-batch-size", type=int, default=8)
    parser.add_argument("--original-tangent-validation-states", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--original-tangent-weight", type=float, default=0.10)
    parser.add_argument("--shear-response-weight", type=float, default=1.0)
    parser.add_argument("--shear-tangent-weight", type=float, default=1.0)
    parser.add_argument("--targeted-shear-tangent-weight", type=float, default=3.0)
    parser.add_argument("--low-shear-max-amplitude", type=float, default=0.02)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--loss-floor", type=float, default=1.0e-12)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--torch-threads", type=int, default=12)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.shear_batch_size < 1:
        parser.error("epochs and batch sizes must be positive")
    if (
        args.shear_tangent_batch_size < 1
        or args.original_tangent_batch_size < 1
        or args.original_tangent_validation_states < 1
    ):
        parser.error("tangent batch and validation-state counts must be positive")
    if args.learning_rate <= 0.0 or args.loss_floor <= 0.0:
        parser.error("learning rate and loss floor must be positive")
    if args.low_shear_max_amplitude <= 0.0:
        parser.error("low-shear maximum amplitude must be positive")
    if min(
        args.original_tangent_weight,
        args.shear_response_weight,
        args.shear_tangent_weight,
        args.targeted_shear_tangent_weight,
    ) < 0.0:
        parser.error("loss weights cannot be negative")
    if args.validation_interval < 1 or args.patience < 1:
        parser.error("validation interval and patience must be positive")
    train(args)


if __name__ == "__main__":
    main()
