"""Compare reference-state RVE and pICNN moduli with rigid-sphere micromechanics.

The audit reads stored 9x9 tangents at F=I, projects them onto orthonormal
volumetric and deviatoric strain modes, and compares the resulting effective
bulk/shear moduli with the matrix-based Mori-Tanaka estimate in the rigid
spherical-inclusion limit. For a stiffer inclusion phase this estimate is
also the Hashin-Shtrikman lower reference under the usual isotropic assumptions.
The pICNN comparison records the dataset split for each selected reference
state. The Phase 7 reference states used here are test-split records, excluded
from fitting but previously inspected in project work; they are not a fresh
independent test.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.func import hessian

from src.picnn import HyperelasticNN


DEFAULT_VOLUME_FRACTIONS = (0.02, 0.05, 0.10)
IDENTITY_TOLERANCE = 1.0e-12


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
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


def _symmetric_strain_basis():
    """Orthonormal 3D basis: one volumetric and five deviatoric modes."""
    basis = np.zeros((6, 3, 3), dtype=float)
    basis[0] = np.eye(3) / np.sqrt(3.0)
    basis[1] = np.diag((1.0, -1.0, 0.0)) / np.sqrt(2.0)
    basis[2] = np.diag((1.0, 1.0, -2.0)) / np.sqrt(6.0)
    basis[3, 0, 1] = basis[3, 1, 0] = 1.0 / np.sqrt(2.0)
    basis[4, 0, 2] = basis[4, 2, 0] = 1.0 / np.sqrt(2.0)
    basis[5, 1, 2] = basis[5, 2, 1] = 1.0 / np.sqrt(2.0)
    return basis


def _extract_moduli(tangent):
    tangent = np.asarray(tangent, dtype=float)
    if tangent.shape != (9, 9) or not np.all(np.isfinite(tangent)):
        raise ValueError("each tangent must be a finite 9x9 matrix")

    basis = _symmetric_strain_basis().reshape(6, 9)
    projected = basis @ tangent @ basis.T
    projected_symmetric = 0.5 * (projected + projected.T)
    deviatoric = projected_symmetric[1:, 1:]
    deviatoric_eigenvalues = np.linalg.eigvalsh(deviatoric)
    bulk_modulus = float(projected_symmetric[0, 0] / 3.0)
    shear_modulus = float(np.mean(deviatoric_eigenvalues) / 2.0)
    if bulk_modulus <= 0.0 or shear_modulus <= 0.0:
        raise ValueError(
            f"nonpositive projected modulus: K={bulk_modulus}, G={shear_modulus}"
        )

    isotropic_deviatoric = 2.0 * shear_modulus * np.eye(5)
    major_symmetry_error = float(
        np.linalg.norm(tangent - tangent.T)
        / max(1.0, float(np.linalg.norm(tangent)))
    )
    deviatoric_anisotropy = float(
        np.linalg.norm(deviatoric - isotropic_deviatoric)
        / max(float(np.linalg.norm(isotropic_deviatoric)), 1.0e-12)
    )
    volumetric_deviatoric_coupling = float(
        np.linalg.norm(projected_symmetric[0, 1:])
        / max(abs(float(projected_symmetric[0, 0])), 1.0)
    )
    young_modulus = float(
        9.0 * bulk_modulus * shear_modulus
        / (3.0 * bulk_modulus + shear_modulus)
    )
    poisson_ratio = float(
        (3.0 * bulk_modulus - 2.0 * shear_modulus)
        / (2.0 * (3.0 * bulk_modulus + shear_modulus))
    )
    return {
        "bulk_modulus": bulk_modulus,
        "shear_modulus": shear_modulus,
        "young_modulus": young_modulus,
        "poisson_ratio": poisson_ratio,
        "deviatoric_mode_eigenvalues": deviatoric_eigenvalues.tolist(),
        "deviatoric_anisotropy_relative": deviatoric_anisotropy,
        "volumetric_deviatoric_coupling_relative": volumetric_deviatoric_coupling,
        "tangent_major_symmetry_relative_error": major_symmetry_error,
    }


def _rigid_sphere_mori_tanaka(volume_fraction, matrix_bulk, matrix_shear):
    """Mori-Tanaka isotropic moduli for rigid spherical inclusions."""
    phi = float(volume_fraction)
    if not 0.0 < phi < 1.0:
        raise ValueError("volume_fraction must lie strictly between 0 and 1")
    eta = (
        matrix_shear * (9.0 * matrix_bulk + 8.0 * matrix_shear)
        / (6.0 * (matrix_bulk + 2.0 * matrix_shear))
    )
    bulk = matrix_bulk + phi * (
        matrix_bulk + 4.0 * matrix_shear / 3.0
    ) / (1.0 - phi)
    shear = matrix_shear + phi * (matrix_shear + eta) / (1.0 - phi)
    young = 9.0 * bulk * shear / (3.0 * bulk + shear)
    poisson = (3.0 * bulk - 2.0 * shear) / (2.0 * (3.0 * bulk + shear))
    return {
        "bulk_modulus": float(bulk),
        "shear_modulus": float(shear),
        "young_modulus": float(young),
        "poisson_ratio": float(poisson),
        "interpretation": (
            "matrix-based Mori-Tanaka spherical-inclusion estimate in the rigid "
            "particle limit; corresponding Hashin-Shtrikman lower reference "
            "for a stiffer inclusion phase under isotropic assumptions"
        ),
    }


def _picnn_split_scope(split):
    if split == "train":
        return "in_sample_training_record"
    if split == "validation":
        return "validation_record_used_for_model_selection"
    if split == "test":
        return (
            "held_out_from_fitting_but_from_the_existing_phase7_test_split; "
            "not_a_fresh_independent_test"
        )
    return f"unexpected_split:{split}"


def _select_identity_records(data, volume_fractions):
    sample_ids = np.asarray(data["sample_id"]).astype(str)
    gradients = np.asarray(data["deformation_gradient"], dtype=float)
    all_vfs = np.asarray(data["volume_fraction"], dtype=float)
    splits = np.asarray(data["split"]).astype(str)
    tangents = np.asarray(data["tangent"], dtype=float)
    identity = np.eye(3)
    identity_indices = np.flatnonzero(
        np.max(np.abs(gradients - identity), axis=(1, 2)) <= IDENTITY_TOLERANCE
    )
    if not len(identity_indices):
        raise ValueError("dataset contains no undeformed F=I tangent records")

    selected = []
    for target_vf in volume_fractions:
        candidates = identity_indices[
            np.isclose(all_vfs[identity_indices], target_vf, rtol=0.0, atol=1.0e-12)
        ]
        if not len(candidates):
            raise ValueError(f"dataset has no F=I record at volume fraction {target_vf}")
        phase7_candidates = [
            index for index in candidates if sample_ids[index].startswith("phase7_")
        ]
        chosen = phase7_candidates or list(candidates)
        chosen.sort(key=lambda index: sample_ids[index])
        index = chosen[0]
        selected.append(
            {
                "sample_id": str(sample_ids[index]),
                "split": str(splits[index]),
                "state_index": int(data["state_index"][index]),
                "volume_fraction": float(all_vfs[index]),
                "deformation_gradient": gradients[index].tolist(),
                "direct_rve_tangent": tangents[index],
            }
        )
    return selected


def _predict_tangents(checkpoint, selected, torch_threads):
    torch.set_num_threads(int(torch_threads))
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    dtype = torch.float64
    model = HyperelasticNN(n_ctx=1).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    identity = torch.eye(3, dtype=dtype, device=device)
    predicted = []
    with torch.enable_grad():
        for item in selected:
            context = torch.tensor(
                [[item["volume_fraction"]]], dtype=dtype, device=device
            )
            reference = model.reference(dtype, device, context)

            def energy_flat(flat_gradient):
                gradient = flat_gradient.reshape(3, 3)
                return model.energy(
                    gradient.unsqueeze(0), context, ref=reference
                ).squeeze()

            tangent = hessian(energy_flat)(identity.reshape(9))
            tangent_np = tangent.detach().cpu().numpy()
            if tangent_np.shape != (9, 9) or not np.all(np.isfinite(tangent_np)):
                raise FloatingPointError(
                    f"nonfinite pICNN tangent at vf={item['volume_fraction']}"
                )
            predicted.append(tangent_np)
    return predicted


def _plot_moduli(results, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fractions = np.asarray([item["volume_fraction"] for item in results])
    series = (
        ("Direct RVE", "direct_rve", "o"),
        ("Frozen pICNN (in-sample)", "frozen_picnn", "s"),
        ("Mori-Tanaka rigid-sphere", "mori_tanaka_rigid_sphere_reference", "^"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for axis, property_name, ylabel, title in (
        (axes[0], "bulk_modulus", "Effective bulk modulus, K", "Volumetric response"),
        (axes[1], "shear_modulus", "Effective shear modulus, G", "Deviatoric response"),
    ):
        for label, key, marker in series:
            values = [item[key][property_name] for item in results]
            axis.plot(
                100.0 * fractions,
                values,
                marker=marker,
                linewidth=1.8,
                markersize=6,
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("Rigid-particle volume fraction (%)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(100.0 * fractions)
        axis.grid(True, alpha=0.28)
    axes[1].legend(frameon=False, fontsize=8.5)
    figure.suptitle("Small-strain effective moduli at F = I")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_audit(args):
    dataset_path = Path(args.dataset)
    manifest_path = Path(args.manifest)
    checkpoint_path = Path(args.checkpoint)
    report_path = Path(args.output_json)
    csv_path = Path(args.output_csv)

    dataset_sha256 = _sha256_file(dataset_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("dataset SHA-256 does not match its manifest")
    if args.expected_dataset_sha256 and dataset_sha256 != args.expected_dataset_sha256:
        raise ValueError(
            "dataset SHA-256 does not match expected frozen hash: "
            f"actual={dataset_sha256}, expected={args.expected_dataset_sha256}"
        )
    if manifest.get("tangent_flattening") != "row-major: (i,J) -> 3*i + J":
        raise ValueError("unexpected or missing tangent flattening convention")

    checkpoint_sha256 = _sha256_file(checkpoint_path)
    if args.expected_checkpoint_sha256 and checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise ValueError(
            "checkpoint SHA-256 does not match expected frozen hash: "
            f"actual={checkpoint_sha256}, expected={args.expected_checkpoint_sha256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("dataset_sha256") != dataset_sha256:
        raise ValueError("checkpoint training dataset hash does not match the selected dataset")

    required = {
        "sample_id", "split", "state_index", "volume_fraction",
        "deformation_gradient", "tangent",
    }
    with np.load(dataset_path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"dataset is missing fields: {sorted(missing)}")
        data = {name: np.array(archive[name]) for name in required}
    if data["deformation_gradient"].shape != (len(data["sample_id"]), 3, 3):
        raise ValueError("deformation_gradient must have shape (N,3,3)")
    if data["tangent"].shape != (len(data["sample_id"]), 9, 9):
        raise ValueError("tangent must have shape (N,9,9)")
    if not all(
        np.all(np.isfinite(data[name]))
        for name in ("volume_fraction", "deformation_gradient", "tangent")
    ):
        raise ValueError("dataset contains nonfinite numerical values")

    selected = _select_identity_records(data, args.volume_fractions)
    predicted_tangents = _predict_tangents(
        checkpoint, selected, args.torch_threads
    )
    result_rows = []
    for item, predicted_tangent in zip(selected, predicted_tangents):
        phi = item["volume_fraction"]
        direct_metrics = _extract_moduli(item["direct_rve_tangent"])
        picnn_metrics = _extract_moduli(predicted_tangent)
        reference = _rigid_sphere_mori_tanaka(
            phi, args.matrix_bulk_modulus, args.matrix_shear_modulus
        )
        for key in ("bulk_modulus", "shear_modulus", "young_modulus", "poisson_ratio"):
            direct_value = direct_metrics[key]
            picnn_value = picnn_metrics[key]
            reference_value = reference[key]
            reference["direct_rve_" + key + "_relative_difference"] = float(
                abs(direct_value - reference_value)
                / max(abs(direct_value), abs(reference_value), 1.0e-12)
            )
            reference["picnn_" + key + "_relative_difference"] = float(
                abs(picnn_value - reference_value)
                / max(abs(picnn_value), abs(reference_value), 1.0e-12)
            )
        result_rows.append(
            {
                "sample_id": item["sample_id"],
                "split": item["split"],
                "state_index": item["state_index"],
                "volume_fraction": phi,
                "direct_rve": direct_metrics,
                "frozen_picnn": picnn_metrics,
                "mori_tanaka_rigid_sphere_reference": reference,
                "picnn_label_scope": _picnn_split_scope(item["split"]),
            }
        )

    figure_path = _plot_moduli(result_rows, args.output_figure)
    report = {
        "status": "complete",
        "purpose": "small_strain_effective_moduli_micromechanics_benchmark",
        "scope_note": (
            "Moduli are projected from the full row-major dP/dF tangent at F=I. "
            "The direct-RVE values are compared with a matrix-based spherical "
            "rigid-inclusion Mori-Tanaka estimate (the corresponding HS lower "
            "reference for a stiffer inclusion phase under isotropic assumptions). "
            "Each pICNN comparison is labeled by its dataset split. The selected "
            "Phase 7 reference states are in the test split and were excluded "
            "from fitting, but that test set was inspected in earlier project "
            "work, so this is not a fresh independent test."
        ),
        "dataset_path": str(dataset_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_training_dataset_sha256": checkpoint.get("dataset_sha256"),
        "figure_path": str(figure_path.resolve()),
        "matrix_properties": {
            "shear_modulus": float(args.matrix_shear_modulus),
            "bulk_modulus": float(args.matrix_bulk_modulus),
        },
        "volume_fractions": [float(value) for value in args.volume_fractions],
        "strain_basis": (
            "one normalized volumetric mode and five orthonormal symmetric "
            "deviatoric modes; K=C_vol,vol/3 and G=mean(eig(C_dev))/2"
        ),
        "mori_tanaka_reference": {
            "inclusion": "rigid sphere, infinite bulk and shear stiffness limit",
            "matrix_based": True,
            "hashin_shtrikman_relation": (
                "the spherical estimate corresponds to the HS lower reference "
                "for a stiffer inclusion phase under quasi-isotropic assumptions"
            ),
            "references": [
                "Mori and Tanaka, Acta Metallurgica 21 (1973), 571-574, DOI 10.1016/0001-6160(73)90064-3",
                "Hashin and Shtrikman, J. Mech. Phys. Solids 11 (1963), 127-140, DOI 10.1016/0022-5096(63)90060-7",
            ],
        },
        "results": result_rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_path, report)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "sample_id", "split", "volume_fraction", "method",
                "bulk_modulus", "shear_modulus", "young_modulus", "poisson_ratio",
                "deviatoric_anisotropy_relative",
                "volumetric_deviatoric_coupling_relative",
                "tangent_major_symmetry_relative_error",
                "bulk_relative_difference_vs_mori_tanaka",
                "shear_relative_difference_vs_mori_tanaka",
            ]
        )
        for row in result_rows:
            for method_key, method_label in (
                ("direct_rve", "direct_rve"),
                ("frozen_picnn", "frozen_picnn_in_sample"),
            ):
                metrics = row[method_key]
                ref = row["mori_tanaka_rigid_sphere_reference"]
                prefix = "direct_rve" if method_key == "direct_rve" else "picnn"
                writer.writerow(
                    [
                        row["sample_id"], row["split"], row["volume_fraction"],
                        method_label, metrics["bulk_modulus"], metrics["shear_modulus"],
                        metrics["young_modulus"], metrics["poisson_ratio"],
                        metrics["deviatoric_anisotropy_relative"],
                        metrics["volumetric_deviatoric_coupling_relative"],
                        metrics["tangent_major_symmetry_relative_error"],
                        ref[prefix + "_bulk_modulus_relative_difference"],
                        ref[prefix + "_shear_modulus_relative_difference"],
                    ]
                )
            ref = row["mori_tanaka_rigid_sphere_reference"]
            writer.writerow(
                [
                    row["sample_id"], row["split"], row["volume_fraction"],
                    "mori_tanaka_rigid_sphere_reference", ref["bulk_modulus"],
                    ref["shear_modulus"], ref["young_modulus"], ref["poisson_ratio"],
                    "", "", "", 0.0, 0.0,
                ]
            )
    print(f"Small-strain micromechanics JSON: {report_path}", flush=True)
    print(f"Small-strain micromechanics CSV: {csv_path}", flush=True)
    print("Effective moduli by volume fraction:", flush=True)
    for row in result_rows:
        print(
            f"vf={row['volume_fraction']:.3f} "
            f"RVE(K,G)=({row['direct_rve']['bulk_modulus']:.8g},"
            f"{row['direct_rve']['shear_modulus']:.8g}) "
            f"MT(K,G)=({row['mori_tanaka_rigid_sphere_reference']['bulk_modulus']:.8g},"
            f"{row['mori_tanaka_rigid_sphere_reference']['shear_modulus']:.8g})",
            flush=True,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-figure", required=True)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument(
        "--volume-fractions", nargs="+", type=float,
        default=DEFAULT_VOLUME_FRACTIONS,
    )
    parser.add_argument("--matrix-bulk-modulus", type=float, required=True)
    parser.add_argument("--matrix-shear-modulus", type=float, required=True)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.matrix_bulk_modulus <= 0 or args.matrix_shear_modulus <= 0:
        raise ValueError("matrix bulk and shear moduli must be positive")
    if args.torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    run_audit(args)


if __name__ == "__main__":
    main()
