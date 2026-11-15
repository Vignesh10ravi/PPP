"""Build a grouped Phase 6 + Phase 7 training dataset for pICNN fitting.

Phase 6 test records are omitted. Phase 7's original training and validation
splits are retained, and its test records remain test-only. When the same
macroscopic deformation occurs in both sources, all retained records for that
deformation are assigned to the most restrictive split (test > validation >
train) to prevent split leakage.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


FIELDS = (
    "sample_id",
    "split",
    "state_index",
    "volume_fraction",
    "deformation_gradient",
    "energy",
    "stress",
    "tangent",
)
SPLIT_PRIORITY = {"train": 0, "validation": 1, "test": 2}
COMPATIBLE_CONFIG_KEYS = (
    "volume_fractions",
    "mu",
    "bulk_modulus",
    "maxh",
    "displacement_order",
    "integration_order",
    "microscopic_jacobian_floor",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_source(dataset_path, manifest_path, label):
    dataset_path = Path(dataset_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256_file(dataset_path)
    if manifest.get("dataset_sha256") != digest:
        raise ValueError(f"{label} dataset hash does not match its manifest")

    with np.load(dataset_path, allow_pickle=False) as archive:
        missing = set(FIELDS).difference(archive.files)
        if missing:
            raise ValueError(f"{label} dataset is missing fields: {sorted(missing)}")
        arrays = {name: np.array(archive[name]) for name in FIELDS}

    n_samples = len(arrays["sample_id"])
    if int(manifest.get("n_samples", -1)) != n_samples:
        raise ValueError(f"{label} manifest sample count does not match the NPZ")
    if arrays["deformation_gradient"].shape != (n_samples, 3, 3):
        raise ValueError(f"{label} deformation gradients must have shape (N,3,3)")
    if arrays["stress"].shape != (n_samples, 3, 3):
        raise ValueError(f"{label} stresses must have shape (N,3,3)")
    if arrays["tangent"].shape != (n_samples, 9, 9):
        raise ValueError(f"{label} tangents must have shape (N,9,9)")
    if not all(
        np.all(np.isfinite(value))
        for value in arrays.values()
        if np.issubdtype(value.dtype, np.number)
    ):
        raise ValueError(f"{label} dataset contains nonfinite numerical values")

    arrays["sample_id"] = arrays["sample_id"].astype(str)
    arrays["split"] = arrays["split"].astype(str)
    arrays["state_index"] = arrays["state_index"].astype(np.int64)
    if not set(np.unique(arrays["split"])).issubset(SPLIT_PRIORITY):
        raise ValueError(f"{label} dataset has unknown split labels")

    # Verify each original deformation group has a consistent split and F.
    for state in np.unique(arrays["state_index"]):
        indices = np.flatnonzero(arrays["state_index"] == state)
        if len(np.unique(arrays["split"][indices])) != 1:
            raise ValueError(f"{label} state {state} crosses original splits")
        if not np.allclose(
            arrays["deformation_gradient"][indices],
            arrays["deformation_gradient"][indices[0]],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(f"{label} state {state} has inconsistent gradients")
    return arrays, manifest, digest


def _deformation_key(gradient):
    # Rounding merges numerically identical F records without conflating
    # physically distinct states at the precision used by the dataset.
    return np.round(np.asarray(gradient, dtype=np.float64), decimals=12).tobytes()


def merge_training_data(
    phase6_dataset,
    phase6_manifest,
    phase7_dataset,
    phase7_manifest,
    output_dataset,
    output_manifest,
):
    phase6, manifest6, hash6 = _read_source(
        phase6_dataset, phase6_manifest, "Phase 6"
    )
    phase7, manifest7, hash7 = _read_source(
        phase7_dataset, phase7_manifest, "Phase 7"
    )

    config6 = manifest6.get("config", {})
    config7 = manifest7.get("config", {})
    for key in COMPATIBLE_CONFIG_KEYS:
        if key in config6 and key in config7 and config6[key] != config7[key]:
            raise ValueError(
                f"Phase 6 and Phase 7 have incompatible {key}: "
                f"{config6[key]!r} versus {config7[key]!r}"
            )

    groups = {}
    excluded_phase6_test_records = 0
    for label, arrays in (("phase6", phase6), ("phase7", phase7)):
        for index, original_split in enumerate(arrays["split"]):
            if label == "phase6" and original_split == "test":
                excluded_phase6_test_records += 1
                continue
            gradient = arrays["deformation_gradient"][index]
            key = _deformation_key(gradient)
            group = groups.setdefault(
                key,
                {
                    "gradient": gradient,
                    "split": original_split,
                    "records": [],
                },
            )
            if not np.allclose(
                group["gradient"], gradient, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError("deformation fingerprint collision detected")
            if SPLIT_PRIORITY[original_split] > SPLIT_PRIORITY[group["split"]]:
                group["split"] = original_split
            group["records"].append((label, index))

    ordered_keys = sorted(groups)
    records = []
    for new_state, key in enumerate(ordered_keys):
        group = groups[key]
        for label, index in group["records"]:
            source = phase6 if label == "phase6" else phase7
            records.append(
                {
                    "source": label,
                    "index": index,
                    "state_index": new_state,
                    "split": group["split"],
                    "sample_id": f"{label}_{source['sample_id'][index]}",
                }
            )

    if not records:
        raise ValueError("no records remain after applying the split policy")
    records.sort(
        key=lambda item: (
            SPLIT_PRIORITY[item["split"]],
            item["state_index"],
            float(
                (phase6 if item["source"] == "phase6" else phase7)[
                    "volume_fraction"
                ][item["index"]]
            ),
            item["source"],
            item["sample_id"],
        )
    )
    if len({item["sample_id"] for item in records}) != len(records):
        raise ValueError("prefixed sample IDs are not unique")

    merged = {
        "sample_id": np.asarray([item["sample_id"] for item in records]),
        "split": np.asarray([item["split"] for item in records]),
        "state_index": np.asarray(
            [item["state_index"] for item in records], dtype=np.int64
        ),
    }
    for field in FIELDS:
        if field in {"sample_id", "split", "state_index"}:
            continue
        source_arrays = []
        for item in records:
            source = phase6 if item["source"] == "phase6" else phase7
            source_arrays.append(source[field][item["index"]])
        merged[field] = np.asarray(source_arrays)

    n_states = len(ordered_keys)
    split_counts = {
        split: int(np.count_nonzero(merged["split"] == split))
        for split in SPLIT_PRIORITY
    }
    split_state_counts = {
        split: int(
            len(np.unique(merged["state_index"][merged["split"] == split]))
        )
        for split in SPLIT_PRIORITY
    }
    if any(split_counts[name] == 0 for name in ("train", "validation", "test")):
        raise ValueError(f"merged dataset has an empty split: {split_counts}")

    output_dataset = Path(output_dataset)
    output_manifest = Path(output_manifest)
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_dataset = output_dataset.with_name(f".{output_dataset.name}.tmp")
    with temporary_dataset.open("wb") as stream:
        np.savez_compressed(stream, **merged)
    os.replace(temporary_dataset, output_dataset)
    output_hash = sha256_file(output_dataset)

    config = {
        "dataset_kind": "Phase 6 + Phase 7 grouped pICNN training dataset",
        "phase6_dataset_sha256": hash6,
        "phase7_dataset_sha256": hash7,
        "phase6_split_policy": "include train and validation; exclude test",
        "phase7_split_policy": "retain train, validation, and test assignments",
        "cross_source_grouping": "round deformation_gradient to 12 decimals",
        "conflict_policy": "test > validation > train",
        "n_deformation_states": n_states,
        "split_counts": split_counts,
        "split_state_counts": split_state_counts,
        "excluded_phase6_test_records": excluded_phase6_test_records,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    manifest = {
        "n_samples": len(records),
        "n_deformation_states": n_states,
        "dataset_sha256": output_hash,
        "configuration_sha256": config_hash,
        "config": config,
        "split_counts": split_counts,
        "split_state_counts": split_state_counts,
        "fields": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in merged.items()
        },
        "tangent_flattening": "row-major: (i,J) -> 3*i + J",
        "split_scope_note": (
            "Phase 7 test records are excluded from fitting and kept as the "
            "test split. Phase 6 test records are omitted from this composite. "
            "For a new independent claim on the retrained model, generate a "
            "fresh grouped holdout because the Phase 7 set was previously "
            "evaluated with the prior checkpoint."
        ),
    }
    temporary_manifest = output_manifest.with_name(f".{output_manifest.name}.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, output_manifest)

    print(f"Merged dataset saved: {output_dataset}")
    print(f"Manifest saved: {output_manifest}")
    print(f"Records: {len(records)}; grouped deformation states: {n_states}")
    print(f"Split records: {split_counts}")
    print(f"Split deformation states: {split_state_counts}")
    print(f"Excluded Phase 6 test records: {excluded_phase6_test_records}")
    print(f"Dataset SHA-256: {output_hash}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase6-dataset", type=Path, required=True)
    parser.add_argument("--phase6-manifest", type=Path, required=True)
    parser.add_argument("--phase7-dataset", type=Path, required=True)
    parser.add_argument("--phase7-manifest", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    merge_training_data(
        args.phase6_dataset,
        args.phase6_manifest,
        args.phase7_dataset,
        args.phase7_manifest,
        args.output_dataset,
        args.output_manifest,
    )


if __name__ == "__main__":
    main()
