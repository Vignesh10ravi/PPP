"""Break a completed surrogate/RVE benchmark down by stress component.

The audit reads saved benchmark JSON only. It performs no solver calls and
does not change or select a model. Component-relative metrics are reported
alongside errors normalized by the original global training stress scale.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _metrics(error, labels, training_scale):
    error = np.asarray(error, dtype=float)
    labels = np.asarray(labels, dtype=float)
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    label_rms = float(np.sqrt(np.mean(labels**2)))
    return {
        "rmse": rmse,
        "mae": mae,
        "holdout_label_rms": label_rms,
        "nrmse_component_label_rms": (
            rmse / label_rms if label_rms > 0.0 else None
        ),
        "nrmse_global_training_scale": rmse / training_scale,
    }


def build_audit(args):
    report = json.loads(args.benchmark.read_text(encoding="utf-8"))
    cases = report.get("cases", [])
    if not cases:
        raise ValueError("benchmark report has no per-case tensor data")
    accuracy = report.get("accuracy_vs_frozen_holdout", {})
    global_rmse = float(accuracy["stress_component_rmse"])
    global_nrmse = float(accuracy["stress_nrmse_training_scale"])
    if global_nrmse <= 0.0:
        raise ValueError("benchmark report has an invalid global stress NRMSE")
    training_scale = global_rmse / global_nrmse

    aggregate = []
    by_volume_fraction = []
    csv_rows = []
    volume_fractions = sorted({float(case["volume_fraction"]) for case in cases})
    for vf in [None, *volume_fractions]:
        selected = cases if vf is None else [
            case for case in cases
            if np.isclose(
                float(case["volume_fraction"]), vf, rtol=0.0, atol=1.0e-12
            )
        ]
        for i in range(3):
            for j in range(3):
                surrogate_errors = []
                direct_errors = []
                labels = []
                for case in selected:
                    label = np.asarray(case["holdout_stress_label"], dtype=float)
                    surrogate_error = np.asarray(
                        case["surrogate_vs_holdout_stress_error"], dtype=float
                    )
                    direct_error = np.asarray(
                        case["direct_rve_vs_holdout_stress_error"], dtype=float
                    )
                    labels.append(label[i, j])
                    surrogate_errors.append(surrogate_error[i, j])
                    direct_errors.append(direct_error[i, j])
                surrogate_metrics = _metrics(
                    surrogate_errors, labels, training_scale
                )
                direct_metrics = _metrics(direct_errors, labels, training_scale)
                row = {
                    "volume_fraction": "all" if vf is None else vf,
                    "component": f"P{i}{j}",
                    "n_cases": len(selected),
                    "surrogate_rmse": surrogate_metrics["rmse"],
                    "surrogate_mae": surrogate_metrics["mae"],
                    "holdout_label_rms": surrogate_metrics["holdout_label_rms"],
                    "surrogate_nrmse_component_label_rms": surrogate_metrics[
                        "nrmse_component_label_rms"
                    ],
                    "surrogate_nrmse_global_training_scale": surrogate_metrics[
                        "nrmse_global_training_scale"
                    ],
                    "direct_rve_rmse": direct_metrics["rmse"],
                    "direct_rve_mae": direct_metrics["mae"],
                    "direct_rve_nrmse_component_label_rms": direct_metrics[
                        "nrmse_component_label_rms"
                    ],
                    "direct_rve_nrmse_global_training_scale": direct_metrics[
                        "nrmse_global_training_scale"
                    ],
                }
                (aggregate if vf is None else by_volume_fraction).append(row)
                csv_rows.append(row)

    output = {
        "status": "descriptive_component_audit_complete",
        "scope": (
            "Post-processing of saved held-out benchmark cases only; no solver "
            "calls, model fitting, or model selection. Component NRMSE uses the "
            "RMS of that component's holdout labels; global-scale NRMSE uses the "
            "checkpoint's global training stress scale recovered from the stored "
            "benchmark aggregate."
        ),
        "benchmark_path": str(args.benchmark.resolve()),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "holdout_dataset_sha256": report.get("holdout_dataset_sha256"),
        "n_cases": len(cases),
        "global_training_stress_scale": training_scale,
        "global_stress_nrmse_training_scale": global_nrmse,
        "all_volume_fractions": aggregate,
        "by_volume_fraction": by_volume_fraction,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({
        "report": str(args.output.resolve()),
        "csv": str(csv_path.resolve()),
        "cases": len(cases),
        "global_training_stress_scale": training_scale,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_audit(args)


if __name__ == "__main__":
    main()
