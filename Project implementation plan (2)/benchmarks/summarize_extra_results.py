"""Create compact tables and thesis-ready plots from completed extra runs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_json(path):
    return None if path is None else json.loads(Path(path).read_text(encoding="utf-8"))


def _flat_rows(summaries):
    rows = []
    for group, metrics in summaries.items():
        for name, value in metrics.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            rows.append({
                "result_group": group,
                "metric": name,
                "value": value,
            })
    return rows


def _summarize(inverse, volume_map, benchmark, loading_paths):
    summaries = {}
    if inverse is not None:
        optimization = inverse.get("optimization", {})
        validation = inverse.get("validation", {})
        summaries["inverse_design"] = {
            "status": validation.get("status", inverse.get("status", "unknown")),
            "volume_fraction": optimization.get("volume_fraction"),
            "target_response": inverse.get("target_response"),
            "predicted_response": optimization.get("predicted_response"),
            "verified_response": validation.get("verified_optimum_response"),
            "verified_relative_target_error": validation.get(
                "verified_target_relative_error"
            ),
            "target_error_limit": validation.get("target_error_limit"),
            "checkpoint_sha256": inverse.get("checkpoint_sha256"),
        }
    if volume_map is not None:
        verification = volume_map.get("verification", {})
        validation = volume_map.get("direct_rve_validation", {})
        summaries["volume_fraction_map"] = {
            "status": volume_map.get("status", "unknown"),
            "n_points": len(volume_map.get("surrogate_map", [])),
            "maximum_scaled_derivative_error": verification.get(
                "maximum_scaled_derivative_error"
            ),
            "derivative_tolerance": verification.get("tolerance"),
            "direct_rve_cases": validation.get("n_direct_rve_cases"),
            "energy_rmse_vs_direct_rve": validation.get("energy_rmse"),
            "stress_component_rmse_vs_direct_rve": validation.get(
                "stress_component_rmse"
            ),
            "checkpoint_sha256": volume_map.get("checkpoint_sha256"),
        }
    if benchmark is not None:
        accuracy = benchmark.get("accuracy_vs_frozen_holdout", {})
        runtime = benchmark.get("runtime", {})
        summaries["surrogate_rve_benchmark"] = {
            "status": benchmark.get("status", "unknown"),
            "cases": benchmark.get("n_cases"),
            "energy_nrmse_training_scale": accuracy.get(
                "energy_nrmse_training_scale"
            ),
            "stress_nrmse_training_scale": accuracy.get(
                "stress_nrmse_training_scale"
            ),
            "median_paired_solver_speedup": runtime.get(
                "median_paired_solver_speedup"
            ),
            "median_paired_amortized_speedup": runtime.get(
                "median_paired_amortized_speedup"
            ),
            "holdout_dataset_sha256": benchmark.get("holdout_dataset_sha256"),
            "checkpoint_sha256": benchmark.get("checkpoint_sha256"),
        }
    if loading_paths is not None:
        result = loading_paths.get("summary", {})
        summaries["loading_path_sweep"] = {
            "status": loading_paths.get("status", "unknown"),
            "scope": loading_paths.get("validation_scope"),
            "load_modes": len(loading_paths.get("load_modes", [])),
            "volume_fractions": len(loading_paths.get("volume_fractions", [])),
            "path_points": result.get("n_path_points"),
            "maximum_work_conjugacy_scaled_error": result.get(
                "maximum_work_conjugacy_scaled_error"
            ),
            "maximum_tangent_major_symmetry_error": result.get(
                "maximum_tangent_major_symmetry_error"
            ),
            "checkpoint_sha256": loading_paths.get("checkpoint_sha256"),
        }
    return summaries


def _read_target_sweeps(specifications):
    reports = {}
    paths = {}
    for specification in specifications or []:
        if "=" not in specification:
            raise ValueError(
                "target sweeps must use MODE=PATH, for example "
                "simple_shear=/path/to/report.json"
            )
        mode, path_text = specification.split("=", 1)
        mode = mode.strip()
        path = Path(path_text).expanduser()
        if not mode or mode in reports:
            raise ValueError(f"target-sweep mode is empty or repeated: {mode!r}")
        reports[mode] = _read_json(path)
        paths[mode] = path
    return reports, paths


def _summarize_target_sweeps(reports):
    summaries = {}
    for mode, report in reports.items():
        limit = float(report.get("target_error_limit", 0.05))
        targets = []
        for item in report.get("targets", []):
            optimizer = item.get("optimizer", {})
            in_range = bool(item.get(
                "target_in_predicted_response_range",
                optimizer.get("target_in_predicted_range", False),
            ))
            predicted_error = float(item.get(
                "predicted_relative_target_error", float("inf")
            ))
            direct_error = float(item["direct_rve_relative_target_error"])
            direct_status = "pass" if direct_error <= limit else "target_miss"
            if not in_range:
                design_status = "surrogate_target_out_of_range"
            elif predicted_error <= limit and direct_error <= limit:
                design_status = "pass"
            elif direct_error > limit:
                design_status = "direct_target_miss"
            else:
                design_status = "surrogate_target_miss"
            targets.append({
                "target_fraction_of_endpoint_span": item.get(
                    "target_fraction_of_endpoint_span"
                ),
                "target_response": item.get("target_response"),
                "predicted_volume_fraction": item.get("predicted_volume_fraction"),
                "predicted_response": item.get("predicted_response"),
                "predicted_relative_target_error": predicted_error,
                "predicted_response_range": optimizer.get(
                    "predicted_response_range"
                ),
                "target_in_predicted_response_range": in_range,
                "direct_rve_response": item.get("direct_rve_result", {}).get(
                    "response"
                ),
                "direct_rve_relative_target_error": direct_error,
                "surrogate_to_direct_relative_error": item.get(
                    "surrogate_to_direct_relative_error"
                ),
                "direct_rve_target_status": direct_status,
                "design_status": design_status,
            })
        n_passed = sum(item["design_status"] == "pass" for item in targets)
        n_out_of_range = sum(
            item["design_status"] == "surrogate_target_out_of_range"
            for item in targets
        )
        status = (
            "pass" if n_passed == len(targets) and targets
            else "no_surrogate_feasible_targets"
            if targets and n_passed == 0 and n_out_of_range == len(targets)
            else "partial"
        )
        summaries[mode] = {
            "status": status,
            "response": report.get("response"),
            "stress_component_ij": report.get("stress_component_ij"),
            "n_targets": len(targets),
            "n_surrogate_supported_passed": int(n_passed),
            "n_surrogate_out_of_range": int(n_out_of_range),
            "n_direct_rve_target_passed": int(sum(
                item["direct_rve_target_status"] == "pass" for item in targets
            )),
            "maximum_direct_target_error": max(
                (item["direct_rve_relative_target_error"] for item in targets),
                default=None,
            ),
            "maximum_surrogate_to_direct_error": max(
                (item["surrogate_to_direct_relative_error"] for item in targets),
                default=None,
            ),
            "maximum_scaled_derivative_error": report.get(
                "verification", {}
            ).get("maximum_scaled_derivative_error"),
            "target_error_limit": limit,
            "targets": targets,
            "checkpoint_sha256": report.get("checkpoint_sha256"),
        }
    return summaries


def _plot_loading_paths(report, output_path, volume_map=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = report.get("load_modes", [])
    rows = report.get("load_path", [])
    if not modes or not rows:
        return False
    volume_fractions = report.get("volume_fractions", [])
    colors = plt.get_cmap("viridis")(
        np.linspace(0.12, 0.88, max(1, len(volume_fractions)))
    )
    fig, axes = plt.subplots(
        len(modes),
        2,
        figsize=(11.5, 2.4 * len(modes)),
        squeeze=False,
        constrained_layout=True,
    )
    for mode_index, mode in enumerate(modes):
        mode_records = [item for item in rows if item["load_mode"] == mode]
        for vf_index, vf in enumerate(volume_fractions):
            records = sorted(
                [
                    item for item in mode_records
                    if np.isclose(item["volume_fraction"], vf, rtol=0.0, atol=1.0e-12)
                ],
                key=lambda item: item["amplitude"],
            )
            if not records:
                continue
            amplitudes = [item["amplitude"] for item in records]
            axes[mode_index, 0].plot(
                amplitudes,
                [item["energy"] for item in records],
                color=colors[vf_index],
                linewidth=1.6,
                label=f"vf={vf:.2f}",
            )
            axes[mode_index, 1].plot(
                amplitudes,
                [item["generalized_force_from_stress"] for item in records],
                color=colors[vf_index],
                linewidth=1.6,
                label=f"vf={vf:.2f}",
            )
        axes[mode_index, 0].set_ylabel("Energy density")
        axes[mode_index, 1].set_ylabel("Generalized force")
        axes[mode_index, 0].set_title(mode.replace("_", " "))
        for axis in axes[mode_index]:
            axis.grid(True, alpha=0.25)
            axis.set_xlabel("Loading amplitude")
        axes[mode_index, 0].legend(frameon=False, ncol=3, fontsize=8)

    if volume_map is not None:
        aliases = {
            "uniaxial_tension": "uniaxial_5pct",
            "simple_shear": "simple_shear_5pct",
            "uniaxial_compression": "uniaxial_compression_5pct",
        }
        mode_indices = {mode: index for index, mode in enumerate(modes)}
        for item in volume_map.get("surrogate_map", []):
            mode = next(
                (key for key, value in aliases.items() if value == item["load_case"]),
                None,
            )
            direct = item.get("direct_rve")
            if mode not in mode_indices or direct is None:
                continue
            row_index = mode_indices[mode]
            vf = float(item["volume_fraction"])
            vf_index = int(np.argmin(np.abs(np.asarray(volume_fractions) - vf)))
            color = colors[vf_index]
            axes[row_index, 0].scatter(
                [0.05],
                [direct["energy"]],
                marker="x",
                color=color,
                s=28,
                zorder=5,
            )
            F_np = np.asarray(item["deformation_gradient"], dtype=float)
            if mode == "uniaxial_tension":
                dF = np.diag([1.0, 0.0, 0.0])
            elif mode == "uniaxial_compression":
                dF = np.diag([-1.0, 0.0, 0.0])
            else:
                dF = np.zeros((3, 3))
                dF[0, 1] = 1.0
            direct_force = float(
                np.sum(np.asarray(direct["stress"], dtype=float) * dF)
            )
            axes[row_index, 1].scatter(
                [0.05],
                [direct_force],
                marker="x",
                color=color,
                s=28,
                zorder=5,
            )
    fig.suptitle(
        "Frozen pICNN loading paths; crosses show direct-RVE checks where available",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_volume_map(report, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report.get("surrogate_map", [])
    if not rows:
        return False
    cases = list(dict.fromkeys(item["load_case"] for item in rows))
    fig, axes = plt.subplots(
        len(cases),
        2,
        figsize=(10.5, 3.0 * len(cases)),
        squeeze=False,
        constrained_layout=True,
    )
    for case_index, case in enumerate(cases):
        selected = sorted(
            [item for item in rows if item["load_case"] == case],
            key=lambda item: item["volume_fraction"],
        )
        stress_component = (0, 1) if case == "simple_shear_5pct" else (0, 0)
        stress_label = (
            "P01" if stress_component == (0, 1) else "P00"
        )
        vf = [item["volume_fraction"] for item in selected]
        axes[case_index, 0].plot(
            vf,
            [item["energy"] for item in selected],
            marker="o",
            label="pICNN",
        )
        direct_rows = [item for item in selected if item.get("direct_rve") is not None]
        if direct_rows:
            axes[case_index, 0].scatter(
                [item["volume_fraction"] for item in direct_rows],
                [item["direct_rve"]["energy"] for item in direct_rows],
                marker="x",
                s=45,
                label="direct RVE",
            )
        axes[case_index, 1].plot(
            vf,
            [item["stress"][stress_component[0]][stress_component[1]] for item in selected],
            marker="o",
            label=f"pICNN {stress_label}",
        )
        if direct_rows:
            axes[case_index, 1].scatter(
                [item["volume_fraction"] for item in direct_rows],
                [
                    item["direct_rve"]["stress"][stress_component[0]][stress_component[1]]
                    for item in direct_rows
                ],
                marker="x",
                s=45,
                label=f"direct RVE {stress_label}",
            )
        axes[case_index, 0].set_ylabel("Energy density")
        axes[case_index, 1].set_ylabel(stress_label)
        axes[case_index, 0].set_title(case.replace("_", " "))
        for axis in axes[case_index]:
            axis.set_xlabel("Particle volume fraction")
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_inverse_design(report, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    optimization = report.get("optimization", {})
    grid = optimization.get("search_grid", [])
    if not grid:
        return False
    vf = [item["volume_fraction"] for item in grid]
    response = [item["predicted_response"] for item in grid]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(vf, response, color="#2563A6", linewidth=1.8, label="pICNN")
    ax.axhline(
        float(report["target_response"]),
        color="#D55E00",
        linestyle="--",
        linewidth=1.4,
        label="target",
    )
    ax.axvline(
        float(optimization["volume_fraction"]),
        color="#17804A",
        linestyle=":",
        linewidth=1.4,
        label="selected volume fraction",
    )
    for item in report.get("validation", {}).get("direct_rve_results", []):
        ax.scatter(
            [item["volume_fraction"]],
            [item["response"]],
            color="#6A3D9A",
            marker="x",
            zorder=5,
        )
    ax.set_xlabel("Particle volume fraction")
    ax.set_ylabel(report.get("response", "Response"))
    ax.set_title("Bounded inverse-design response")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_target_sweeps(reports, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = {mode: report for mode, report in reports.items() if report.get("targets")}
    if not reports:
        return False
    fig, axes = plt.subplots(
        len(reports), 1,
        figsize=(8.4, 3.4 * len(reports)),
        squeeze=False,
        constrained_layout=True,
    )
    target_colors = plt.get_cmap("viridis")(
        np.linspace(0.18, 0.82, 3)
    )
    for axis, (mode, report) in zip(axes[:, 0], reports.items()):
        targets = report["targets"]
        first_optimizer = targets[0].get("optimizer", {})
        grid = first_optimizer.get("search_grid", [])
        if grid:
            axis.plot(
                [row["volume_fraction"] for row in grid],
                [row["predicted_response"] for row in grid],
                color="#2563A6",
                linewidth=1.8,
                label="pICNN response",
            )
        for endpoint in report.get("direct_rve_endpoint_results", []):
            axis.scatter(
                [endpoint["volume_fraction"]],
                [endpoint["response"]],
                marker="D",
                color="#111111",
                s=35,
                zorder=5,
                label="direct-RVE endpoint",
            )
        for index, item in enumerate(targets):
            color = target_colors[index % len(target_colors)]
            fraction = float(item["target_fraction_of_endpoint_span"])
            axis.axhline(
                float(item["target_response"]),
                color=color,
                linestyle="--",
                alpha=0.75,
                linewidth=1.2,
                label=f"{fraction:.0%} target",
            )
            axis.scatter(
                [item["predicted_volume_fraction"]],
                [item["predicted_response"]],
                marker="o",
                color=color,
                s=34,
                zorder=6,
                label=f"{fraction:.0%} pICNN candidate",
            )
            direct = item.get("direct_rve_result", {})
            if direct:
                axis.scatter(
                    [item["predicted_volume_fraction"]],
                    [direct["response"]],
                    marker="x",
                    color=color,
                    s=45,
                    zorder=7,
                    label=f"{fraction:.0%} direct RVE",
                )
        axis.set_title(mode.replace("_", " "))
        axis.set_xlabel("Particle volume fraction")
        axis.set_ylabel(report.get("response", "Response"))
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=7, ncol=3)
        if any(
            not item.get("target_in_predicted_response_range", True)
            for item in targets
        ):
            axis.text(
                0.99,
                0.02,
                "At least one target is outside the pICNN response range",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#9C2F21",
            )
    fig.suptitle(
        "Multi-target inverse design: pICNN predictions and direct-RVE checks",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def build_summary(args):
    target_sweeps, target_sweep_paths = _read_target_sweeps(args.target_sweep)
    reports = {
        "inverse_design": _read_json(args.inverse_design),
        "volume_fraction_map": _read_json(args.volume_map),
        "surrogate_rve_benchmark": _read_json(args.benchmark),
        "loading_path_sweep": _read_json(args.loading_paths),
    }
    if not any(report is not None for report in reports.values()) and not target_sweeps:
        raise ValueError("provide at least one completed result report")
    checkpoint_hashes = {
        report.get("checkpoint_sha256")
        for report in list(reports.values()) + list(target_sweeps.values())
        if report is not None and report.get("checkpoint_sha256")
    }
    if len(checkpoint_hashes) > 1:
        raise ValueError(
            "input reports use different checkpoints; summarize them separately"
        )
    summaries = _summarize(
        reports["inverse_design"],
        reports["volume_fraction_map"],
        reports["surrogate_rve_benchmark"],
        reports["loading_path_sweep"],
    )
    if target_sweeps:
        summaries["inverse_design_loading_modes"] = _summarize_target_sweeps(
            target_sweeps
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "extra_results_summary.json"
    json_path.write_text(
        json.dumps({
            "checkpoint_sha256": (
                next(iter(checkpoint_hashes)) if checkpoint_hashes else None
            ),
            "scope_note": (
                "Reported values retain the units and normalization of the "
                "underlying frozen checkpoint and RVE configuration."
            ),
            "summaries": summaries,
            "source_reports": {
                key: None if value is None else str(value.resolve())
                for key, value in (
                    ("inverse_design", args.inverse_design),
                    ("volume_fraction_map", args.volume_map),
                    ("surrogate_rve_benchmark", args.benchmark),
                    ("loading_path_sweep", args.loading_paths),
                )
            },
            "source_target_sweeps": {
                mode: str(path.resolve())
                for mode, path in target_sweep_paths.items()
            },
        }, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "extra_results_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("result_group", "metric", "value"))
        writer.writeheader()
        writer.writerows(_flat_rows(summaries))

    generated_figures = []
    if not args.no_figure:
        if reports["loading_path_sweep"] is not None:
            path = output_dir / "loading_path_curves.png"
            if _plot_loading_paths(
                reports["loading_path_sweep"],
                path,
                reports["volume_fraction_map"],
            ):
                generated_figures.append(str(path.resolve()))
        if reports["volume_fraction_map"] is not None:
            path = output_dir / "volume_fraction_map.png"
            if _plot_volume_map(reports["volume_fraction_map"], path):
                generated_figures.append(str(path.resolve()))
        if reports["inverse_design"] is not None:
            path = output_dir / "inverse_design_curve.png"
            if _plot_inverse_design(reports["inverse_design"], path):
                generated_figures.append(str(path.resolve()))
        if target_sweeps:
            path = output_dir / "inverse_design_loading_modes.png"
            if _plot_target_sweeps(target_sweeps, path):
                generated_figures.append(str(path.resolve()))
    print(json.dumps({
        "summary_json": str(json_path.resolve()),
        "summary_csv": str(csv_path.resolve()),
        "figures": generated_figures,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inverse-design", type=Path)
    parser.add_argument("--volume-map", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--loading-paths", type=Path)
    parser.add_argument(
        "--target-sweep",
        action="append",
        metavar="MODE=PATH",
        help=(
            "multi-target inverse-design report; repeat for multiple modes, "
            "for example simple_shear=/path/to/report.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    build_summary(args)


if __name__ == "__main__":
    main()
