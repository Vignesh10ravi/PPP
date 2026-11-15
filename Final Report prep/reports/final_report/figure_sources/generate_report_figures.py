from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Values are copied from the archived V3 and V4 verification logs.
epsilon = np.array([1e-2, 1e-4, 1e-6, 1e-8], dtype=float)
fd_error = np.array([1.25e-5, 1.25e-9, 1.69e-10, 1.42e-8], dtype=float)
newton_iteration = np.arange(5)
newton_residual = np.array([9.058e-1, 1.043e-1, 3.463e-3, 6.412e-6, 4.023e-11])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), constrained_layout=True)

ax = axes[0]
order = np.argsort(epsilon)
ax.loglog(epsilon[order], fd_error[order], marker="o", linewidth=1.7,
          markersize=5, color="#2563A6")
ax.scatter([1e-6], [1.69e-10], s=50, color="#D55E00", zorder=4,
           label=r"minimum observed error ($\epsilon=10^{-6}$)")
ax.set_xlabel(r"Central-difference perturbation, $\epsilon$")
ax.set_ylabel("Relative tangent error")
ax.set_title("(a) Tangent finite-difference diagnostic")
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="best", frameon=False)

ax = axes[1]
ax.semilogy(newton_iteration, newton_residual, marker="o", linewidth=1.7,
           markersize=5, color="#17804A")
ax.set_xlabel("Newton update")
ax.set_ylabel(r"Free residual norm, $\|R_f\|_2$")
ax.set_title("(b) NGSolve equilibrium convergence")
ax.set_xticks(newton_iteration)
ax.grid(True, which="both", alpha=0.25)
for x, y in zip(newton_iteration, newton_residual):
    offset = (10, -13) if x == 0 else (4, 5)
    ax.annotate(f"{y:.1e}", (x, y), xytext=offset, textcoords="offset points", fontsize=11)

fig.savefig(OUT / "phase1_verification_diagnostics.pdf", bbox_inches="tight")
fig.savefig(OUT / "phase1_verification_diagnostics.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Matched neural/analytical FE comparison values from the archived V6 log.
labels = ["Free displacement\nfield", "Deformation\ngradient", "Reaction\nvector"]
relative_error_pct = np.array([1.376, 0.3504, 13.60])
acceptance_pct = 15.0
fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
bars = ax.bar(labels, relative_error_pct,
              color=["#5689B8", "#5689B8", "#D55E00"], width=0.62)
ax.axhline(acceptance_pct, color="#333333", linestyle="--", linewidth=1.3,
           label="Acceptance limit (15%)")
ax.set_ylabel("Relative error (%)")
ax.set_title("Matched analytical-versus-neural FE response")
ax.set_ylim(0, 17)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(frameon=False, loc="upper left")
ax.tick_params(axis="x", labelrotation=0)
for bar, value in zip(bars, relative_error_pct):
    ax.annotate(f"{value:g}%", (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=11)
fig.savefig(OUT / "phase1_matched_fe_errors.pdf", bbox_inches="tight")
fig.savefig(OUT / "phase1_matched_fe_errors.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# This is the schema-1 300-record Phase-2 pilot. Plot one determinant per
# deformation state in the coverage panel; the other panel shows all records.
with np.load(HERE / "phase2_pilot.npz", allow_pickle=False) as archive:
    data = {name: archive[name] for name in archive.files}
state_ids = np.unique(data["state_index"])
state_determinants = np.array([
    data["determinant"][np.flatnonzero(data["state_index"] == state)[0]]
    for state in state_ids
])
record_residual = np.maximum.reduce((
    data["mixed_residual"],
    data["rotation_residual"],
    data["condensed_residual"],
))

fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), constrained_layout=True)
ax = axes[0]
ax.hist(state_determinants, bins=16, color="#5689B8", edgecolor="white", linewidth=0.6)
ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1.1, label=r"reference $J=1$")
ax.set_xlabel(r"Macroscopic determinant, $\det(\overline{F})$")
ax.set_ylabel("Count of states")
ax.set_title("(a) Baseline loading-state coverage")
ax.legend(frameon=False)
ax.grid(True, axis="y", alpha=0.25)

ax = axes[1]
colors = ["#0072B2", "#D55E00", "#009E73"]
for color, vf in zip(colors, np.sort(np.unique(data["volume_fraction"]))):
    mask = data["volume_fraction"] == vf
    ax.scatter(data["state_index"][mask], record_residual[mask], s=16, alpha=0.78,
               color=color, label=fr"$v_f={vf:g}$")
ax.axhline(1e-8, color="#333333", linestyle="--", linewidth=1.1,
           label=r"acceptance limit $10^{-8}$")
ax.set_yscale("log")
ax.set_xlabel("Deformation-state index")
ax.set_ylabel("Maximum stored residual metric")
ax.set_title("(b) RVE solve acceptance checks")
ax.grid(True, which="both", alpha=0.25)
ax.legend(frameon=True, framealpha=0.95, ncol=2, loc="upper right", fontsize=11)
fig.savefig(OUT / "phase2_pilot_coverage_and_residuals.pdf", bbox_inches="tight")
fig.savefig(OUT / "phase2_pilot_coverage_and_residuals.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Created figures in {OUT}")
