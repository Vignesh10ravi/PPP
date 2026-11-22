"""Regenerate all data figures of the report from the files in figure_sources/data.

Run from the report root:  python figure_sources/make_figures.py
Outputs vector PDFs into figures/.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT.parent / "figures"
OUT.mkdir(exist_ok=True)

# Validated categorical order (blue, orange, aqua, yellow); text in neutral ink.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
})


def load(name):
    return json.load(open(DATA / name))


def panel_label(ax, s):
    ax.set_title(s, loc="left", fontweight="bold", color=INK)


# ---------------------------------------------------------------- Phase 1
def fig_phase1():
    d = load("phase1_logs.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.6))
    fig.subplots_adjust(wspace=0.38)
    v3 = d["v3_tangent_fd"]
    import numpy as _np
    eps_v = _np.array(v3["eps"], float); err_v = _np.array(v3["rel_err_rerun"], float)
    a.loglog(eps_v, err_v, "o-", color=C1, label="central differences vs. AD")
    eg = _np.array([1e-2, 1e-5])
    a.loglog(eg, 0.15 * err_v[0] * (eg / eg[0]) ** 2, ":", color=INK2, lw=1.0, label=r"slope $\epsilon^2$")
    a.set_xlabel(r"central-difference step $\epsilon$")
    a.set_ylabel("relative tangent error")
    a.invert_xaxis()
    a.legend(loc="upper right")
    panel_label(a, "(a) V3: tangent check")
    r = d["v4_newton_checkpoint"]
    b.semilogy(range(len(r)), r, "o-", color=C1, label="trained Phase 1 network")
    r2 = d["v4_newton_seeded"]
    b.semilogy(range(len(r2)), r2, "s--", color=C2, label="seeded (untrained) network")
    b.axhline(1e-8, color=INK2, lw=0.8, ls=":")
    b.text(0.05, 2e-8, "tolerance $10^{-8}$", color=INK2, fontsize=7)
    b.set_xlabel("Newton iteration $k$")
    b.set_ylabel(r"$\|\mathbf{R}_\mathrm{free}\|_2$")
    b.set_xticks(range(5))
    b.legend(loc="lower left", fontsize=7)
    panel_label(b, "(b) V4: Newton residual")
    fig.savefig(OUT / "fig_phase1_diagnostics.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- training coverage
def fig_training_coverage():
    d = load("training_J_histogram.json")
    e = np.array(d["edges"])
    fig, ax = plt.subplots(figsize=(6.3, 2.3))
    ax.bar(0.5 * (e[1:] + e[:-1]), d["counts"], width=np.diff(e) * 0.9, color=C1)
    ax.axvspan(0.98, 1.02, color=C2, alpha=0.25, lw=0)
    ax.annotate(r"near-isochoric band $|J-1|<0.02$", xy=(1.0, max(d["counts"]) * 0.75),
                xytext=(0.62, max(d["counts"]) * 1.02), color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6), fontsize=8)
    ax.set_xlabel(r"macroscopic volume ratio $J=\det\overline{\boldsymbol{F}}$")
    ax.set_ylabel("records")
    ax.text(0.99, 0.95, f"{d['n']} records (all splits)\nenergy RMS = {d['energy_rms']:.0f}, "
            f"stress RMS = {d['stress_rms']:.0f}\n($\\mu=1$, $\\kappa=10^4$)", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color=INK2)
    fig.savefig(OUT / "fig_training_coverage.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- split errors
def fig_split_errors():
    rows = list(csv.DictReader(open(DATA / "phase7_record_errors.csv")))
    J = np.array([float(r["J"]) for r in rows])
    pe = np.array([float(r["p_err_abs"]) / max(abs(float(r["p_ref"])), 1e-12) for r in rows])
    de = np.array([float(r["dev_err_norm"]) / max(float(r["dev_ref_norm"]), 1e-12) for r in rows])
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.8), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(wspace=0.35)
    a.scatter(J, 100 * de, s=9, color=C2, marker="^", label="deviatoric stress", lw=0)
    a.scatter(J, 100 * pe, s=9, color=C1, marker="o", label="pressure", lw=0)
    a.set_yscale("log")
    a.set_xlabel(r"$J=\det\overline{\boldsymbol{F}}$")
    a.set_ylabel("relative error per record (%)")
    a.legend(loc="upper center", markerscale=1.5, bbox_to_anchor=(0.5, -0.2), fontsize=7, ncol=2)
    panel_label(a, "(a) per record (frozen model)")
    sm = load("split_metrics.json")["phase7_independent"]["models"]
    groups = ["compression J<0.7", "intermediate", "near-isochoric |J-1|<0.02", "expansion J>1.4"]
    labels = ["$J<0.7$", "intermediate", r"$|J-1|<0.02$", "$J>1.4$"]
    x = np.arange(len(groups))
    w = 0.38
    fz = [sm["frozen"]["groups"][g]["deviatoric_rel_err_median_pct"] for g in groups]
    sh = [sm["shear_adapted"]["groups"][g]["deviatoric_rel_err_median_pct"] for g in groups]
    b.bar(x - w / 2 - 0.01, fz, w, color=C2, label="frozen")
    b.bar(x + w / 2 + 0.01, sh, w, color=C3, label="shear-adapted")
    for xi, v in zip(x - w / 2, fz):
        b.text(xi, v * 1.08, f"{v:.0f}", ha="center", fontsize=6.2, color=INK, rotation=90, va="bottom")
    for xi, v in zip(x + w / 2, sh):
        b.text(xi, v * 1.08, f"{v:.0f}", ha="center", fontsize=6.2, color=INK, rotation=90, va="bottom")
    b.set_yscale("log")
    b.set_ylim(10, 20000)
    b.set_xticks(x, labels, fontsize=6.5, rotation=30, ha="right")
    b.set_ylabel("median deviatoric error (%)")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, fontsize=7)
    panel_label(b, "(b) by loading regime")
    fig.savefig(OUT / "fig_split_errors.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- moduli
def fig_moduli():
    mm = load("rve_small_strain_micromechanics.json")["results"]
    sm = load("split_metrics.json")["small_strain_moduli"]
    vf = np.array([r["volume_fraction"] for r in mm])
    G_rve = [r["direct_rve"]["shear_modulus"] for r in mm]
    G_mt = [r["mori_tanaka_rigid_sphere_reference"]["shear_modulus"] for r in mm]
    K_rve = [r["direct_rve"]["bulk_modulus"] for r in mm]
    K_mt = [r["mori_tanaka_rigid_sphere_reference"]["bulk_modulus"] for r in mm]
    keys = ["0.02", "0.05", "0.10"]
    G_fz = [sm["frozen"][k]["mu_eff"] for k in keys]
    G_sh = [sm["shear_adapted"][k]["mu_eff"] for k in keys]
    K_fz = [sm["frozen"][k]["K_eff"] for k in keys]
    K_sh = [sm["shear_adapted"][k]["K_eff"] for k in keys]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.6))
    fig.subplots_adjust(wspace=0.35)
    for ax, ys, unit in ((a, (G_rve, G_mt, G_fz, G_sh), r"$\mu_\mathrm{eff}/\mu$"),
                         (b, (np.array(K_rve) / 1e4, np.array(K_mt) / 1e4, np.array(K_fz) / 1e4,
                              np.array(K_sh) / 1e4), r"$K_\mathrm{eff}/\kappa$")):
        ax.plot(vf, ys[0], "o-", color=C1, label="direct RVE")
        ax.plot(vf, ys[1], "k:", lw=1.2, label="Mori–Tanaka (rigid spheres)")
        ax.plot(vf, ys[2], "^-", color=C2, label="pICNN frozen")
        ax.plot(vf, ys[3], "s-", color=C3, label="pICNN shear-adapted")
        ax.set_xlabel(r"volume fraction $v_\mathrm{f}$")
        ax.set_ylabel(unit)
        ax.set_xticks(vf)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.34)
    panel_label(a, "(a) shear modulus at $F=I$")
    panel_label(b, "(b) bulk modulus at $F=I$")
    fig.savefig(OUT / "fig_small_strain_moduli.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- shear holdout
def fig_shear_holdout():
    pp = load("shear_final_holdout_report.json")["paired_points"]
    ref = np.array([p["P01_direct_rve"] for p in pp])
    fz = np.array([p["P01_frozen"] for p in pp])
    sh = np.array([p["P01_candidate"] for p in pp])
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    lim = [0, 1.05 * max(fz.max(), ref.max())]
    ax.plot(lim, lim, color=INK2, lw=0.8, ls="--", label="exact")
    ax.scatter(ref, fz, color=C2, marker="^", s=22, label="frozen", zorder=3)
    ax.scatter(ref, sh, color=C3, marker="s", s=20, label="shear-adapted", zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"direct RVE $\overline{P}_{01}$")
    ax.set_ylabel(r"surrogate $\overline{P}_{01}$")
    ax.legend(loc="upper left")
    fig.savefig(OUT / "fig_shear_holdout.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- macro FE
def fig_macro():
    m = load("macro_steps.json")
    p1 = {0.25: "rve_phase7_macro_fe_order1_energy_secant12",
          0.20: "rve_phase7_macro_fe_order1_energy_secant40_h020_mem128",
          0.16: "rve_phase7_macro_fe_order1_energy_secant40_h016_mem128"}
    p2 = {0.25: "rve_phase7_macro_fe_order2_meshstudy_1pct_h025_16c",
          0.20: "rve_phase7_macro_fe_order2_meshstudy_1pct_h020_16c",
          0.16: "rve_phase7_macro_fe_order2_meshstudy_0p5pct_h016_16c"}
    marks = {0.25: "o", 0.20: "s", 0.16: "^"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.9))
    fig.subplots_adjust(wspace=0.32)
    for h, run in p1.items():
        st = m[run]["steps"]
        a.loglog([s["stretch"] for s in st], [s["energy"] for s in st], marks[h] + "-", color=C1,
                 ms=3, markevery=10, lw=1.2, label=f"P1, $h={h}$")
        b.plot([s["stretch"] * 100 for s in st], [len(s["res"]) - 1 for s in st], marks[h], color=C1,
               ms=3, label=f"P1, $h={h}$")
    for h, run in p2.items():
        st = m[run]["steps"]
        a.loglog([s["stretch"] for s in st], [s["energy"] for s in st], marks[h] + "--", color=C2,
                 ms=3, lw=1.2, label=f"P2, $h={h}$")
        b.plot([s["stretch"] * 100 for s in st], [len(s["res"]) - 1 for s in st], marks[h], color=C2,
               ms=3, mfc="none", label=f"P2, $h={h}$")
    eps = np.logspace(np.log10(5e-4), np.log10(5e-2), 20)
    for mu, lab, ls in ((1.124, r"free uniaxial, $\mu_\mathrm{RVE}$", ":"),
                        (1.731, r"free uniaxial, $\mu_\mathrm{pICNN}$", "-.")):
        a.loglog(eps, 1.5 * mu * eps ** 2, ls, color=INK2, lw=0.9, label=lab)
    a.set_xlabel(r"axial stretch $\varepsilon$")
    a.set_ylabel(r"stored energy $\int_{\Omega}\Psi\,\mathrm{d}V$")
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6.8, bbox_to_anchor=(0.5, -0.2))
    panel_label(a, "(a) energy along the load path")
    b.set_xlabel(r"axial stretch $\varepsilon$ (%)")
    b.set_ylabel("Newton iterations per load step")
    b.set_xscale("log")
    panel_label(b, "(b) nonlinear solver effort")
    fig.savefig(OUT / "fig_macro_energy.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- timing
def fig_timing():
    rows = list(csv.DictReader(open(DATA / "rve_surrogate_rve_benchmark.csv")))
    fig, ax = plt.subplots(figsize=(6.3, 1.9))
    for i, (vf, c, mk) in enumerate((("0.02", C1, "o"), ("0.05", C2, "s"), ("0.1", C3, "^"))):
        t = [float(r["direct_rve_seconds_median"]) for r in rows if r["volume_fraction"] == vf]
        ax.scatter(t, np.full(len(t), 1 + 0.12 * (i - 1)), color=c, marker=mk, s=18,
                   label=f"direct RVE, $v_f={float(vf):.2f}$")
    ts = [float(r["surrogate_seconds_median"]) for r in rows]
    ax.scatter(ts, np.zeros(len(ts)), color=INK2, marker="D", s=14, label="pICNN (energy + stress)")
    ax.set_xscale("log")
    ax.set_yticks([0, 1], ["surrogate", "direct RVE"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("wall time per constitutive evaluation (s), median of repeats, 1 thread")
    ax.legend(loc="center", fontsize=7, ncol=2)
    fig.savefig(OUT / "fig_timing.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- training history
def fig_training_curves():
    h = load("training_histories.json")
    tl = load("test_losses.json")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.6, 2.9))
    fig.subplots_adjust(wspace=0.45, bottom=0.30, top=0.88)
    l1 = h["phase1_train_loss"]
    a.semilogy(range(1, len(l1) + 1), l1, color=C1, lw=1.0, label="training")
    a.set_xlabel("epoch")
    a.set_ylabel("loss")
    panel_label(a, "(a) Phase 1 (neo-Hooke)")
    p6 = h["phase6"]
    b.semilogy(p6["epoch"], p6["train"], color=C1, label="training")
    b.semilogy(p6["epoch"], p6["val"], "--", color=C2, label="validation")
    b.set_xlabel("epoch")
    b.set_ylabel(r"$\mathcal{L}_\Psi+\mathcal{L}_P$")
    panel_label(b, "(b) Phase 6 training")
    fz = h["frozen"]
    c.semilogy(fz["epoch"], fz["train"], color=C1, label="training")
    c.semilogy(fz["epoch"], fz["val"], "--", color=C2, label="validation")
    c.semilogy(fz["epoch"], fz["val_energy"], ":", color=C3, label="val. energy term")
    be = fz["best_epoch"]
    c.axvline(be, color=INK2, lw=0.6, ls=":")
    c.scatter([be], [tl["phase7_independent"]["frozen"]["loss_energy_plus_stress"]], marker="D", color=INK, s=18,
              zorder=4, label="test (Phase 7)")
    c.scatter([be], [tl["fresh_holdout"]["frozen"]["loss_energy_plus_stress"]], marker="s", color=C4, s=18,
              zorder=4, label="test (fresh)")
    c.set_xlabel("epoch")
    c.set_ylabel(r"$\mathcal{L}_\Psi+\mathcal{L}_P$")
    c.set_xlim(right=be * 1.12)
    from matplotlib.ticker import LogLocator, NullFormatter
    c.set_yticks([2e-7, 5e-7, 1e-6, 2e-6, 4e-6])
    c.set_yticklabels([r"$2\cdot10^{-7}$", r"$5\cdot10^{-7}$", r"$10^{-6}$", r"$2\cdot10^{-6}$", r"$4\cdot10^{-6}$"])
    c.yaxis.set_minor_formatter(NullFormatter())
    hs, ls = c.get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=5, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 0.0))
    panel_label(c, "(c) Phase 6/7 fine-tune")
    fig.savefig(OUT / "fig_training_curves.pdf")
    plt.close(fig)


def fig_parity():
    rows = list(csv.DictReader(open(DATA / "parity_phase7_independent.csv")))
    g = lambda k: np.array([float(r[k]) for r in rows])
    psi_r, psi, p_r, p, d_r, d = g("psi_ref"), g("psi"), g("p_ref"), g("p"), g("dev_ref"), g("dev")
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.6))
    fig.subplots_adjust(wspace=0.42, top=0.8)
    specs = ((axs[0], psi_r, psi, r"$\overline{\Psi}$", "log"), (axs[1], p_r, p, r"$p$", "symlog"),
             (axs[2], d_r, d, r"$\Vert s\Vert$", "log"))
    for ax, x, y, lab, sc in specs:
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        if sc == "log":
            lo = max(lo, 1e-4 if ax is axs[2] else 1e-3)
            m = (x > 0) & (y > 0); x, y = x[m], y[m]
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.7, ls="--")
        ax.scatter(x, y, s=5, color=C2 if ax is axs[2] else C1, lw=0)
        if sc == "log":
            ax.set_xscale("log"); ax.set_yscale("log")
        elif sc == "symlog":
            ax.set_xscale("symlog", linthresh=10); ax.set_yscale("symlog", linthresh=10)
            from matplotlib.ticker import FixedLocator, NullLocator
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(FixedLocator([-1e4, -1e2, 0, 1e2, 1e4]))
                axis.set_minor_locator(NullLocator())
        ax.set_xlabel("direct RVE " + lab)
        ax.set_ylabel("pICNN " + lab)
    rel = lambda a, b: 100 * np.median(np.abs(a - b) / np.maximum(np.abs(a), 1e-12))
    for ax, t, v in ((axs[0], "(a) energy", f"{rel(psi_r, psi):.2f}"), (axs[1], "(b) pressure", f"{rel(p_r, p):.3f}"),
                     (axs[2], "(c) deviator", f"{rel(d_r, d):.0f}")):
        panel_label(ax, f"{t}\nmedian rel. error {v} %")
    fig.savefig(OUT / "fig_parity.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- RVE mesh convergence
def fig_rve_convergence():
    c = load("rve_convergence.json")
    qty = (("energy_relative", "energy", 2e-3), ("stress_relative", "stress", 5e-3),
           ("tangent_relative", "tangent", 1e-2))
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.4), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    for k, (ax, (key, lab, thr)) in enumerate(zip(axs, qty)):
        for vf, col in ((0.02, C1), (0.1, C2)):
            for q, mk, ls in ((6, "o", "-"), (10, "s", "--")):
                rows = [r for r in c["results"] if f"vf={vf}|" in r["key"] and r["key"].endswith(f"q={q}")]
                hs = [0.3, 0.22] + ([0.16] if q == 6 else [])
                emax = [max(max(r["errors_from_finest"][key], 1e-16) for r in rows if r["result"]["maxh"] == h)
                        for h in hs]
                ax.semilogy(hs, emax, mk + ls, color=col, mfc="white" if q == 10 else col, ms=4.5,
                            label=rf"$v_\mathrm{{f}}={vf:g}$, $q={q}$")
        ax.axhline(thr, color=INK2, ls=":", lw=1.0, label="accuracy target")
        ax.axvline(0.22, color=GRID, lw=3, zorder=0)
        ax.set_xlim(0.31, 0.15); ax.set_xticks([0.30, 0.22, 0.16])
        ax.set_xlabel(r"mesh size $h$")
        ax.set_ylim(1e-14, 5e-2)
        panel_label(ax, f"({'abc'[k]}) {lab}")
    axs[0].set_ylabel("max. relative error, 8 load cases")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.2))
    fig.savefig(OUT / "fig_rve_convergence.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- volume-fraction dependence
def fig_vf_dependence():
    vm = load("volume_fraction_map.json")
    mt = {"K": {0.02: 10204.0, 0.05: 10526.0, 0.10: 11111.0}, "mu": {0.02: 1.051, 0.05: 1.132, 0.10: 1.278}}
    cases = (("uniaxial_5pct", (0, 0), r"uniaxial tension, $\overline{P}_{11}$", "K"),
             ("uniaxial_compression_5pct", (0, 0), r"uniaxial compression, $|\overline{P}_{11}|$", "K"),
             ("simple_shear_5pct", (0, 1), r"simple shear, $\overline{P}_{12}$", "mu"))
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.3), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    for k, (ax, (case, (i, j), title, mod)) in enumerate(zip(axs, cases)):
        pts = sorted([p for p in vm if p["load_case"] == case], key=lambda p: p["volume_fraction"])
        vf = np.array([p["volume_fraction"] for p in pts])
        nn = np.array([abs(p["stress"][i][j]) for p in pts])
        dr = np.array([abs(p["direct_rve"]["stress"][i][j]) for p in pts])
        ref = np.array([mt[mod][round(v, 2)] for v in vf])
        ax.plot(vf, dr / dr[0], "o-", color=C1, label="direct RVE")
        ax.plot(vf, nn / nn[0], "s--", color=C2, label="pICNN")
        ax.plot(vf, ref / ref[0], "^:", color=INK2, ms=4,
                label=("Mori--Tanaka " + (r"$K$" if mod == "K" else r"$\mu$")))
        ax.set_xlabel(r"volume fraction $v_\mathrm{f}$")
        panel_label(ax, f"({'abc'[k]}) " + ("tension", "compression", "shear")[k])
        ax.set_xticks([0.02, 0.05, 0.10])
    axs[0].set_ylabel(r"response / response at $v_\mathrm{f}=0.02$")
    fig.legend(*axs[2].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.17))
    fig.savefig(OUT / "fig_vf_dependence.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- five loading paths
MODES = (("uniaxial_tension", "uniaxial tension", (0, 0)),
         ("uniaxial_compression", "uniaxial compression", (0, 0)),
         ("simple_shear", "simple shear", (0, 1)),
         ("equibiaxial_tension", "equibiaxial tension", (0, 0)),
         ("volumetric_expansion", "volumetric expansion", (0, 0)))
VFC = {0.02: C1, 0.05: C2, 0.1: C3}


def direct_path_points():
    """Archived direct-RVE points (5 % map, shear path) plus the additional path solves."""
    pts = []
    for p in load("volume_fraction_map.json"):
        mode = {"uniaxial_5pct": "uniaxial_tension", "uniaxial_compression_5pct": "uniaxial_compression",
                "simple_shear_5pct": "simple_shear"}[p["load_case"]]
        pts.append(dict(mode=mode, vf=round(p["volume_fraction"], 2), amp=0.05,
                        W_rve=p["direct_rve"]["energy"], P_rve=p["direct_rve"]["stress"],
                        W_nn=p["energy"], P_nn=p["stress"]))
    for p in load("shear_path_validation.json"):
        if abs(p["shear_amplitude"] - 0.05) < 1e-9:
            continue
        pts.append(dict(mode="simple_shear", vf=round(p["volume_fraction"], 2), amp=p["shear_amplitude"],
                        W_rve=p["energy_direct_rve"], P_rve=p["stress_direct_rve"],
                        W_nn=p["energy_pICNN"], P_nn=p["stress_pICNN"]))
    f = DATA / "direct_paths.json"
    if f.exists():
        for p in json.load(open(f)):
            pts.append(dict(mode=p["mode"], vf=round(p["vf"], 2), amp=p["amp"], W_rve=p["energy_rve"],
                            P_rve=p["stress_rve"], W_nn=p["energy_nn"], P_nn=p["stress_nn"]))
    return pts


def fig_load_paths():
    lp = load("loading_path_sweep.json")
    dp = direct_path_points()
    fig, axs = plt.subplots(3, 5, figsize=(6.6, 5.0))
    fig.subplots_adjust(wspace=1.0, hspace=0.55)
    for c, (mode, title, (i, j)) in enumerate(MODES):
        for vf, col in VFC.items():
            pts = sorted([p for p in lp if p["load_mode"] == mode and abs(p["volume_fraction"] - vf) < 1e-9],
                         key=lambda p: p["amplitude"])
            a = np.array([p["amplitude"] for p in pts]) * 100
            axs[0, c].plot(a, [p["energy"] for p in pts], "-", color=col, lw=1.2)
            axs[1, c].plot(a, [p["stress"][i][j] for p in pts], "-", color=col, lw=1.2)
            d = sorted([p for p in dp if p["mode"] == mode and p["vf"] == round(vf, 2)], key=lambda p: p["amp"])
            if d:
                ad = np.array([p["amp"] for p in d]) * 100
                axs[0, c].plot(ad, [p["W_rve"] for p in d], "o", color=col, mfc="white", ms=4, mew=1.1)
                axs[1, c].plot(ad, [p["P_rve"][i][j] for p in d], "o", color=col, mfc="white", ms=4, mew=1.1)
                er = [abs(p["P_nn"][i][j] - p["P_rve"][i][j]) / abs(p["P_rve"][i][j]) for p in d]
                axs[2, c].semilogy(ad, er, "o-", color=col, ms=3.5, lw=0.9)
        comp = "11" if (i, j) == (0, 0) else "12"
        axs[0, c].set_title(title.replace(" ", "\n", 1), fontsize=8, color=INK)
        axs[1, c].set_ylabel(r"$\overline{P}_{" + comp + "}$", labelpad=1)
        axs[2, c].set_xlabel("amplitude (%)")
        axs[2, c].set_ylim(3e-5, 3)
        for r in range(3):
            axs[r, c].set_xlim(0, 10.5)
            axs[r, c].tick_params(labelsize=6.5)
    axs[0, 0].set_ylabel(r"$\overline{\Psi}$")
    axs[2, 0].set_ylabel(r"rel. error of $\overline{P}_{ij}$")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=VFC[v], lw=2.0, label=rf"pICNN, $v_\mathrm{{f}}={v:g}$") for v in VFC]
    h += [Line2D([], [], color=INK2, marker="o", mfc="white", ls="none", label="direct RVE")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.align_ylabels(axs)
    fig.savefig(OUT / "fig_load_paths.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- bottlenecks
def fig_bottlenecks():
    rows = list(csv.DictReader(open(DATA / "rve_surrogate_rve_benchmark.csv")))
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 6.4))
    fig.subplots_adjust(wspace=0.42, hspace=0.95)
    a, b, c, d = axs.ravel()
    # (a) offline: RVE solve time vs. distance from the reference state
    for vf, col, mk in ((0.02, C1, "o"), (0.05, C2, "s"), (0.1, C3, "^")):
        rr = [r for r in rows if abs(float(r["volume_fraction"]) - vf) < 1e-9]
        a.plot([abs(float(r["determinant"]) - 1) for r in rr], [float(r["direct_rve_seconds_median"]) for r in rr],
               mk, color=col, ms=4, label=rf"$v_\mathrm{{f}}={vf:g}$")
    a.set_xlabel(r"$|\overline{J}-1|$ of the holdout state")
    a.set_ylabel("direct RVE wall time (s)")
    a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, handletextpad=0.2, columnspacing=0.8)
    panel_label(a, "(a) offline: RVE solve")
    # (b) online: stage shares of one load increment
    cfg = (("bridge_h0.25_p1.json", r"$P_1$, $h{=}0.25$"), ("bridge_h0.16_p1.json", r"$P_1$, $h{=}0.16$"),
           ("bridge_h0.25_p2.json", r"$P_2$, $h{=}0.25$"))
    groups = (("AD tangent", ("2 pICNN energy, stress, tangent (AD)",), C1),
              ("write tangent", ("3 write tangent A to quadrature points",), C2),
              ("stiffness assembly", ("5 stiffness assembly",), C3),
              ("other", None, "#b9b8b3"))
    for k, (f, lab) in enumerate(cfg):
        bj = load(f); tot = bj["wall_seconds"]; left = 0.0
        named = sum(v["seconds"] for kk, v in bj["stages"].items()
                    if kk.startswith(("2", "3", "5")))
        for g, keys, col in groups:
            val = (sum(bj["stages"][kk]["seconds"] for kk in keys) if keys else tot - named) / tot * 100
            b.barh(k, val, left=left, color=col, height=0.6, edgecolor="white", lw=1.0,
                   label=g if k == 0 else None)
            if val > 12:
                b.text(left + val / 2, k, f"{val:.0f}%", ha="center", va="center", fontsize=7, color="white")
            left += val
    b.set_yticks(range(len(cfg))); b.set_yticklabels([c_[1] for c_ in cfg], fontsize=7)
    b.invert_yaxis(); b.set_xlim(0, 100); b.set_xlabel("share of wall time (%)")
    b.grid(axis="y", visible=False)
    b.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2, handlelength=1.0, columnspacing=0.8)
    panel_label(b, "(b) online: one load increment")
    # (c) macro: cumulative Newton iterations along the load path
    m = load("macro_steps.json")
    runs = (("rve_phase7_macro_fe_order1_energy_secant12", r"$P_1$, $h=0.25$", C1, "-"),
            ("rve_phase7_macro_fe_order1_energy_secant40_h020_mem128", r"$P_1$, $h=0.20$", C1, "--"),
            ("rve_phase7_macro_fe_order1_energy_secant40_h016_mem128", r"$P_1$, $h=0.16$", C1, ":"),
            ("rve_phase7_macro_fe_order2_meshstudy_1pct_h025_16c", r"$P_2$, $h=0.25$", C2, "-"),
            ("rve_phase7_macro_fe_order2_meshstudy_1pct_h020_16c", r"$P_2$, $h=0.20$", C2, "--"),
            ("rve_phase7_macro_fe_order2_meshstudy_0p5pct_h016_16c", r"$P_2$, $h=0.16$", C2, ":"))
    for run, lab, col, ls in runs:
        st = m[run]["steps"]
        it = np.cumsum([len(s["res"]) - 1 for s in st])
        c.plot([s["stretch"] * 100 for s in st], it, ls, color=col, lw=1.3, label=lab)
    c.set_xlabel(r"axial stretch $\varepsilon$ (%)"); c.set_ylabel("cumulative Newton iterations")
    c.set_xscale("log"); c.set_yscale("log")
    c.legend(fontsize=6.3, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25), handlelength=1.6, columnspacing=0.8); c.set_ylim(2, 6e4)
    panel_label(c, "(c) macro: nonlinear iterations")
    # (d) cost budget: surrogate (offline + online) vs. direct RVE per evaluation
    n = np.logspace(1, 8, 200)
    t_rve, t_off, t_nn = 134.0, 1671 * 134.0, 36e-6
    d.loglog(n, t_rve * n / 3600, "-", color=C2, label="direct RVE at every point (FE$^2$)")
    d.loglog(n, (t_off + t_nn * n) / 3600, "-", color=C1, label="pICNN: dataset + evaluations")
    for x, lab in ((1671, "break-even"), (18354, "1 iteration\n$P_1$, $h=0.25$"),
                   (18354 * 2005, "full $P_1$ run\n(2005 it.)")):
        d.axvline(x, color=INK2, lw=0.7, ls=":")
        d.text(x * (1.25 if x == 18354 else 0.8), 3e6, lab, fontsize=6.3, color=INK2, rotation=90, va="top",
               ha=("left" if x == 18354 else "right"),
               bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))
    d.set_xlabel("constitutive evaluations"); d.set_ylabel("single-core hours")
    d.set_ylim(1e-3, 1e7)
    d.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    panel_label(d, "(d) cost budget")
    fig.savefig(OUT / "fig_bottlenecks.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- RVE fields
def fig_rve_fields():
    d = load("rve_fields.json")
    x = np.array(d["x"]); X, Y = np.meshgrid(x, x)
    r = d["radius"]
    titles = {"uniaxial_5pct": r"uniaxial tension, $\overline{F}_{11}=1.05$",
              "shear_5pct": r"simple shear, $\overline{F}_{12}=0.05$"}
    fig, axs = plt.subplots(2, 3, figsize=(6.4, 4.3))
    fig.subplots_adjust(wspace=0.45, hspace=0.38)
    th = np.linspace(0, 2 * np.pi, 200)
    for row, (name, cs) in enumerate(d["cases"].items()):
        F = np.array(cs["F"]); R = np.array(cs["rotation"])
        w = np.array(cs["w"]); p = np.array(cs["p"]); dev = np.array(cs["dev"])
        # deformed positions in the mid-plane (z = 0.5): x + (F - I)(x - c) + w
        xc, yc = X - 0.5, Y - 0.5
        Xd = 0.5 + F[0, 0] * xc + F[0, 1] * yc + w[..., 0]
        Yd = 0.5 + F[1, 0] * xc + F[1, 1] * yc + w[..., 1]
        wm = np.linalg.norm(w, axis=2)
        pm = p - np.nanmean(p)
        fields = ((wm * 1e3, r"$|\boldsymbol{w}|$ ($10^{-3}$)", "viridis"),
                  (pm, r"$p-\langle p\rangle$", "RdBu_r"),
                  (dev, r"$\|\mathrm{dev}\,\bar{\boldsymbol{C}}\|$", "magma"))
        for col, (f, lab, cmap) in enumerate(fields):
            ax = axs[row, col]
            m = np.isfinite(f)
            if cmap == "RdBu_r":
                v = np.nanmax(np.abs(f)); lev = np.linspace(-v, v, 21)
            else:
                lev = np.linspace(np.nanmin(f), np.nanmax(f), 21)
            cf = ax.contourf(np.where(m, Xd, np.nan), np.where(m, Yd, np.nan), np.where(m, f, np.nan),
                             levels=lev, cmap=cmap)
            # rigid particle boundary after the rigid motion
            px, py = r * np.cos(th), r * np.sin(th)
            qx = 0.5 + R[0, 0] * px + R[0, 1] * py + 0 * F[0, 0]
            qy = 0.5 + R[1, 0] * px + R[1, 1] * py
            ax.fill(qx, qy, color="#d9d8d4", ec=INK2, lw=0.6)
            ax.set_aspect("equal"); ax.grid(False)
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.tick_params(labelsize=6.5)
            cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
            from matplotlib.ticker import MaxNLocator, FormatStrFormatter
            cb.locator = MaxNLocator(5); cb.formatter = FormatStrFormatter("%.2f" if col else "%.1f")
            cb.update_ticks(); cb.ax.tick_params(labelsize=6)
            ax.set_title(lab, fontsize=8, color=INK)
        axs[row, 0].set_ylabel(titles[name], fontsize=7.5)
    fig.savefig(OUT / "fig_rve_fields.pdf")
    plt.close(fig)


# ================================================================ verification figures
VTESTS = [  # (id, type, key value or None, note)
    ("V0a", "M", 2.2e-16, None), ("V0b", "M", 4.0e-14, None), ("V0c", "M", 2.2e-14, None),
    ("V0d", "M", 1.5e-9, None), ("V0e", "M", None, "18 zero modes (6 rigid + 12 hourglass)"),
    ("V1", "C", None, "interfaces, shapes, dtype: pass"), ("V2", "C", None, r"$\Psi(\mathbf{I})=0$, $\mathbf{P}(\mathbf{I})=\mathbf{0}$ exactly"),
    ("V2b", "S", 5.37e-2, None), ("V3", "M", 2.3e-10, None), ("V4", "M", 4.0e-11, None),
    ("V5", "M", 2.7e-16, None), ("V6", "S", 1.34e-1, None), ("V7", "M", 5.5e-15, None),
    ("V8", "M", 2.0e-15, None), ("P2-V1", "C", 2.2e-16, None), ("P2-V2", "C", 7.2e-18, None),
    ("P2-V3", "M", 2.9e-13, None), ("P2-V4", "C", 1.9e-14, None), ("P2-V5", "S", 1.2e-15, None),
    ("P2-V6", "M", 3.0e-16, None), ("P2-V7", "C", 2.5e-16, None), ("P2-V8", "C", 1.2e-15, None),
    ("P2-V9", "M", 5.2e-12, None), ("P2-V10", "C", None, "0 split leaks, regeneration error 0"),
    ("S-V1", "M", 4.9e-8, None), ("M-V0", "M", 6.5e-10, None)]
VLABEL = {"V0a": "partition of unity", "V0b": "linear field, gradient", "V0c": "volume, area (Nanson)",
          "V0d": "rigid-mode span", "V2b": "pICNN vs. neo-Hooke (energy)", "V3": "AD vs. central diff.",
          "V4": "final Newton residual", "V5": "patch residual", "V6": "reaction vs. analytical",
          "V7": "Gauss-point stress", "V8": "stretch error", "P2-V1": "periodic jump",
          "P2-V2": "rigid constraint", "P2-V3": "pressure error", "P2-V4": "RVE residual",
          "P2-V5": "Hill--Mandel", "P2-V6": "tangent symmetry", "P2-V7": r"$\|\mathbf{R}^T\mathbf{R}-\mathbf{I}\|$",
          "P2-V8": "gauge stress change", "P2-V9": "tangent vs. re-solved FD", "S-V1": "max. audit error",
          "M-V0": "directional tangent"}


def fig_v_overview():
    col = {"M": C1, "S": C2, "C": C3}
    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    n = len(VTESTS)
    for k, (tid, typ, val, note) in enumerate(VTESTS):
        y = n - 1 - k
        if val is None:
            ax.text(1.5e-18, y, note, va="center", fontsize=6.8, color=INK2, style="italic")
            ax.plot(1e-18, y, "D", color=col[typ], ms=3.5)
        else:
            ax.hlines(y, val, 1.0, color=GRID, lw=1.2, zorder=1)
            ax.plot(val, y, "o", color=col[typ], ms=5, zorder=3)
            ax.text(val / 2.2, y, VLABEL.get(tid, ""), va="center", ha="right", fontsize=6.3, color=INK2)
    ax.axvline(2.22e-16, color=INK2, ls=":", lw=0.9)
    ax.text(2.6e-16, n - 0.4, "machine $\\varepsilon$", fontsize=6.5, color=INK2)
    ax.set_yticks(range(n)); ax.set_yticklabels([t[0] for t in VTESTS][::-1], fontsize=7)
    ax.set_xscale("log"); ax.set_xlim(1e-24, 1.0); ax.set_ylim(-0.7, n - 0.1)
    ax.set_xticks([1e-20, 1e-16, 1e-12, 1e-8, 1e-4, 1e0])
    ax.set_xlabel("achieved error measure of the test (relative or absolute, see Table 7.1)")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="none", color=col[t], label=l) for t, l in
         (("M", "mandatory (guideline)"), ("S", "standard benchmark"), ("C", "custom"))]
    ax.legend(handles=h, loc="upper center", fontsize=7, ncol=3, bbox_to_anchor=(0.45, 1.07))
    fig.savefig(OUT / "fig_v_overview.pdf")
    plt.close(fig)


def _cube_axes(ax):
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=22, azim=-58)
    ax.set_xlim(-0.05, 1.25); ax.set_ylim(-0.05, 1.1); ax.set_zlim(-0.05, 1.1)


def fig_v_setup():
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    tm = load("test_meshes.json")
    fig = plt.figure(figsize=(6.4, 3.4))
    titles = ("(a) V0d/V0e: single elements", "(b) V4, V6: stretched block",
              "(c) V5: affine patch", "(d) V7/V8: one hexahedron")
    for k in range(4):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d"); _cube_axes(ax)
        ax.set_title(titles[k], fontsize=7.5, color=INK, pad=-2)
        if k == 0:
            V = np.array(tm["tet"]["V"])
            faces = [[V[i] for i in f] for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))]
            ax.add_collection3d(Poly3DCollection(faces, facecolor=C1, alpha=0.25, edgecolor=INK2, lw=0.6))
            H = np.array(tm["hex"]["V"]) * 0.45 + np.array([0.8, 0.55, 0.0])
            ax.scatter(*V.T, s=6, color=C1); ax.scatter(*H.T, s=6, color=C2)
            hexf = [[H[i] for i in f] for f in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4))]
            ax.add_collection3d(Poly3DCollection(hexf, facecolor=C2, alpha=0.18, edgecolor=INK2, lw=0.6))
            ax.text2D(0.5, 0.02, "free: 6 rigid modes", transform=ax.transAxes, ha="center", fontsize=6.5, color=INK2)
            continue
        if k in (1, 2):
            m = tm["v4" if k == 1 else "v56"]; V = np.array(m["V"])
            tri = [[V[i] for i in t] for t in m["T"]]
            cols = []
            for t in m["T"]:
                xs = V[t, 0]
                if k == 1 and np.all(xs < 1e-9): cols.append(C1)
                elif k == 1 and np.all(xs > 1 - 1e-9): cols.append(C2)
                elif k == 2: cols.append(C3)
                else: cols.append("#f2f1ee")
            ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolor=INK2, lw=0.25, alpha=0.9))
            if k == 1:
                ax.quiver(1.0, 0.5, 0.5, 0.22, 0, 0, color=C2, lw=1.2, arrow_length_ratio=0.35)
                ax.text2D(0.5, 0.02, f"{m['ne']} / {tm['v56']['ne']} tets; $u_x=0.1x$ on $x=0,1$",
                          transform=ax.transAxes, ha="center", fontsize=6.3, color=INK2)
            else:
                ax.text2D(0.5, 0.02, rf"{m['ne']} tets; $\mathbf{{u}}=(\mathbf{{F}}-\mathbf{{I}})\mathbf{{X}}$ on $\partial\Omega$",
                          transform=ax.transAxes, ha="center", fontsize=6.3, color=INK2)
            continue
        # (d) unit hexahedron with traction on x = 1 and symmetry supports
        c = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
        f = [[c[i] for i in q] for q in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4))]
        fc = ["#f2f1ee", "#f2f1ee", "#f2f1ee", "#f2f1ee", C2, C1]
        ax.add_collection3d(Poly3DCollection(f, facecolors=fc, edgecolor=INK2, lw=0.6, alpha=0.6))
        for yy in (0.25, 0.75):
            for zz in (0.25, 0.75):
                ax.quiver(1.0, yy, zz, 0.25, 0, 0, color=C2, lw=1.0, arrow_length_ratio=0.35)
        ax.text2D(0.5, 0.02, r"V7: $\mathbf{u}=(\overline{\mathbf{F}}-\mathbf{I})\mathbf{X}$;  V8: traction $t$",
                  transform=ax.transAxes, ha="center", fontsize=6.3, color=INK2)
    fig.subplots_adjust(wspace=0.0, left=0.0, right=1.0, top=0.9, bottom=0.05)
    fig.savefig(OUT / "fig_v_setup.pdf")
    plt.close(fig)


def fig_v0_spectra():
    d = load("verification_data.json")["v0"]
    cases = (("tet1", r"(a) tetrahedron, $P_1$"), ("hex1", r"(b) hexahedron, $P_1$"), ("hex2", r"(c) hexahedron, $P_2$"))
    mods = (("neo-Hookean (analytical)", C1, "o"), ("RVE pICNN", C2, "s"), ("Phase 1 pICNN", C3, "^"))
    fig, axs = plt.subplots(1, 4, figsize=(6.5, 2.4))
    fig.subplots_adjust(wspace=0.45)
    for ax, (key, title) in zip(axs, cases):
        for name, col, mk in mods:
            lam = np.array(d[f"{key}|{name}"])
            ax.plot(np.arange(1, len(lam) + 1), lam, mk, color=col, ms=2.6, mfc="white" if name.startswith("Phase") else col,
                    label=name)
        ax.set_yscale("symlog", linthresh=1e-6); ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold", color=INK)
        ax.set_xlabel("eigenvalue index", fontsize=7.5); ax.tick_params(labelsize=6.5)
        ax.set_yticks([-1e-2, 0, 1e-4, 1]); ax.set_ylim(-0.1, 1.5)
    axs[0].set_ylabel(r"$\lambda_i/\max|\lambda|$")
    lam = np.array(d["hex1_onepoint|neo-Hookean (analytical)"])
    axs[3].plot(np.arange(1, len(lam) + 1), lam, "o", color=C1, ms=2.6)
    axs[3].set_yscale("symlog", linthresh=1e-6); axs[3].axhline(0, color=INK2, lw=0.6)
    axs[3].axvspan(0.5, 18.5, color=C4, alpha=0.15, lw=0)
    axs[3].text(9.5, 0.3, "18 zero\nmodes", ha="center", fontsize=6.5, color=INK2)
    axs[3].set_title("(d) hex $P_1$, 1-point", loc="left", fontsize=8, fontweight="bold", color=INK)
    axs[3].set_xlabel("eigenvalue index", fontsize=7.5); axs[3].tick_params(labelsize=6.5)
    axs[3].set_yticks([-1e-2, 0, 1e-4, 1]); axs[3].set_ylim(-0.1, 1.5)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.2))
    fig.savefig(OUT / "fig_v0_spectra.pdf")
    plt.close(fig)


def fig_v_phase1():
    d = load("verification_data.json"); lg = load("verification_logs.json")
    v = d["v2b"]
    fig, axs = plt.subplots(1, 4, figsize=(6.5, 2.35))
    fig.subplots_adjust(wspace=0.6)
    a, b, c, e = axs
    pe, pn = np.array(v["psi_exact"]), np.array(v["psi_nn"])
    a.plot(pe, pn, "o", color=C1, ms=2.2, alpha=0.7); lim = [0, pe.max() * 1.05]
    a.plot(lim, lim, "-", color=INK2, lw=0.8); a.set_xlabel(r"$\Psi$ neo-Hooke"); a.set_ylabel(r"$\Psi$ pICNN")
    panel_label(a, "(a) V2b energy")
    Pe, Pn = np.array(v["P_exact"]), np.array(v["P_nn"])
    b.plot(Pe[:, [0, 4, 8]].ravel(), Pn[:, [0, 4, 8]].ravel(), "o", color=C1, ms=1.8, alpha=0.6, label="normal")
    b.plot(Pe[:, [1, 2, 3, 5, 6, 7]].ravel(), Pn[:, [1, 2, 3, 5, 6, 7]].ravel(), "o", color=C2, ms=1.8, alpha=0.6, label="shear")
    lim = [Pe.min(), Pe.max()]; b.plot(lim, lim, "-", color=INK2, lw=0.8)
    b.set_xlabel(r"$P_{ij}$ neo-Hooke"); b.set_ylabel(r"$P_{ij}$ pICNN"); b.legend(fontsize=6, loc="upper left")
    panel_label(b, "(b) V2b stress")
    fd = lg["v7_fd"]
    for name, col, mk in (("St. Venant-Kirchhoff", C1, "o"), ("neo-Hookean", C2, "s")):
        c.loglog(fd["eps"], fd[name], mk + "-", color=col, ms=3.5, label=name.replace("St. Venant-Kirchhoff", "SVK"))
    eg = np.array([1e-3, 1e-5]); c.loglog(eg, 8.49e-6 * (eg / 1e-3) ** 2 * 0.25, ":", color=INK2, lw=0.9, label=r"$\epsilon^2$")
    c.invert_xaxis(); c.set_xlabel(r"step $\epsilon$"); c.set_ylabel("rel. tangent error"); c.legend(fontsize=6, loc="lower left")
    panel_label(c, "(c) V7 tangent")
    v8 = d["v8"]; t = np.array(v8["t"])
    for key, lab, col, ls in (("svk", "SVK", C1, "-"), ("neohooke", "neo-Hooke", C2, "-"), ("phase1", "Phase 1 pICNN", C3, "--")):
        ab = np.array(v8[key]); e.plot(ab[:, 0], t, ls, color=col, lw=1.3, label=lab)
    for key, col in (("svk", C1), ("neohooke", C2)):
        ab = np.array(v8[key]); idx = [8, 16, 24]
        e.plot(ab[idx, 0], t[idx], "o", color=col, mfc="white", ms=4)
    e.set_xlabel(r"stretch $F_{11}$"); e.set_ylabel(r"traction $t=P_{11}$"); e.legend(fontsize=6, loc="upper left")
    panel_label(e, "(d) V8 stress control")
    fig.savefig(OUT / "fig_v_phase1.pdf")
    plt.close(fig)


def fig_v_newton():
    lg = load("verification_logs.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.7), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(wspace=0.32)
    cols = [C1, C2, C3, C4, INK2, C1, C2, C3]; lss = ["-", "-", "-", "-", "-", "--", "--", "--"]
    mks = ["o", "s", "^", "D", "v", "o", "s", "^"]
    for k, (name, r) in enumerate(lg["newton"].items()):
        r = np.array(r); a.semilogy(range(len(r)), r / r[0], mks[k] + lss[k], color=cols[k], ms=3.5, lw=1.1, label=name)
    a.set_xlabel("Newton iteration $k$"); a.set_ylabel(r"$\|\mathbf{R}_k\|/\|\mathbf{R}_0\|$")
    a.set_ylim(1e-17, 3)
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.5, -0.2))
    panel_label(a, "(a) Newton convergence of all solvers")
    t = lg["p2v6_taylor"]
    b.loglog(t["step"], t["remainder"], "o-", color=C1, ms=4, label="P2-V6 remainder")
    s = np.array([1e-3, 2.5e-4]); b.loglog(s, 6.76e-3 * (s / 1e-3) ** 2 * 0.6, ":", color=INK2, lw=0.9, label="slope 2")
    b.set_xlabel(r"perturbation size $\epsilon$"); b.set_ylabel("Taylor remainder"); b.legend(fontsize=6.5)
    b.minorticks_off(); b.set_xticks(t["step"]); b.set_xticklabels([r"$10^{-3}$", r"$5\cdot10^{-4}$", r"$2.5\cdot10^{-4}$"], fontsize=6.5)
    b.set_yticks([1e-4, 1e-3, 1e-2])
    panel_label(b, "(b) condensed RVE tangent")
    fig.savefig(OUT / "fig_v_newton.pdf")
    plt.close(fig)


def fig_v_rve_surrogate():
    lg = load("verification_logs.json"); s = load("s_v1_constitutive_checks.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(wspace=0.75)
    items = list(lg["consistency"].items())
    y = np.arange(len(items))[::-1]
    a.hlines(y, [v for _, v in items], 1e-12, color=GRID, lw=1.1)
    a.plot([v for _, v in items], y, "o", color=C1, ms=4)
    a.axvline(2.22e-16, color=INK2, ls=":", lw=0.9)
    a.set_yticks(y); a.set_yticklabels([k for k, _ in items], fontsize=6.2); a.set_xscale("log")
    a.set_xlim(1e-20, 1e-12); a.set_xlabel("error"); a.grid(axis="y", visible=False)
    panel_label(a, "(a) RVE consistency checks")
    keys = (("energy_gradient_relative_error", "energy gradient", C1, "o"),
            ("tangent_directional_relative_error", "tangent (directional)", C2, "s"),
            ("objectivity_stress_relative_error", "objectivity, stress", C3, "^"),
            ("tangent_symmetry_error", "tangent symmetry", C4, "D"))
    for k, lab, col, mk in keys:
        J = [r["determinant"] for r in s["records"]]; e = [max(r["metrics"][k], 1e-17) for r in s["records"]]
        b.semilogy(J, e, mk, color=col, ms=3.2, label=lab)
        b.axhline(s["thresholds"][k], color=col, lw=0.8, ls="--")
    b.set_xlabel(r"$\overline{J}$ of the test record"); b.set_ylabel("error (dashed: threshold)")
    b.set_ylim(1e-17, 1e-3); b.legend(fontsize=5.8, loc="upper left", ncol=2)
    panel_label(b, "(b) S-V1 surrogate audit")
    fig.savefig(OUT / "fig_v_rve_surrogate.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_training_coverage()
    fig_split_errors()
    fig_moduli()
    fig_shear_holdout()
    fig_macro()
    fig_timing()
    fig_training_curves()
    fig_parity()
    print("figures written to", OUT)
    fig_rve_convergence()
    fig_vf_dependence()
    fig_load_paths()
    fig_bottlenecks()
    fig_rve_fields()
    import verification_figures
    verification_figures.run()
