"""One figure per verification test (chapter 7).  Run via make_figures.py."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import make_figures as mf
from make_figures import (C1, C2, C3, C4, INK, INK2, GRID, OUT, load, VTESTS, VLABEL)

W1 = 3.1   # single-column figure width (in); two of these sit side by side in the text


def _save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)


# ------------------------------------------------------------------ overview
def fig_v_overview():
    col = {"M": C1, "S": C2, "C": C3}
    fig, ax = plt.subplots(figsize=(6.3, 6.6))
    n = len(VTESTS)
    names = []
    for k, (tid, typ, val, note) in enumerate(VTESTS):
        y = n - 1 - k
        lab = VLABEL.get(tid) or {"V0e": "zero modes", "V1": "interfaces, shapes, dtype",
                                  "V2": r"$\Psi(\mathbf{I})=0,\ \mathbf{P}(\mathbf{I})=\mathbf{0}$",
                                  "P2-V10": "split leaks, regeneration"}.get(tid, "")
        names.append(f"{tid}  {lab}")
        if val is None:
            ax.plot(1e-23, y, "D", color=col[typ], ms=4, zorder=3)
        else:
            ax.hlines(y, val, 1.0, color=GRID, lw=1.1, zorder=1)
            ax.plot(val, y, "o", color=col[typ], ms=4.5, zorder=3)
    ax.axvline(2.22e-16, color=INK2, ls=":", lw=0.9)
    ax.set_yticks(range(n)); ax.set_yticklabels(names[::-1], fontsize=7)
    ax.set_xscale("log"); ax.set_xlim(1e-24, 1.0); ax.set_ylim(-0.7, n - 0.3)
    ax.set_xticks([1e-20, 1e-16, 1e-12, 1e-8, 1e-4, 1e0])
    ax.set_xlabel("achieved error measure of the test (relative or absolute, Table 7.1)")
    ax.grid(axis="y", visible=False)
    h = [Line2D([], [], marker="o", ls="none", color=col[t], label=l) for t, l in
         (("M", "mandatory (guideline)"), ("S", "standard benchmark"), ("C", "custom"))]
    h.append(Line2D([], [], marker="D", ls="none", color=INK2, label="count / pass criterion"))
    h.append(Line2D([], [], color=INK2, ls=":", label=r"machine precision $2.2\times10^{-16}$"))
    ax.legend(handles=h, loc="upper center", fontsize=7, ncol=3, bbox_to_anchor=(0.35, -0.07))
    _save(fig, "fig_v_overview.pdf")


# ------------------------------------------------------------------ setup figures
def _cube_axes(ax):
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=22, azim=-58)
    ax.set_xlim(-0.05, 1.3); ax.set_ylim(-0.05, 1.1); ax.set_zlim(-0.05, 1.1)


def _setup_fig(w=2.6, h=2.6):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111, projection="3d"); _cube_axes(ax)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.12)
    return fig, ax


def _cap(ax, txt):
    ax.text2D(0.5, -0.02, txt, transform=ax.transAxes, ha="center", va="top", fontsize=7, color=INK2)


def fig_v_setup_all():
    tm = load("test_meshes.json")
    # V0d / V0e: single tetrahedron and hexahedron
    fig, ax = _setup_fig(3.0, 2.7)
    V = np.array(tm["tet"]["V"])
    faces = [[V[i] for i in f] for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))]
    ax.add_collection3d(Poly3DCollection(faces, facecolor=C1, alpha=0.25, edgecolor=INK2, lw=0.6))
    H = np.array(tm["hex"]["V"]) * 0.45 + np.array([0.8, 0.55, 0.0])
    ax.scatter(*V.T, s=8, color=C1); ax.scatter(*H.T, s=8, color=C2)
    hexf = [[H[i] for i in f] for f in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4))]
    ax.add_collection3d(Poly3DCollection(hexf, facecolor=C2, alpha=0.18, edgecolor=INK2, lw=0.6))
    _cap(ax, "irregular tetrahedron (blue) and distorted hexahedron (orange), no supports")
    _save(fig, "fig_v_setup_v0.pdf")
    # V4 / V6: stretched block
    for key, name, txt in (("v4", "fig_v_setup_v4.pdf", None), ("v56", "fig_v_setup_v6.pdf", None)):
        fig, ax = _setup_fig()
        m = tm[key]; Vv = np.array(m["V"])
        tri = [[Vv[i] for i in t] for t in m["T"]]
        cols = []
        for t in m["T"]:
            xs = Vv[t, 0]
            cols.append(C1 if np.all(xs < 1e-9) else C2 if np.all(xs > 1 - 1e-9) else "#f2f1ee")
        ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolor=INK2, lw=0.25, alpha=0.9))
        ax.quiver(1.0, 0.5, 0.5, 0.25, 0, 0, color=C2, lw=1.2, arrow_length_ratio=0.35)
        _cap(ax, f"{m['ne']} tetrahedra; $u_x=0.1x$ on $x=0$ (blue), $x=1$ (orange)")
        _save(fig, name)
    # V5: affine patch
    fig, ax = _setup_fig()
    m = tm["v56"]; Vv = np.array(m["V"])
    tri = [[Vv[i] for i in t] for t in m["T"]]
    ax.add_collection3d(Poly3DCollection(tri, facecolors=C3, edgecolor=INK2, lw=0.25, alpha=0.9))
    _cap(ax, rf"{m['ne']} tetrahedra; boundary nodes follow $\mathbf{{F}}_\mathrm{{target}}$")
    _save(fig, "fig_v_setup_v5.pdf")
    # V7 / V8: one hexahedron
    fig, ax = _setup_fig()
    c = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    f = [[c[i] for i in q] for q in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4))]
    fc = ["#f2f1ee", "#f2f1ee", "#f2f1ee", "#f2f1ee", C2, C1]
    ax.add_collection3d(Poly3DCollection(f, facecolors=fc, edgecolor=INK2, lw=0.6, alpha=0.6))
    for yy in (0.25, 0.75):
        for zz in (0.25, 0.75):
            ax.quiver(1.0, yy, zz, 0.25, 0, 0, color=C2, lw=1.0, arrow_length_ratio=0.35)
    _cap(ax, "V8: symmetry supports (blue), traction $t$ (orange);\nV7: all nodes prescribed")
    _save(fig, "fig_v_setup_v78.pdf")


# ------------------------------------------------------------------ V0d / V0e spectra
def fig_v0_spectra():
    d = load("verification_data.json")["v0"]
    cases = (("tet1", r"(a) tetrahedron, $P_1$"), ("hex1", r"(b) hexahedron, $P_1$"), ("hex2", r"(c) hexahedron, $P_2$"))
    mods = (("neo-Hookean (analytical)", C1, "o"), ("RVE pICNN", C2, "s"), ("Phase 1 pICNN", C3, "^"))
    fig, axs = plt.subplots(2, 2, figsize=(6.2, 4.6))
    fig.subplots_adjust(wspace=0.32, hspace=0.45)
    axs = axs.ravel()
    for ax, (key, title) in zip(axs, cases):
        for name, col, mk in mods:
            lam = np.array(d[f"{key}|{name}"])
            ax.plot(np.arange(1, len(lam) + 1), lam, mk, color=col, ms=3, mfc="white" if name.startswith("Phase") else col, label=name)
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold", color=INK)
    lam = np.array(d["hex1_onepoint|neo-Hookean (analytical)"])
    axs[3].plot(np.arange(1, len(lam) + 1), lam, "o", color=C1, ms=3)
    axs[3].axvspan(0.5, 18.5, color=C4, alpha=0.15, lw=0)
    axs[3].annotate("18 zero modes", xy=(9.5, 0), xytext=(9.5, 0.35), ha="center", fontsize=7, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    axs[3].set_title(r"(d) hexahedron, $P_1$, one-point rule", loc="left", fontsize=8, fontweight="bold", color=INK)
    for ax in axs:
        ax.set_yscale("symlog", linthresh=1e-6); ax.axhline(0, color=INK2, lw=0.6)
        ax.set_yticks([-1e-2, 0, 1e-4, 1]); ax.set_ylim(-0.1, 1.5)
        ax.set_xlabel("eigenvalue index"); ax.set_ylabel(r"$\lambda_i/\max|\lambda|$"); ax.xaxis.set_major_locator(__import__("matplotlib").ticker.MaxNLocator(integer=True))
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    _save(fig, "fig_v0_spectra.pdf")


# ------------------------------------------------------------------ V2b
def fig_v2b():
    d = load("verification_data.json")["v2b"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.2, 2.9))
    fig.subplots_adjust(wspace=0.35, bottom=0.25)
    pe, pn = np.array(d["psi_exact"]), np.array(d["psi_nn"])
    a.plot(pe, pn, "o", color=C1, ms=2.5, alpha=0.7)
    lim = [0, pe.max() * 1.05]; a.plot(lim, lim, "-", color=INK2, lw=0.8)
    a.set_xlabel(r"$\Psi$ neo-Hooke"); a.set_ylabel(r"$\Psi$ pICNN"); a.set_title("(a) energy", loc="left", fontweight="bold")
    Pe, Pn = np.array(d["P_exact"]), np.array(d["P_nn"])
    b.plot(Pe[:, [0, 4, 8]].ravel(), Pn[:, [0, 4, 8]].ravel(), "o", color=C1, ms=2.2, alpha=0.6, label="normal components")
    b.plot(Pe[:, [1, 2, 3, 5, 6, 7]].ravel(), Pn[:, [1, 2, 3, 5, 6, 7]].ravel(), "o", color=C2, ms=2.2, alpha=0.6, label="shear components")
    lim = [Pe.min(), Pe.max()]; b.plot(lim, lim, "-", color=INK2, lw=0.8)
    b.set_xlabel(r"$P_{ij}$ neo-Hooke"); b.set_ylabel(r"$P_{ij}$ pICNN"); b.set_title("(b) first Piola stress", loc="left", fontweight="bold")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7, markerscale=2)
    _save(fig, "fig_v2b.pdf")


# ------------------------------------------------------------------ V3, V4
def fig_v3_v4():
    d = load("phase1_logs.json")
    v3 = d["v3_tangent_fd"]
    eps_v = np.array(v3["eps"], float); err_v = np.array(v3["rel_err_rerun"], float)
    fig, ax = plt.subplots(figsize=(W1 + 0.3, 2.7))
    ax.loglog(eps_v, err_v, "o-", color=C1, label="central differences vs. AD")
    eg = np.array([1e-2, 1e-5]); ax.loglog(eg, 0.15 * err_v[0] * (eg / eg[0]) ** 2, ":", color=INK2, lw=1.0, label=r"slope $\epsilon^2$")
    ax.set_xlabel(r"central-difference step $\epsilon$"); ax.set_ylabel("relative tangent error"); ax.invert_xaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=7.5)
    _save(fig, "fig_v3.pdf")
    r = d["v4_newton_checkpoint"]; r2 = d["v4_newton_seeded"]
    fig, ax = plt.subplots(figsize=(W1 + 0.3, 2.7))
    ax.semilogy(range(len(r)), r, "o-", color=C1, label="trained Phase 1 network")
    ax.semilogy(range(len(r2)), r2, "s--", color=C2, label="seeded (untrained) network")
    ax.axhline(1e-8, color=INK2, lw=0.8, ls=":", label=r"tolerance $10^{-8}$")
    ax.set_xlabel("Newton iteration $k$"); ax.set_ylabel(r"$\|\mathbf{R}_\mathrm{free}\|_2$"); ax.set_xticks(range(5))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=7.5)
    _save(fig, "fig_v4.pdf")


# ------------------------------------------------------------------ V7, V8
def fig_v7_v8():
    lg = load("verification_logs.json"); d = load("verification_data.json")
    fd = lg["v7_fd"]
    fig, ax = plt.subplots(figsize=(W1 + 0.3, 2.7))
    for name, col, mk in (("St. Venant-Kirchhoff", C1, "o"), ("neo-Hookean", C2, "s")):
        ax.loglog(fd["eps"], fd[name], mk + "-", color=col, ms=3.5, label=name.replace("St. Venant-Kirchhoff", "SVK"))
    eg = np.array([1e-3, 1e-5]); ax.loglog(eg, 8.49e-6 * (eg / 1e-3) ** 2 * 0.25, ":", color=INK2, lw=0.9, label=r"$\epsilon^2$")
    ax.invert_xaxis(); ax.set_xlabel(r"step $\epsilon$"); ax.set_ylabel("relative tangent error")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=7.5)
    _save(fig, "fig_v7.pdf")
    v8 = d["v8"]; t = np.array(v8["t"])
    fig, ax = plt.subplots(figsize=(W1 + 0.3, 2.7))
    for key, lab, col, ls in (("svk", "SVK", C1, "-"), ("neohooke", "neo-Hooke", C2, "-"), ("phase1", "Phase 1 pICNN", C3, "--")):
        ab = np.array(v8[key]); ax.plot(ab[:, 0], t, ls, color=col, lw=1.3, label=lab)
    for key, col in (("svk", C1), ("neohooke", C2)):
        ab = np.array(v8[key]); idx = [8, 16, 24]
        ax.plot(ab[idx, 0], t[idx], "o", color=col, mfc="white", ms=4)
    ax.set_xlabel(r"stretch $F_{11}$"); ax.set_ylabel(r"traction $t=P_{11}$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=7.5)
    _save(fig, "fig_v8.pdf")


# ------------------------------------------------------------------ Newton (all solvers), P2-V6
def fig_v_newton_all():
    lg = load("verification_logs.json")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    cols = [C1, C2, C3, C4, INK2, C1, C2, C3]; lss = ["-", "-", "-", "-", "-", "--", "--", "--"]
    mks = ["o", "s", "^", "D", "v", "o", "s", "^"]
    for k, (name, r) in enumerate(lg["newton"].items()):
        r = np.array(r); ax.semilogy(range(len(r)), r / r[0], mks[k] + lss[k], color=cols[k], ms=3.5, lw=1.1, label=name)
    ax.set_xlabel("Newton iteration $k$"); ax.set_ylabel(r"$\|\mathbf{R}_k\|/\|\mathbf{R}_0\|$"); ax.set_ylim(1e-17, 3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=7)
    _save(fig, "fig_v_newton_all.pdf")


def fig_v_p2v6():
    lg = load("verification_logs.json"); t = lg["p2v6_taylor"]
    fig, ax = plt.subplots(figsize=(W1 + 0.3, 2.7))
    ax.loglog(t["step"], t["remainder"], "o-", color=C1, ms=4, label="Taylor remainder")
    s = np.array([1e-3, 2.5e-4]); ax.loglog(s, 6.76e-3 * (s / 1e-3) ** 2 * 0.6, ":", color=INK2, lw=0.9, label="slope 2")
    ax.set_xlabel(r"perturbation size $\epsilon$"); ax.set_ylabel("Taylor remainder")
    ax.minorticks_off(); ax.set_xticks(t["step"]); ax.set_xticklabels([r"$10^{-3}$", r"$5\cdot10^{-4}$", r"$2.5\cdot10^{-4}$"], fontsize=7)
    ax.set_yticks([1e-4, 1e-3, 1e-2])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.5)
    _save(fig, "fig_v_p2v6.pdf")


# ------------------------------------------------------------------ RVE checks, S-V1
def fig_v_rve_checks():
    lg = load("verification_logs.json")
    items = list(lg["consistency"].items()); y = np.arange(len(items))[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hlines(y, [v for _, v in items], 1e-12, color=GRID, lw=1.1)
    ax.plot([v for _, v in items], y, "o", color=C1, ms=4)
    ax.axvline(2.22e-16, color=INK2, ls=":", lw=0.9, label=r"machine precision")
    ax.set_yticks(y); ax.set_yticklabels([k for k, _ in items], fontsize=7); ax.set_xscale("log")
    ax.set_xlim(1e-20, 1e-12); ax.set_xlabel("error (absolute or relative)"); ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7.5)
    _save(fig, "fig_v_rve_checks.pdf")


def fig_v_s1():
    s = load("s_v1_constitutive_checks.json")
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    keys = (("energy_gradient_relative_error", "energy gradient", C1, "o"),
            ("tangent_directional_relative_error", "tangent (directional)", C2, "s"),
            ("objectivity_stress_relative_error", "objectivity, stress", C3, "^"),
            ("tangent_symmetry_error", "tangent symmetry", C4, "D"))
    for k, lab, col, mk in keys:
        J = [r["determinant"] for r in s["records"]]; e = [max(r["metrics"][k], 1e-17) for r in s["records"]]
        ax.semilogy(J, e, mk, color=col, ms=3.5, label=lab)
        ax.axhline(s["thresholds"][k], color=col, lw=0.8, ls="--")
    ax.set_xlabel(r"$\overline{J}$ of the test record"); ax.set_ylabel("error (dashed: threshold of the check)")
    ax.set_ylim(1e-17, 1e-3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, fontsize=7, columnspacing=1.0, handletextpad=0.3)
    _save(fig, "fig_v_s1.pdf")


def run():
    fig_v_overview(); fig_v_setup_all(); fig_v0_spectra(); fig_v2b(); fig_v3_v4(); fig_v7_v8()
    fig_v_newton_all(); fig_v_p2v6(); fig_v_rve_checks(); fig_v_s1()


if __name__ == "__main__":
    run()
