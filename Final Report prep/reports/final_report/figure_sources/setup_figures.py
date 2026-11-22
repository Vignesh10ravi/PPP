"""Setup figures of the RVE and macroscopic FE models (mesh views and boundary-condition sketches).

Run from the report root:  python figure_sources/setup_figures.py
Mesh data: figure_sources/data/meshes.json, written by export_meshes.py.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Arc  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import make_figures as mf  # noqa: E402  (shared style and colours)

C1, C2, C3, INK, INK2 = mf.C1, mf.C2, mf.C3, mf.INK, mf.INK2
D = json.load(open(mf.DATA / "meshes.json"))
CUBE_EDGES = [((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)), ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
              ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
              ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1))]


def setup3d(ax, elev=22, azim=-58):
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_axis_off()


def tris(points, faces):
    P = np.asarray(points)
    return [P[f] for f in faces]


def axes_triad(ax, o=(1.12, -0.12, 0.0), L=0.2):
    for d, lab in zip(np.eye(3), ("$X_1$", "$X_2$", "$X_3$")):
        e = np.asarray(o) + L * d
        ax.plot(*zip(o, e), color=INK2, lw=0.8)
        ax.text(*(np.asarray(o) + 1.25 * L * d), lab, color=INK2, fontsize=7, ha="center", va="center")


def sketch_axes(ax):
    ax.set_aspect("equal"); ax.set_axis_off(); ax.grid(False)


def arrow(ax, p, q, color=INK, lw=1.0, style="-|>", ms=8, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, **kw))


# ------------------------------------------------------------------ RVE
def fig_rve_setup():
    r = D["rve"]
    P, F = r["points"], r["faces"]
    fig = plt.figure(figsize=(6.6, 3.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    setup3d(ax)
    back = [t for k in ("xmin", "ymax", "zmin") for t in tris(P, F[k])]
    part = tris(P, F["particle"])
    fc = ["#ecebe7"] * len(back) + [C1] * len(part)
    ec = ["#9c9a94"] * len(back) + ["#174f92"] * len(part)
    ax.add_collection3d(Poly3DCollection(back + part, facecolors=fc, edgecolors=ec, lw=0.25))
    for a, b in CUBE_EDGES:
        ax.plot(*zip(a, b), color=INK2, lw=0.7)
    ax.set_title("(a) RVE mesh, $v_f=0.05$", loc="left", fontweight="bold", color=INK, y=0.98)
    ax.text2D(0.02, 0.0, "862 tetrahedra, $h=0.22$; back faces\nand cavity surface shown",
              transform=ax.transAxes, fontsize=7, color=INK2)

    b = fig.add_subplot(1, 2, 2)
    sketch_axes(b)
    b.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f4f3ef", edgecolor=INK, lw=1.0))
    R = r["radius"] * 1.25
    b.add_patch(Circle((0.5, 0.5), R, facecolor=C1, edgecolor="#174f92", lw=1.0))
    b.text(0.5, 0.6, "rigid\nparticle", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    b.plot(0.5, 0.5, "o", color="white", ms=2.5)
    b.text(0.5, 0.47, r"$\mathbf{X}_c$", ha="center", va="top", color="white", fontsize=7)
    # rotation arc
    b.add_patch(Arc((0.5, 0.5), 2 * R + 0.07, 2 * R + 0.07, theta1=15, theta2=75, color=C2, lw=1.2))
    arrow(b, (0.5 + (R + 0.035) * np.cos(np.radians(70)), 0.5 + (R + 0.035) * np.sin(np.radians(70))),
          (0.5 + (R + 0.035) * np.cos(np.radians(78)), 0.5 + (R + 0.035) * np.sin(np.radians(78))), color=C2, ms=7)
    b.text(0.60, 0.85, r"$\mathbf{R}\in SO(3)$", color=C2, fontsize=7.5)
    # periodic pairs
    for y in (0.08, 0.92):
        arrow(b, (0.04, y), (0.96, y), color=C3, lw=0.8, style="<|-|>", ms=6, ls=(0, (3, 2)))
    for x in (0.08, 0.92):
        arrow(b, (x, 0.04), (x, 0.96), color=C3, lw=0.8, style="<|-|>", ms=6, ls=(0, (3, 2)))
    b.text(0.5, 0.11, r"$\mathbf{w}^{+}=\mathbf{w}^{-}$", ha="center", color=C3, fontsize=7.5)
    b.text(0.5, 1.03, "matrix (PDMS): $P_2$ fluctuation $\\mathbf{w}$, $P_1$ pressure $p$\n"
           r"$\mu=1,\ \kappa=10^4$", ha="center", va="bottom", fontsize=7, color=INK)
    # macro deformation
    b.text(1.05, 0.5, r"$\overline{\mathbf{F}}$ prescribed:" "\n" r"$\mathbf{u}=(\overline{\mathbf{F}}-\mathbf{I})\mathbf{X}+\mathbf{w}$",
           fontsize=7.5, va="center", color=INK)
    b.text(1.05, 0.17, "periodic faces:\n$x$, $y$, $z$ pairs\n(identified mesh)", fontsize=7, va="center", color=C3)
    b.text(1.05, 0.83, "cavity surface:\n" r"$\mathbf{u}_p=(\overline{\mathbf{F}}-\mathbf{I})\mathbf{X}_c$" "\n"
           r"$\ +(\mathbf{R}-\mathbf{I})(\mathbf{X}-\mathbf{X}_c)$", fontsize=7, va="center", color=C1)
    b.text(0.5, -0.07, r"output: $\overline{\Psi},\ \overline{\mathbf{P}},\ \overline{\mathbb{A}}$ (Schur complement)",
           ha="center", va="top", fontsize=7.5, color=INK)
    b.set_xlim(-0.05, 1.75); b.set_ylim(-0.17, 1.17)
    b.set_title("(b) Boundary conditions (section)", loc="left", fontweight="bold", color=INK)
    fig.subplots_adjust(wspace=0.02, left=0.0, right=1.0)
    fig.savefig(mf.OUT / "fig_rve_setup.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ macro
def fig_macro_setup():
    m = D["macro"]
    P = np.asarray(m["points"]); T = np.asarray(m["faces"]["default"])
    tri = P[T]; c = tri.mean(axis=1)
    fig = plt.figure(figsize=(6.6, 3.0))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    setup3d(ax)
    groups = {"x0": [], "x1": [], "lat": []}
    for t, cc in zip(tri, c):
        if cc[0] < 1e-9:
            groups["x0"].append(t)
        elif cc[0] > 1 - 1e-9:
            groups["x1"].append(t)
        elif cc[1] < 1e-9 or cc[2] > 1 - 1e-9:      # visible lateral faces only
            groups["lat"].append(t)
    ax.add_collection3d(Poly3DCollection(groups["lat"], facecolor="#f4f3ef", edgecolor="#8a8882", lw=0.35))
    ax.add_collection3d(Poly3DCollection(groups["x1"], facecolor="#f6c9b3", edgecolor="#a5472a", lw=0.35))
    for a, b_ in CUBE_EDGES:
        ax.plot(*zip(a, b_), color=INK2, lw=0.6)
    ax.set_title("(a) Macroscopic mesh, $h=0.25$", loc="left", fontweight="bold", color=INK, y=0.98)
    ax.text2D(0.02, 0.0, "399 tetrahedra; $P_1$: 405 / $P_2$: 2334 free DOFs\n"
              "frozen pICNN at each quadrature point (order 4)", transform=ax.transAxes, fontsize=7, color=INK2)

    b = fig.add_subplot(1, 2, 2)
    sketch_axes(b)
    b.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f4f3ef", edgecolor=INK, lw=1.0))
    # clamp hatch
    b.add_patch(Rectangle((-0.08, -0.05), 0.08, 1.1, facecolor="none", edgecolor=INK2, hatch="////", lw=0))
    b.plot([0, 0], [-0.05, 1.05], color=INK, lw=2)
    # deformed outline (schematic, exaggerated)
    yy = np.linspace(0, 1, 30)
    s = 0.25
    xs = np.linspace(0, 1 + s, 30)
    neck = 0.06 * np.sin(np.pi * xs / (1 + s))
    b.plot(xs, 1 - neck, color=C2, lw=1.0, ls="--"); b.plot(xs, neck, color=C2, lw=1.0, ls="--")
    b.plot([1 + s, 1 + s], [0, 1], color=C2, lw=2)
    for y in (0.2, 0.5, 0.8):
        arrow(b, (1.0, y), (1 + s, y), color=C2, lw=1.0, ms=7)
    b.text(1 + s + 0.03, 0.5, r"$\mathbf{u}=(\varepsilon,0,0)$" "\n" r"on $X_1=1$" "\n" r"$\varepsilon\leq5\,\%$",
           fontsize=7.5, va="center", color=C2)
    b.text(-0.12, 0.5, r"$\mathbf{u}=\mathbf{0}$" "\non $X_1=0$", fontsize=7.5, va="center", ha="right", color=INK)
    b.text(0.5, 1.07, "lateral faces traction-free", ha="center", fontsize=7, color=INK2)
    b.text(0.5, 0.5, r"$v_f=0.05$" "\n" r"$\mathbf{P}=\partial_{\mathbf{F}}\widehat\Psi$," "\n"
           r"$\mathbb{A}=\partial^2_{\mathbf{F}}\widehat\Psi$", ha="center", va="center", fontsize=7.5, color=INK)
    b.text(0.62, -0.1, "dashed: deformed shape (schematic, exaggerated)", ha="center", va="top", fontsize=6.5, color=C2)
    arrow(b, (-0.05, -0.2), (0.2, -0.2), color=INK2, lw=0.8, ms=6); b.text(0.22, -0.2, "$X_1$", va="center", fontsize=7, color=INK2)
    b.set_xlim(-0.45, 1.75); b.set_ylim(-0.25, 1.15)
    b.set_title("(b) Boundary conditions (side view)", loc="left", fontweight="bold", color=INK)
    fig.subplots_adjust(wspace=0.02, left=0.0, right=1.0)
    fig.savefig(mf.OUT / "fig_macro_setup.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_rve_setup()
    fig_macro_setup()
    print("setup figures written")
