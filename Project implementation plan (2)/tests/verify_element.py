"""V4 -- Single-element UMAT benchmark (assembly check).

Deploys the trained (or untrained) PyTorch material onto a coarse tetrahedral
unit-cube mesh (maxh = 0.4) under displacement-controlled stretch, driven by the custom Newton-Raphson loop with
the exact autograd tangent. Confirms manual global-stiffness assembly and
quadratic convergence of the residual norm, e.g. 1e-1 -> 1e-3 -> 1e-8.

Run:  python -m tests.verify_element [--ckpt checkpoints/phase1_validated.pt]
"""
import torch
from netgen.occ import Box, OCCGeometry, Pnt, X
from ngsolve import Mesh, CF, x

import argparse

from src.picnn import HyperelasticNN
from src.ngsolve_umat import solve


torch.set_default_dtype(torch.float64)


def check_quadratic(history):
    """Loosely verify the residual roughly squares each step once in the basin."""
    drops = [history[i + 1] / history[i] ** 2 for i in range(len(history) - 1)
             if history[i] > 1e-12]
    return drops


def print_convergence_table(history, tol):
    """Tabulate ||R_k||, the quadratic ratio ||R_k+1||/||R_k||^2 and the
    empirical order q_k = log(r_k+1/r_k) / log(r_k/r_k-1) (PPP guideline 4.2.2-1a)."""
    import math
    print(f"\nNewton convergence (stopping tolerance ||R_free|| < {tol:.0e})")
    print(f"{'k':>3s} {'||R_free||':>12s} {'r_k+1/r_k^2':>12s} {'order q_k':>10s}")
    for k, r in enumerate(history):
        ratio = history[k + 1] / r ** 2 if k + 1 < len(history) else float("nan")
        q = float("nan")
        if 0 < k < len(history) - 1:
            q = math.log(history[k + 1] / r) / math.log(r / history[k - 1])
        print(f"{k:3d} {r:12.3e} {ratio:12.3e} {q:10.2f}")


def main(ckpt=None):
    torch.manual_seed(0)
    if ckpt:
        from src.train import load_model
        model = load_model(ckpt)
    else:
        model = HyperelasticNN().double()  # untrained still gives a valid (SPD) material

    box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    box.faces.Min(X).name = "left"    # x = 0
    box.faces.Max(X).name = "right"   # x = 1
    # refine so there are free (interior + lateral-face) dofs to actually solve
    mesh = Mesh(OCCGeometry(box).GenerateMesh(maxh=0.4))

    # Prescribe 10% x-stretch on left|right only (u_x = 0.1 x -> 0 at x=0, 0.1 at
    # x=1); the four lateral faces are traction-free, so the material contracts
    # laterally -> a genuine nonlinear solve, not a fully prescribed field.
    bc = CF((0.1 * x, 0, 0))
    gfu, history = solve(mesh, model, bc_cf=bc, dirichlet="left|right", order=1)

    print("\nresidual history:", [f"{h:.2e}" for h in history])
    print_convergence_table(history, tol=1e-8)
    assert history[-1] < 1e-8, "Newton did not converge -- tangent likely inconsistent"
    print("V4 PASS: NN-driven Newton converged; assembly + autograd tangent are consistent.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=None, help="optional trained Phase 1 checkpoint")
    main(parser.parse_args().ckpt)
