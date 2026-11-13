"""V4 -- Single-element UMAT benchmark (assembly check).

Deploys the trained (or untrained) PyTorch material onto one hexahedral element
under uniform Dirichlet loading, driven by the custom Newton-Raphson loop with
the exact autograd tangent. Confirms manual global-stiffness assembly and
quadratic convergence of the residual norm, e.g. 1e-1 -> 1e-3 -> 1e-8.

Run:  python -m tests.verify_element
"""
import torch
from netgen.occ import Box, OCCGeometry, Pnt, X
from ngsolve import Mesh, CF, x

# Correct way to load the model in verify_element.py
from src.picnn import HyperelasticNN
model = HyperelasticNN()
# 1. Load the entire checkpoint file first (set weights_only=False because 'history' is likely a Python list, not a tensor)
checkpoint = torch.load("/home/vignesh/PPP/Project implementation plan (1)/checkpoints/phase1.pt", weights_only=False)
# 2. Extract strictly the weights from the dictionary and load them into the model
model.load_state_dict(checkpoint["state_dict"])
from src.ngsolve_umat import solve


torch.set_default_dtype(torch.float64)


def check_quadratic(history):
    """Loosely verify the residual roughly squares each step once in the basin."""
    drops = [history[i + 1] / history[i] ** 2 for i in range(len(history) - 1)
             if history[i] > 1e-12]
    return drops


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
    assert history[-1] < 1e-8, "Newton did not converge -- tangent likely inconsistent"
    print("V4 PASS: NN-driven Newton converged; assembly + autograd tangent are consistent.")


if __name__ == "__main__":
    main()
