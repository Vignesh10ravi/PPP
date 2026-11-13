"""V6 -- Matched analytical Neo-Hookean and neural FE problems."""
import argparse

import numpy as np
import torch
from netgen.occ import Box, OCCGeometry, Pnt, X
from ngsolve import (
    CF,
    Grad,
    GridFunction,
    Id,
    InnerProduct,
    LinearForm,
    Mesh,
    dx,
    x,
)

try:
    from ngsolve import IntegrationRuleSpace
except ImportError:
    from ngsolve.comp import IntegrationRuleSpace

from src.constitutive import stress
from src.neohooke import psi_neohooke
from src.ngsolve_umat import solve
from src.train import load_model

torch.set_default_dtype(torch.float64)


class AnalyticalNeoHooke(torch.nn.Module):
    """Expose the analytical material through the neural material API."""

    def energy(self, F, c=None, ref=None):
        return psi_neohooke(F)

    def reference(self, dtype, device, c=None):
        return (
            torch.zeros((), dtype=dtype, device=device),
            torch.zeros((3, 3), dtype=dtype, device=device),
        )


def relative_l2(actual, expected):
    denominator = max(
        np.linalg.norm(expected),
        np.finfo(float).eps,
    )
    return np.linalg.norm(actual - expected) / denominator


def material_state_and_residual(mesh, gfu, model, order=1):
    """Evaluate F and assemble the internal-force vector."""
    irs = IntegrationRuleSpace(mesh, order=2 * order)
    irs_dx = dx(intrules=irs.GetIntegrationRules())

    # Read F component-by-component, matching ngsolve_umat.py.
    gF = GridFunction(irs ** 9)
    gF.Interpolate((Id(3) + Grad(gfu)).Reshape((9,)))

    F_components = [
        gF.components[i].vec.FV().NumPy().copy()
        for i in range(9)
    ]
    F_ip = np.stack(
        F_components,
        axis=1,
    ).reshape(-1, 3, 3)

    # Evaluate and write P component-by-component.
    P_ip = stress(
        model,
        torch.from_numpy(F_ip),
    ).detach().numpy()

    gP = GridFunction(irs ** 9)
    P_flat = P_ip.reshape(-1, 9)

    for component in range(9):
        gP.components[component].vec.FV().NumPy()[:] = (
            P_flat[:, component]
        )

    Pmat = CF(
        tuple(gP.components[i] for i in range(9)),
        dims=(3, 3),
    )

    fes = gfu.space
    v = fes.TestFunction()

    residual = LinearForm(fes)
    residual += InnerProduct(Pmat, Grad(v)) * irs_dx
    residual.Assemble()

    return F_ip, residual.vec.FV().NumPy().copy()


def main(
    ckpt="checkpoints/phase1_validated.pt",
    displacement_rtol=0.15,
    gradient_rtol=0.15,
    reaction_rtol=0.15,
):
    neural_model = load_model(ckpt)
    analytical_model = AnalyticalNeoHooke().double()

    box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    box.faces.Min(X).name = "left"
    box.faces.Max(X).name = "right"

    mesh = Mesh(
        OCCGeometry(box).GenerateMesh(maxh=0.3)
    )
    boundary_displacement = CF((0.1 * x, 0, 0))

    print("Solving analytical Neo-Hookean problem")
    gfu_exact, history_exact = solve(
        mesh,
        analytical_model,
        bc_cf=boundary_displacement,
        dirichlet="left|right",
        order=1,
    )

    print("\nSolving neural material problem")
    gfu_nn, history_nn = solve(
        mesh,
        neural_model,
        bc_cf=boundary_displacement,
        dirichlet="left|right",
        order=1,
    )

    free = np.array(
        [
            bool(value)
            for value in gfu_exact.space.FreeDofs()
        ],
        dtype=bool,
    )

    u_exact = gfu_exact.vec.FV().NumPy().copy()
    u_nn = gfu_nn.vec.FV().NumPy().copy()

    # The local solver uses a three-component product space. FreeDofs()
    # returns the scalar-space mask, while the GridFunction vector stores
    # one complete block for each displacement component.
    if len(free) != len(u_exact):
        if len(u_exact) % len(free) != 0:
            raise RuntimeError(
                "displacement vector length is incompatible with FreeDofs"
            )

        component_count = len(u_exact) // len(free)
        free = np.tile(free, component_count)
        print(
            f"expanded scalar DOF mask across "
            f"{component_count} displacement components"
        )

    displacement_error = relative_l2(
        u_nn[free],
        u_exact[free],
    )

    F_exact, residual_exact = material_state_and_residual(
        mesh,
        gfu_exact,
        analytical_model,
    )
    F_nn, residual_nn = material_state_and_residual(
        mesh,
        gfu_nn,
        neural_model,
    )

    gradient_error = relative_l2(F_nn, F_exact)

    # Constrained internal-force entries are nodal reactions.
    constrained = ~free
    reaction_error = relative_l2(
        residual_nn[constrained],
        residual_exact[constrained],
    )

    print(f"\nmesh elements:                 {mesh.ne}")
    print(
        "analytical final residual:     "
        f"{history_exact[-1]:.3e}"
    )
    print(
        "neural final residual:         "
        f"{history_nn[-1]:.3e}"
    )
    print(
        "free displacement rel. error: "
        f"{displacement_error:.3e} "
        f"(tol {displacement_rtol:.3e})"
    )
    print(
        "deformation-gradient error:    "
        f"{gradient_error:.3e} "
        f"(tol {gradient_rtol:.3e})"
    )
    print(
        "reaction-vector rel. error:    "
        f"{reaction_error:.3e} "
        f"(tol {reaction_rtol:.3e})"
    )

    assert history_exact[-1] < 1e-8, (
        "analytical FE solve did not converge"
    )
    assert history_nn[-1] < 1e-8, (
        "neural FE solve did not converge"
    )
    assert displacement_error < displacement_rtol, (
        "neural free-displacement error exceeds tolerance"
    )
    assert gradient_error < gradient_rtol, (
        "neural deformation-gradient error exceeds tolerance"
    )
    assert reaction_error < reaction_rtol, (
        "neural reaction-vector error exceeds tolerance"
    )

    print(
        "\nV6 PASS: matched neural and analytical "
        "FE responses agree."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="checkpoints/phase1_validated.pt",
    )
    parser.add_argument(
        "--displacement-rtol",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--gradient-rtol",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--reaction-rtol",
        type=float,
        default=0.15,
    )
    main(**vars(parser.parse_args()))
