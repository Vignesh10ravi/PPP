"""V5 -- Multi-element affine patch test."""
import argparse

import numpy as np
import torch
from netgen.occ import Box, OCCGeometry, Pnt
from ngsolve import (
    CF,
    Grad,
    GridFunction,
    Id,
    InnerProduct,
    LinearForm,
    Mesh,
    Norm,
    Projector,
    VectorH1,
    x,
    y,
    z,
    dx,
)

try:
    from ngsolve import IntegrationRuleSpace
except ImportError:
    from ngsolve.comp import IntegrationRuleSpace

from src.constitutive import stress
from src.train import load_model

torch.set_default_dtype(torch.float64)


def sample_deformation_gradients(mesh, gfu, order=1):
    """Evaluate F = I + Grad(u) at the material quadrature points."""
    irs = IntegrationRuleSpace(mesh, order=2 * order)
    gF = GridFunction(irs ** 9)
    gF.Interpolate((Id(3) + Grad(gfu)).Reshape((9,)))
    # Compound IntegrationRuleSpace vectors are component-major:
    # [all component-0 values, all component-1 values, ...].
    raw = gF.vec.FV().NumPy()
    return raw.reshape(9, -1).T.reshape(-1, 3, 3).copy()


def main(ckpt="checkpoints/phase1_validated.pt"):
    model = load_model(ckpt)

    geometry = OCCGeometry(
        Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    )
    mesh = Mesh(geometry.GenerateMesh(maxh=0.3))
    assert mesh.ne > 1, "patch test requires multiple elements"

    # Mixed normal-strain and shear deformation.
    F_target = np.array([
        [1.08, 0.05, 0.00],
        [0.00, 0.97, 0.03],
        [0.00, 0.00, 1.02],
    ])

    # u = (F_target - I) X
    boundary_displacement = CF((
        0.08 * x + 0.05 * y,
        -0.03 * y + 0.03 * z,
        0.02 * z,
    ))

    fes = VectorH1(
        mesh,
        order=1,
        dirichlet=".*",
    )
    gfu = GridFunction(fes)
    gfu.Set(boundary_displacement)

    free_dofs = fes.FreeDofs()
    free_count = sum(bool(value) for value in free_dofs)
    assert free_count > 0, (
        "mesh has no free interior degrees of freedom"
    )

    # Check deformation-gradient uniformity.
    F_ip = sample_deformation_gradients(mesh, gfu)
    F_error = np.max(np.abs(F_ip - F_target))

    # Check constitutive stress uniformity.
    P_ip = stress(
        model,
        torch.from_numpy(F_ip),
    ).detach().numpy()

    stress_scale = max(
        1.0,
        np.max(np.abs(P_ip[0])),
    )
    stress_variation = (
        np.max(np.abs(P_ip - P_ip[0])) / stress_scale
    )

    # Assemble the internal residual through the same
    # IntegrationRuleSpace bridge used by the Newton solver.
    v = fes.TestFunction()
    # Construct the IntegrationRuleSpace stress field directly so this
    # verification does not depend on private ngsolve_umat helper names.
    irs = IntegrationRuleSpace(mesh, order=2)
    irs_dx = dx(intrules=irs.GetIntegrationRules())

    gP = GridFunction(irs ** 9)
    Pmat = CF(
        tuple(gP.components[i] for i in range(9)),
        dims=(3, 3),
    )
    # Convert point-major PyTorch stresses back to component-major
    # IntegrationRuleSpace storage.
    gP.vec.FV().NumPy()[:] = P_ip.reshape(-1, 9).T.ravel()

    residual = LinearForm(fes)
    residual += InnerProduct(Pmat, Grad(v)) * irs_dx
    residual.Assemble()

    # Boundary entries are reactions. A patch test checks equilibrium
    # only on unconstrained interior degrees of freedom.
    free_residual = residual.vec.CreateVector()
    free_residual.data = (
        Projector(free_dofs, True) * residual.vec
    )
    residual_norm = Norm(free_residual)

    print(f"mesh elements:             {mesh.ne}")
    print(f"material points:           {len(F_ip)}")
    print(f"free degrees of freedom:   {free_count}")
    print(f"max |F - F_target|:        {F_error:.3e}")
    print(
        "relative stress variation: "
        f"{stress_variation:.3e}"
    )
    print(f"free residual norm:        {residual_norm:.3e}")

    assert F_error < 1e-10, (
        "deformation gradient is not uniform"
    )
    assert stress_variation < 1e-10, (
        "stress is not uniform"
    )
    assert residual_norm < 1e-10, (
        "assembled free residual is not zero"
    )

    print(
        "V5 PASS: multi-element affine patch is "
        "uniform and in equilibrium."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="checkpoints/phase1_validated.pt",
    )
    main(**vars(parser.parse_args()))
