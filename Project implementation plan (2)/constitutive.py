"""Exact stress and tangent from the NN energy via autograd (no finite differences).

    P = dPsi/dF            (3x3)   1st Piola-Kirchhoff stress
    A = d^2Psi/dF^2        (3x3x3x3)   material tangent  A_ijkl = dP_ij/dF_kl

For total-Lagrange assembly the correct flattening is the *full* 9x9 (F is
unsymmetric), NOT the minor-symmetric 6x6 Voigt used for the 2nd-Piola/
Green-Lagrange tangent.
"""
import torch
from torch.func import jacrev, vmap


def stress(model, F, c=None):
    """Batched first-Piola stress without computing the material tangent."""
    if c is None:
        ref = model.reference(F.dtype, F.device, None)

        def energy_single(f):
            return model.energy(f.unsqueeze(0), ref=ref).squeeze()

        return vmap(jacrev(energy_single))(F)

    def energy_single(f, cc):
        return model.energy(f.unsqueeze(0), cc.unsqueeze(0)).squeeze()

    return vmap(jacrev(energy_single))(F, c)



def stress_and_tangent(model, F, c=None):
    """Batched P and A via autograd.

    F: (n, 3, 3). c: (n, n_ctx) or None. Returns P (n,3,3), A (n,3,3,3,3).
    """
    if c is None:
        # Precompute the reference (psi_I, P_I) once, OUTSIDE the transforms.
        ref = model.reference(F.dtype, F.device, None)

        def energy_single(f):
            return model.energy(f.unsqueeze(0), ref=ref).squeeze()
        P = vmap(jacrev(energy_single))(F)
        A = vmap(jacrev(jacrev(energy_single)))(F)
    else:
        def energy_single(f, cc):
            return model.energy(f.unsqueeze(0), cc.unsqueeze(0)).squeeze()
        P = vmap(jacrev(energy_single))(F, c)
        A = vmap(jacrev(jacrev(energy_single)))(F, c)
    return P, A


def tangent_to_9x9(A):
    """(..., 3,3,3,3) -> (..., 9, 9) full first-Piola tangent for assembly."""
    return A.reshape(*A.shape[:-4], 9, 9)
