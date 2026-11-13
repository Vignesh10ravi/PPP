"""Compressible Neo-Hookean reference model and Phase-1 data generation.

Phase-1 rationale: a single element under an affine (linear) displacement field
has a *homogeneous* deformation gradient F, so its homogenized response equals
the pointwise material response. The training data is therefore the analytic
Neo-Hookean (Psi, P) sampled over admissible F -- this is exactly the
"single-element" data the committee referred to, without any volume averaging.

Energy (Holzapfel form):
    Psi(F) = mu/2 (I1 - 3) - mu ln J + lam/2 (ln J)^2
with I1 = tr(F^T F), J = det F.
"""
import torch

# Nondimensional moduli. The stiff-inclusion contrast is a Phase-2 concern;
# Phase 1 only needs a well-conditioned material to prove the bridge.
MU, LAM = 1.0, 10.0


def psi_neohooke(F, mu=MU, lam=LAM):
    """Strain energy density. F: (..., 3, 3) -> (...,)."""
    C = F.transpose(-1, -2) @ F
    I1 = C.diagonal(dim1=-2, dim2=-1).sum(-1)
    J = torch.linalg.det(F)
    lnJ = torch.log(J)
    return 0.5 * mu * (I1 - 3.0) - mu * lnJ + 0.5 * lam * lnJ ** 2


def P_neohooke(F, mu=MU, lam=LAM):
    """Analytic 1st Piola-Kirchhoff stress P = dPsi/dF, for cross-checking."""
    FinvT = torch.linalg.inv(F).transpose(-1, -2)
    lnJ = torch.log(torch.linalg.det(F))[..., None, None]
    return mu * (F - FinvT) + lam * lnJ * FinvT


def sample_F(n, max_strain=0.3, det_floor=0.2, device="cpu", generator=None):
    """Random admissible deformation gradients in a neighbourhood of I.

    Rejects samples with det F <= det_floor to keep J > 0 (physical) and the
    log terms finite.
    """
    H = max_strain * (torch.rand(n, 3, 3, device=device, generator=generator) - 0.5) * 2.0
    F = torch.eye(3, device=device) + H
    return F[torch.linalg.det(F) > det_floor]


def make_dataset(n=20000, device="cpu", seed=0):
    """Return (F, psi, P) tensors for Sobolev training."""
    g = torch.Generator(device=device).manual_seed(seed)
    F = sample_F(n, device=device, generator=g)
    return F, psi_neohooke(F), P_neohooke(F)
