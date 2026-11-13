"""V2b -- Compare the trained pICNN with analytical Neo-Hookean response.

Energy, first-Piola stress, and the full total-Lagrange tangent are evaluated
on deterministic validation deformations that are not used by training.

Run:
    python -m tests.verify_neohooke_response
    python -m tests.verify_neohooke_response --ckpt checkpoints/phase1.pt
"""
import argparse
from pathlib import Path

import torch

from src.constitutive import stress_and_tangent
from src.neohooke import LAM, MU, P_neohooke, psi_neohooke, sample_F
from src.train import load_model

torch.set_default_dtype(torch.float64)


def relative_l2(actual, expected):
    """Global relative L2 error, stable for entries whose exact value is zero."""
    return ((actual - expected).norm() / expected.norm()).item()


def A_neohooke(F, mu=MU, lam=LAM):
    """Analytical tangent A_ijkl = dP_ij/dF_kl."""
    FinvT = torch.linalg.inv(F).transpose(-1, -2)
    alpha = lam * torch.log(torch.linalg.det(F)) - mu
    eye = torch.eye(3, dtype=F.dtype, device=F.device)

    identity_term = mu * torch.einsum("ik,jl->ijkl", eye, eye)
    volumetric_term = lam * torch.einsum(
        "...ij,...kl->...ijkl", FinvT, FinvT
    )
    inverse_term = alpha[..., None, None, None, None] * torch.einsum(
        "...kj,...il->...ijkl", FinvT, FinvT
    )
    return identity_term + volumetric_term - inverse_term


def main(
    ckpt="checkpoints/phase1.pt",
    n=256,
    max_strain=0.2,
    energy_rtol=0.10,
    stress_rtol=0.10,
    tangent_rtol=0.25,
):
    checkpoint = Path(ckpt)
    assert checkpoint.is_file(), (
        f"checkpoint not found: {checkpoint}; train it with "
        "`python -m src.train --epochs 1500 --ckpt checkpoints/phase1.pt`"
    )

    model = load_model(checkpoint)
    generator = torch.Generator().manual_seed(2026)
    F = sample_F(
        n, max_strain=max_strain, generator=generator
    ).double()
    assert len(F) > 0, "no admissible validation deformations were generated"

    ref = model.reference(F.dtype, F.device)
    with torch.no_grad():
        psi_exact = psi_neohooke(F)
        P_exact = P_neohooke(F)
        A_exact = A_neohooke(F)
        psi_nn = model.energy(F, ref=ref)
    P_nn, A_nn = stress_and_tangent(model, F)

    errors = {
        "energy": relative_l2(psi_nn, psi_exact),
        "stress": relative_l2(P_nn, P_exact),
        "tangent": relative_l2(A_nn, A_exact),
    }

    print(f"checkpoint: {checkpoint}")
    print(f"validation samples: {len(F)} (seed=2026, max_strain={max_strain})")
    print("\nquantity   relative L2 error   tolerance")
    print(f"energy     {errors['energy']:17.3e}   {energy_rtol:.3e}")
    print(f"stress     {errors['stress']:17.3e}   {stress_rtol:.3e}")
    print(f"tangent    {errors['tangent']:17.3e}   {tangent_rtol:.3e}")

    assert errors["energy"] < energy_rtol, "NN energy error exceeds tolerance"
    assert errors["stress"] < stress_rtol, "NN stress error exceeds tolerance"
    assert errors["tangent"] < tangent_rtol, "NN tangent error exceeds tolerance"
    print("\nV2b PASS: trained NN matches the analytical Neo-Hookean response.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/phase1.pt")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--max-strain", type=float, default=0.2)
    parser.add_argument("--energy-rtol", type=float, default=0.10)
    parser.add_argument("--stress-rtol", type=float, default=0.10)
    parser.add_argument("--tangent-rtol", type=float, default=0.25)
    args = parser.parse_args()
    main(**vars(args))
