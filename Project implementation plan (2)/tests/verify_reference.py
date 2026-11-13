"""V2 -- Verify zero energy and zero stress at the reference state.

The normalized energy is

    Psi(F) = Psi_raw(F) - Psi_raw(I) - P_raw(I):(F - I),

so both Psi(I) and dPsi/dF(I) must vanish to floating-point precision.

Run:
    python -m tests.verify_reference
"""
import torch

from src.constitutive import stress_and_tangent
from src.picnn import HyperelasticNN

torch.set_default_dtype(torch.float64)


def main():
    torch.manual_seed(0)
    model = HyperelasticNN().double()
    I = torch.eye(3).unsqueeze(0)

    psi_raw_I, P_raw_I = model.reference(I.dtype, I.device)
    psi_I = model.energy(I, ref=(psi_raw_I, P_raw_I))
    P_I, _ = stress_and_tangent(model, I)

    energy_error = psi_I.abs().max().item()
    stress_error = P_I.abs().max().item()
    raw_stress_scale = max(1.0, P_raw_I.abs().max().item())
    energy_tolerance = 1e-12
    # P(I) subtracts P_raw(I) from an independently evaluated derivative.
    # Judge the resulting cancellation residual relative to that raw scale.
    stress_tolerance = 10 * torch.finfo(P_I.dtype).eps * raw_stress_scale

    print("dtype:", psi_I.dtype)
    print(f"|Psi(I)|            = {energy_error:.3e}")
    print(f"max |P_ij(I)|       = {stress_error:.3e}")
    print(f"max |P_raw,ij(I)|   = {raw_stress_scale:.3e}")
    print(f"stress tolerance    = {stress_tolerance:.3e}")

    assert energy_error < energy_tolerance, "reference energy is not zero"
    assert stress_error <= stress_tolerance, "reference stress is not zero"
    print("V2 PASS: the normalized model is energy-free and stress-free at F = I.")


if __name__ == "__main__":
    main()
