"""V3 -- Tangent finite-difference check (calculus check).

Proves the autograd tangent A is exact, independent of training. Central
differences on P(F) are compared to A over a range of perturbation sizes eps.

This directly answers Pruger's slide-8 question: the FD error is V-shaped,
bottoming near ~1e-8 at eps ~ 1e-6 (truncation error ~eps^2 fights round-off
~eps_mach/eps). Machine precision is UNREACHABLE by finite differences by
construction -- which is exactly the argument for using autograd.

Run:  python -m tests.verify_tangent
"""
import torch

from src.picnn import HyperelasticNN
from src.constitutive import stress_and_tangent

torch.set_default_dtype(torch.float64)


def central_diff_tangent(model, F, eps):
    n = F.shape[0]
    A = torch.zeros(n, 3, 3, 3, 3)
    for k in range(3):
        for l in range(3):
            dF = torch.zeros_like(F)
            dF[:, k, l] = eps
            Pp, _ = stress_and_tangent(model, F + dF)
            Pm, _ = stress_and_tangent(model, F - dF)
            A[..., k, l] = (Pp - Pm) / (2 * eps)
    return A


def main():
    torch.manual_seed(0)
    model = HyperelasticNN().double()
    F = torch.eye(3) + 0.05 * torch.randn(4, 3, 3)
    _, A_exact = stress_and_tangent(model, F)

    print(f"{'eps':>8}  {'rel.err':>10}")
    best = 1.0
    for eps in [1e-2, 1e-4, 1e-6, 1e-8]:
        A_fd = central_diff_tangent(model, F, eps)
        rel = ((A_fd - A_exact).norm() / A_exact.norm()).item()
        best = min(best, rel)
        print(f"{eps:8.0e}  {rel:10.2e}")

    assert best < 1e-6, "autograd tangent disagrees with finite differences"
    print("\nV3 PASS: autograd tangent matches central difference near machine-limited optimum.")


if __name__ == "__main__":
    main()
