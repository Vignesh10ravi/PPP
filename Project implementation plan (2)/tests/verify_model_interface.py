"""V1 -- Verify the public deformation and feature interfaces.

``HyperelasticNN.energy(F)`` is the constitutive interface and accepts
deformation gradients with shape ``(..., 3, 3)``. Precomputed invariant
features are accepted by the wrapped ICNN as ``model.net(feats)``; they must
not be passed to ``model.energy`` or directly to ``model``.

Run:
    python -m tests.verify_model_interface
"""
import torch

from src.constitutive import stress, stress_and_tangent
from src.picnn import HyperelasticNN, invariant_features

torch.set_default_dtype(torch.float64)


def main():
    torch.manual_seed(0)
    model = HyperelasticNN().double()
    F = torch.eye(3).repeat(4, 1, 1)
    F = F + 0.02 * torch.randn_like(F)
    assert torch.all(torch.linalg.det(F) > 0), "test samples require det(F) > 0"

    feats = invariant_features(F)
    raw_from_features = model.net(feats)
    raw_from_deformation = model._raw(F)
    assert raw_from_features.shape == (len(F),)
    assert torch.allclose(raw_from_features, raw_from_deformation)

    ref = model.reference(F.dtype, F.device)
    energy = model.energy(F, ref=ref)
    P, A = stress_and_tangent(model, F)
    P_only = stress(model, F)

    assert energy.shape == (len(F),)
    assert P.shape == (len(F), 3, 3)
    assert A.shape == (len(F), 3, 3, 3, 3)
    assert energy.dtype == P.dtype == A.dtype == torch.float64
    assert torch.allclose(P_only, P)
    assert torch.isfinite(energy).all()
    assert torch.isfinite(P).all()
    assert torch.isfinite(A).all()

    print("feature interface: model.net(invariant_features(F))")
    print("material interface: model.energy(F), model.reference(...), "
          "stress_and_tangent(model, F)")
    print("dtype:", energy.dtype)
    print("V1 PASS: model interfaces, shapes, dtype, and derivatives are consistent.")


if __name__ == "__main__":
    main()
