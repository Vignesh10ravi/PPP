"""Polyconvex (partially) Input-Convex Neural Network hyperelastic surrogate.

Design guarantees (Ball 1977 polyconvexity => existence + SPD tangent):

1. Inputs are convex functions of the polyconvex arguments (F, cof F, det F):
       I1   = tr(C)        (convex in F)
       I2   = tr(cof C)    (convex in cof F)
       J    = det F        (the polyconvex variable itself)
       -lnJ                (convex in J; carries the volumetric growth so the
                            network can reproduce Neo-Hookean's *decreasing*
                            energy in J without breaking monotonicity)
2. All hidden/input weights are non-negative (softplus) and the activation
   (softplus) is convex and non-decreasing => output convex & non-decreasing in
   the features => polyconvex in F.
3. Reference normalization removes energy and stress at F = I while preserving
   frame indifference. For the isotropic reference stress P_raw(I) = p_I I,
   the correction is objective because it uses det(F):
       Psi_hat(F) = Psi_raw(F) - Psi_raw(I) - p_I (det(F) - 1)

The optional context path (volume fraction v_f) enters through an
*unconstrained* sub-network that modulates biases -- this is the "partial" in
pICNN and is used in Phase 2. Set n_ctx=0 for the Phase-1 proof of concept.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn


def invariant_features(F):
    """(..., 3, 3) -> (..., 4) feature vector [I1, I2, J, -lnJ]."""
    C = F.transpose(-1, -2) @ F
    I1 = C.diagonal(dim1=-2, dim2=-1).sum(-1)
    I2 = 0.5 * (I1 ** 2 - (C * C).sum((-1, -2)))
    J = torch.linalg.det(F)
    return torch.stack([I1, I2, J, -torch.log(J)], dim=-1)


class ICNN(nn.Module):
    """Scalar input-convex, non-decreasing network."""

    def __init__(self, n_in=4, n_ctx=0, widths=(32, 32, 32)):
        super().__init__()
        self.n_ctx = n_ctx
        dims = [n_in] + list(widths)
        # affine passthrough of inputs at every layer
        self.Wy = nn.ParameterList(
            [nn.Parameter(torch.randn(dims[l + 1], n_in) * 0.1) for l in range(len(widths))])
        # hidden-to-hidden (kept non-negative via softplus at forward time)
        self.Wz = nn.ParameterList(
            [nn.Parameter(torch.randn(dims[l + 1], dims[l]) * 0.1) for l in range(1, len(widths))])
        self.b = nn.ParameterList(
            [nn.Parameter(torch.zeros(dims[l + 1])) for l in range(len(widths))])
        self.out = nn.Parameter(torch.randn(widths[-1]) * 0.1)
        self.out_b = nn.Parameter(torch.zeros(()))
        if n_ctx:
            self.ctx = nn.Sequential(nn.Linear(n_ctx, 32), nn.Softplus(),
                                     nn.Linear(32, sum(widths)))
        self._widths = widths
        self._reset_parameters()

    @staticmethod
    def _init_positive(parameter, effective_value):
        """Initialize softplus(parameter) near a small positive value."""
        raw_value = math.log(math.expm1(effective_value))
        with torch.no_grad():
            parameter.normal_(mean=raw_value, std=0.05)

    def _reset_parameters(self):
        # Raw values near zero give softplus(0) ~= 0.693. Across wide hidden
        # layers this creates an excessively stiff initial model.
        for weight in self.Wy:
            self._init_positive(weight, effective_value=0.05)

        for weight in self.Wz:
            self._init_positive(weight, effective_value=0.02)

        self._init_positive(self.out, effective_value=0.05)

    def forward(self, y, c=None):
        cb = self.ctx(c) if self.n_ctx else None
        off, z = 0, None
        for l in range(len(self.Wy)):
            pre = Fn.linear(y, Fn.softplus(self.Wy[l])) + self.b[l]
            if z is not None:
                pre = pre + Fn.linear(z, Fn.softplus(self.Wz[l - 1]))
            if cb is not None:
                w = self._widths[l]
                pre = pre + cb[..., off:off + w]
                off += w
            z = Fn.softplus(pre)
        return Fn.linear(z, Fn.softplus(self.out)) + self.out_b


class HyperelasticNN(nn.Module):
    """ICNN wrapped with the stress-free, zero-energy reference normalization."""

    def __init__(self, n_ctx=0, reference_correction="pressure"):
        super().__init__()
        if reference_correction not in ("pressure", "affine"):
            raise ValueError("reference_correction must be 'pressure' or 'affine'")
        self.net = ICNN(n_in=4, n_ctx=n_ctx)
        self.n_ctx = n_ctx
        # "pressure": objective correction p_I (det F - 1) used from Phase 5 on.
        # "affine":   legacy Phase 1 correction P_raw(I) : (F - I).
        self.reference_correction = reference_correction
        self.register_buffer("feature_mean", torch.zeros(4))
        self.register_buffer("feature_scale", torch.ones(4))
        self.register_buffer("context_mean", torch.zeros(n_ctx))
        self.register_buffer("context_scale", torch.ones(n_ctx))

    def set_normalization(
        self,
        feature_mean,
        feature_scale,
        context_mean=None,
        context_scale=None,
    ):
        """Set positive affine input scalings fitted using training data only."""
        feature_mean = torch.as_tensor(
            feature_mean, dtype=self.feature_mean.dtype, device=self.feature_mean.device
        )
        feature_scale = torch.as_tensor(
            feature_scale, dtype=self.feature_scale.dtype, device=self.feature_scale.device
        )
        if feature_mean.shape != (4,) or feature_scale.shape != (4,):
            raise ValueError("feature normalization values must have shape (4,)")
        if not torch.isfinite(feature_mean).all() or not torch.isfinite(feature_scale).all():
            raise ValueError("feature normalization values must be finite")
        if torch.any(feature_scale <= 0):
            raise ValueError("feature scales must be positive")

        with torch.no_grad():
            self.feature_mean.copy_(feature_mean)
            self.feature_scale.copy_(feature_scale)
            if self.n_ctx:
                if context_mean is None or context_scale is None:
                    raise ValueError("context normalization is required when n_ctx > 0")
                context_mean = torch.as_tensor(
                    context_mean,
                    dtype=self.context_mean.dtype,
                    device=self.context_mean.device,
                )
                context_scale = torch.as_tensor(
                    context_scale,
                    dtype=self.context_scale.dtype,
                    device=self.context_scale.device,
                )
                if context_mean.shape != (self.n_ctx,) or context_scale.shape != (self.n_ctx,):
                    raise ValueError("context normalization shape does not match n_ctx")
                if not torch.isfinite(context_mean).all() or not torch.isfinite(context_scale).all():
                    raise ValueError("context normalization values must be finite")
                if torch.any(context_scale <= 0):
                    raise ValueError("context scales must be positive")
                self.context_mean.copy_(context_mean)
                self.context_scale.copy_(context_scale)

    def _raw(self, F, c=None):
        features = (invariant_features(F) - self.feature_mean) / self.feature_scale
        if self.n_ctx:
            if c is None:
                raise ValueError("context c is required for a partial ICNN")
            c = (c - self.context_mean) / self.context_scale
        elif c is not None:
            raise ValueError("context was supplied to an ICNN with n_ctx=0")
        return self.net(features, c)

    def reference(self, dtype, device, c=None):
        """Energy Psi_raw(I) and stress P_raw(I) at the undeformed state.

        Uses the functorch `jacrev` API (not requires_grad_/autograd.grad) so it
        is safe to compose with vmap/jacrev/hessian transforms.
        """
        from torch.func import jacrev
        I = torch.eye(3, dtype=dtype, device=device)
        cc = None if c is None else c.reshape(1, -1)
        psi_I = self._raw(I.unsqueeze(0), cc).squeeze()
        P_I = jacrev(lambda f: self._raw(f.unsqueeze(0), cc).squeeze())(I)
        return psi_I, P_I

    def energy(self, F, c=None, ref=None):
        """Normalized strain energy density; shape (...,).

        Psi_hat(F) = Psi_raw(F) - Psi_raw(I) - p_I (det(F) - 1),
        where p_I = trace(P_raw(I))/3.

        This gives zero energy and stress at F = I while keeping the correction
        frame-indifferent. The isotropic model has P_raw(I) = p_I I.

        `ref` = (psi_I, P_I) may be precomputed once (outside a functorch
        transform) and passed in to avoid recomputation per sample.
        """
        if ref is None:
            ref = self.reference(F.dtype, F.device, c)
        psi_I, P_I = ref
        psi = self._raw(F, c)
        if self.reference_correction == "affine":
            I = torch.eye(3, dtype=F.dtype, device=F.device)
            return psi - psi_I - (P_I * (F - I)).sum((-1, -2))
        reference_pressure = torch.diagonal(P_I).mean()
        J = torch.linalg.det(F)
        return psi - psi_I - reference_pressure * (J - 1.0)
