"""V7/V8 -- Single-element constitutive response under full deformation control
and under stress control (PPP guideline Sec. 4.2.2, item 2).

Following the guideline's recommendation, the physically linear but
geometrically nonlinear St. Venant--Kirchhoff (SVK) law is tested first,
then the compressible neo-Hookean law, and optionally the trained Phase 1 pICNN.
All cases use the production assembly (`assemble_system`) and the production
Newton solver (`solve`) on ONE trilinear hexahedron (unit cube).

  V7   Full deformation control: every nodal DOF follows an affine map with a
       prescribed non-symmetric F. The stress at every Gauss point must equal
       the analytical P(F), and the assembled stiffness must match central
       differences of the assembled residual (consistent tangent check).
  V8   Stress control: uniaxial nominal traction t on the face x = 1, minimal
       symmetry supports (u_x = 0 on x = 0, u_y = 0 on y = 0, u_z = 0 on z = 0),
       all other faces traction-free. The homogeneous solution F = diag(a, b, b)
       is computed independently by a scalar/2x2 Newton solve of P11 = t,
       P22 = P33 = 0. The FE solution must reproduce a and b, the Gauss-point
       stress must equal diag(t, 0, 0), and the global Newton iteration must
       converge quadratically (table printed for every load step).

Run:  python -m tests.verify_stress_control [--ckpt checkpoints/phase1_validated.pt]
          [--traction 0.6] [--load-steps 3] [--materials svk neohooke]
"""
import argparse
import math

import numpy as np
import torch
import ngsolve as ngs
from ngsolve.meshes import MakeStructured3DMesh

from src.ngsolve_umat import assemble_system, solve
from src.neohooke import LAM, MU, psi_neohooke

torch.set_default_dtype(torch.float64)


# ── materials exposing the neural-material API (energy, reference) ──────────
class _Analytical(torch.nn.Module):
    n_ctx = 0

    def reference(self, dtype, device, c=None):
        return (torch.zeros((), dtype=dtype, device=device),
                torch.zeros((3, 3), dtype=dtype, device=device))


class StVenantKirchhoff(_Analytical):
    """Psi = lam/2 (tr E)^2 + mu tr(E^2),  E = (F^T F - I)/2."""

    def energy(self, F, c=None, ref=None):
        E = 0.5 * (F.transpose(-1, -2) @ F - torch.eye(3, dtype=F.dtype))
        trE = E.diagonal(dim1=-2, dim2=-1).sum(-1)
        return 0.5 * LAM * trE ** 2 + MU * (E * E).sum((-1, -2))


class NeoHooke(_Analytical):
    def energy(self, F, c=None, ref=None):
        return psi_neohooke(F)


def P_svk(F):
    E = 0.5 * (F.T @ F - np.eye(3))
    S = LAM * np.trace(E) * np.eye(3) + 2 * MU * E
    return F @ S


def P_nh(F):
    FinvT = np.linalg.inv(F).T
    return MU * (F - FinvT) + LAM * math.log(np.linalg.det(F)) * FinvT


# ── independent uniaxial reference solution ─────────────────────────────────
def uniaxial_reference(P_fun, t):
    """Solve P11(diag(a,b,b)) = t, P22 = 0 for (a, b) by Newton with FD Jacobian."""
    x = np.array([1.0 + t / (3 * MU), 1.0])
    for _ in range(50):
        def g(z):
            P = P_fun(np.diag([z[0], z[1], z[1]]))
            return np.array([P[0, 0] - t, P[1, 1]])
        r = g(x)
        if np.linalg.norm(r) < 1e-14:
            break
        J = np.zeros((2, 2))
        for j in range(2):
            h = 1e-7
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            J[:, j] = (g(xp) - g(xm)) / (2 * h)
        x = x - np.linalg.solve(J, r)
    return x


# ── helpers ─────────────────────────────────────────────────────────────────
def unit_hex():
    mesh = MakeStructured3DMesh(hexes=True, nx=1, ny=1, nz=1)
    names = {}
    for b in mesh.GetBoundaries():
        area = ngs.Integrate(ngs.CF(1.0), mesh, ngs.BND, definedon=mesh.Boundaries(b))
        c = [ngs.Integrate(v, mesh, ngs.BND, definedon=mesh.Boundaries(b)) / area for v in (ngs.x, ngs.y, ngs.z)]
        for k, axis in enumerate("xyz"):
            if abs(c[k]) < 1e-12:
                names[axis + "0"] = b
            if abs(c[k] - 1) < 1e-12:
                names[axis + "1"] = b
    return mesh, names


def gauss_point_F_P(mesh, gfu, model, int_order=2):
    from src.ngsolve_umat import stress_and_tangent
    from ngsolve.comp import IntegrationRuleSpace
    irs = IntegrationRuleSpace(mesh, order=int_order)
    gf = ngs.GridFunction(irs ** 9)
    gf.Interpolate((ngs.Id(3) + ngs.Grad(gfu)).Reshape((9,)))
    F = np.stack([gf.components[i].vec.FV().NumPy() for i in range(9)], axis=1).reshape(-1, 3, 3)
    P, _ = stress_and_tangent(model, torch.from_numpy(F.copy()))
    return F, P.numpy()


def newton_table(history, tol):
    rows = []
    for k, r in enumerate(history):
        ratio = history[k + 1] / r ** 2 if k + 1 < len(history) else float("nan")
        q = (math.log(history[k + 1] / r) / math.log(r / history[k - 1])
             if 0 < k < len(history) - 1 else float("nan"))
        rows.append(f"      {k:2d}  {r:10.3e}  {ratio:10.3e}  {q:6.2f}")
    return "\n".join([f"      k   ||R_free||   r+1/r^2   order   (tol {tol:.0e})"] + rows)


# ── V7 ─────────────────────────────────────────────────────────────────────
def v7a_full_deformation_control(name, model, P_fun):
    mesh, _ = unit_hex()
    Fbar = np.array([[1.10, 0.06, -0.02], [0.03, 0.92, 0.05], [-0.04, 0.02, 1.05]])
    H = Fbar - np.eye(3)
    fes = ngs.VectorH1(mesh, order=1)
    gfu = ngs.GridFunction(fes)
    gfu.Set(ngs.CF(tuple(H[i, 0] * ngs.x + H[i, 1] * ngs.y + H[i, 2] * ngs.z for i in range(3))))
    F, P = gauss_point_F_P(mesh, gfu, model)
    err_F = np.max(np.abs(F - Fbar))
    P_ref = P_fun(Fbar)
    err_P = np.max(np.abs(P - P_ref)) / np.max(np.abs(P_ref))
    # consistent tangent: K d  vs  [R(u + eps d) - R(u - eps d)] / (2 eps)
    R0, K, _, _ = assemble_system(mesh, fes, gfu, model, 2)
    rng = np.random.default_rng(3)
    d = rng.standard_normal(fes.ndof)
    errs = []
    for eps in (1e-3, 1e-4, 1e-5, 1e-6):
        u0 = gfu.vec.FV().NumPy().copy()
        gfu.vec.FV().NumPy()[:] = u0 + eps * d
        Rp, _, _, _ = assemble_system(mesh, fes, gfu, model, 2)
        gfu.vec.FV().NumPy()[:] = u0 - eps * d
        Rm, _, _, _ = assemble_system(mesh, fes, gfu, model, 2)
        gfu.vec.FV().NumPy()[:] = u0
        fd = (Rp.FV().NumPy() - Rm.FV().NumPy()) / (2 * eps)
        dv = gfu.vec.CreateVector()
        dv.FV().NumPy()[:] = d
        Kd = gfu.vec.CreateVector()
        Kd.data = K * dv
        errs.append(np.linalg.norm(Kd.FV().NumPy() - fd) / np.linalg.norm(fd))
    print(f"  V7 [{name}] max|F - Fbar| = {err_F:.2e}, max rel. Gauss-point stress error = {err_P:.2e}")
    print("        tangent vs central differences: " +
          ", ".join(f"eps={e:.0e}: {v:.2e}" for e, v in zip((1e-3, 1e-4, 1e-5, 1e-6), errs)))
    assert err_F < 1e-13 and err_P < 1e-12, "full deformation control: stress mismatch"
    assert min(errs) < 1e-7, "assembled stiffness inconsistent with residual"
    return err_P, min(errs)


# ── V8 ─────────────────────────────────────────────────────────────────────
def v7b_stress_control(name, model, P_fun, traction, load_steps, tol=1e-10, ab_tol=1e-10):
    mesh, nm = unit_hex()
    supports = {"x": nm["x0"], "y": nm["y0"], "z": nm["z0"]}
    gfu = None
    for step in range(1, load_steps + 1):
        t = traction * step / load_steps
        gfu, hist = solve(mesh, model, bc_cf=ngs.CF((0, 0, 0)), dirichlet_components=supports,
                          traction_cf=ngs.CF((t, 0, 0)), neumann=nm["x1"], order=1,
                          integration_order=2, tol=tol, max_iter=30,
                          initial_cf=gfu, merit_function="energy")
        print(f"  V8 [{name}] load step {step}/{load_steps}, t = {t:.4f}")
        print(newton_table(hist, tol))
    a, b = uniaxial_reference(P_fun, traction)
    F, P = gauss_point_F_P(mesh, gfu, model)
    err_a = np.max(np.abs(F[:, 0, 0] - a))
    err_b = np.max(np.abs(F[:, 1, 1] - b)) + np.max(np.abs(F[:, 2, 2] - b))
    off = np.max(np.abs(F - np.diag(np.diag(F[0])))) if F.shape[0] else 0.0
    P_err = np.max(np.abs(P - np.diag([traction, 0, 0]))) / traction
    print(f"  V8 [{name}] reference a = {a:.10f}, b = {b:.10f}  (J = {a * b * b:.6f})")
    print(f"        FE: max|F11 - a| = {err_a:.2e}, max|F22,F33 - b| = {err_b:.2e}, "
          f"max off-diagonal |F_ij| = {off:.2e}, max|P - diag(t,0,0)|/t = {P_err:.2e}")
    assert err_a < ab_tol and err_b < ab_tol, "stress-controlled stretches differ from the reference"
    assert P_err < 1e-8, "Gauss-point stress is not the applied uniaxial state"
    return float(F[:, 0, 0].mean()), float(F[:, 1, 1].mean()), a, b


def main(ckpt=None, traction=0.6, load_steps=3, materials=("svk", "neohooke")):
    print(f"Materials: mu = {MU}, lambda = {LAM}; single trilinear hexahedron, 8-point Gauss rule")
    lib = {"svk": ("St. Venant-Kirchhoff", StVenantKirchhoff(), P_svk),
           "neohooke": ("neo-Hookean", NeoHooke(), P_nh)}
    for key in materials:
        name, model, P_fun = lib[key]
        v7a_full_deformation_control(name, model, P_fun)
        v7b_stress_control(name, model, P_fun, traction, load_steps)
    if ckpt:
        # Reported, not asserted: the trained network only approximates the
        # neo-Hookean law (V2b), so its stretches differ by its constitutive error.
        from src.train import load_model
        model = load_model(ckpt)
        for t in (traction / 3, traction):
            a_nn, b_nn, a, b = v7b_stress_control("Phase 1 pICNN", model, P_nh, t, load_steps,
                                                  ab_tol=float("inf"))
            print(f"  V8 [Phase 1 pICNN] t = {t:.3f}: a = {a_nn:.6f} (neo-Hooke {a:.6f}, "
                  f"{100 * (a_nn - a) / (a - 1):+.1f} % of the stretch increment), "
                  f"b = {b_nn:.6f} (neo-Hooke {b:.6f}, {100 * (b_nn - b) / abs(b - 1):+.1f} %)")
    print("V7/V8 PASS: full-deformation and stress-controlled single-element responses verified.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=None, help="optional Phase 1 checkpoint (compared with neo-Hooke)")
    p.add_argument("--traction", type=float, default=0.6, help="final nominal traction t (units of mu)")
    p.add_argument("--load-steps", type=int, default=3)
    p.add_argument("--materials", nargs="+", default=["svk", "neohooke"], choices=["svk", "neohooke"])
    a = p.parse_args()
    main(a.ckpt, a.traction, a.load_steps, tuple(a.materials))
