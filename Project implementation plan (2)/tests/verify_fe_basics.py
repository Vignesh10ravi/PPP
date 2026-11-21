"""V0 -- Mandatory finite-element basics (PPP guideline, Sec. 4.2.2, item 2).

Checks the discretisation layer that every later solve relies on, using the
same NGSolve spaces and the same production assembly routine
(`src.ngsolve_umat.assemble_system`) as the Phase 1 and macro-FE solvers.

  V0a  Partition of unity of the vertex basis (sum N_i = 1, sum dN_i = 0)
       for H1 order 1 and 2 on an unstructured tetrahedral mesh.
  V0b  Exact reproduction of a linear field and of its gradient (B-matrix
       check) for order 1 and 2.
  V0c  Volume and face area from numerical quadrature of the mapping
       Jacobian after an affine deformation of the mesh; compared with
       det(Fa) and with Nanson's formula.
  V0d  Zero eigenvalues of the unconstrained element stiffness at F = I:
       exactly six (three translations + three rotations) for a single
       irregular tetrahedron and a single hexahedron, for the analytical
       neo-Hookean law and for the Phase 1 pICNN; the six null vectors must
       span the rigid-body modes.
       The legacy Phase 1 checkpoint (affine reference correction) is
       reported, not asserted: its energy is not frame-indifferent, so only
       the three translations are zero modes (see report, Test V0d).
  V0e  Under-integration: a hexahedron with a one-point rule must show the
       12 additional zero-energy (hourglass) modes, i.e. 18 zero eigenvalues.

Run:  python -m tests.verify_fe_basics [--ckpt checkpoints/phase1_validated.pt]
                                       [--rve-ckpt <partial_icnn_*.pt> --vf 0.05]
"""
import argparse

import numpy as np
import torch
import ngsolve as ngs
from netgen.meshing import Element2D, Element3D, FaceDescriptor, MeshPoint
from netgen.meshing import Mesh as NetgenMesh
from netgen.csg import Pnt
from netgen.occ import Box, OCCGeometry, X
from netgen.occ import Pnt as OPnt

from src.ngsolve_umat import assemble_system
from tests.verify_fe_comparison import AnalyticalNeoHooke

torch.set_default_dtype(torch.float64)
TOL = 1.0e-12


def unit_cube_mesh(maxh=0.35):
    box = Box(OPnt(0, 0, 0), OPnt(1, 1, 1))
    box.faces.Min(X).name = "left"
    box.faces.Max(X).name = "right"
    return ngs.Mesh(OCCGeometry(box).GenerateMesh(maxh=maxh))


def single_element_mesh(kind):
    """One irregular tetrahedron or one (affinely distorted) hexahedron."""
    m = NetgenMesh(dim=3)
    m.SetMaterial(1, "solid")
    m.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    m.SetBCName(0, "outer")
    if kind == "tet":
        xyz = [(0.0, 0.0, 0.0), (1.2, 0.1, 0.0), (0.2, 0.9, 0.1), (0.1, 0.3, 1.1)]
        p = [m.Add(MeshPoint(Pnt(*c))) for c in xyz]
        m.Add(Element3D(1, p))
        for f in ((p[0], p[2], p[1]), (p[0], p[1], p[3]), (p[1], p[2], p[3]), (p[0], p[3], p[2])):
            m.Add(Element2D(1, list(f)))
    else:
        base = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
        distort = np.array([[1.10, 0.15, 0.00], [0.05, 0.95, 0.10], [0.00, 0.08, 1.20]])
        p = [m.Add(MeshPoint(Pnt(*(distort @ c)))) for c in base]
        m.Add(Element3D(1, p))
        quads = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
        for q in quads:
            m.Add(Element2D(1, [p[i] for i in q]))
    return ngs.Mesh(m)


def sample_points(mesh, n=200, seed=0):
    """Random points inside elements, returned as mesh points for evaluation."""
    rng = np.random.default_rng(seed)
    pts = []
    for el in mesh.Elements(ngs.VOL):
        verts = np.array([mesh[v].point for v in el.vertices])
        for _ in range(max(1, n // mesh.ne)):
            w = rng.dirichlet(np.ones(len(verts)))
            pts.append(w @ verts)
    return [mesh(*p) for p in pts]


def v0a_partition_of_unity(mesh):
    rows = []
    for order in (1, 2):
        fes = ngs.H1(mesh, order=order)
        gf = ngs.GridFunction(fes)
        gf.vec[:] = 0.0
        # In NGSolve's hierarchical H1 basis the first nv DOFs are the vertex
        # (barycentric) functions; their sum must be one everywhere.
        for i in range(mesh.nv):
            gf.vec[i] = 1.0
        pts = sample_points(mesh)
        val = max(abs(gf(p) - 1.0) for p in pts)
        grad = max(np.max(np.abs(ngs.Grad(gf)(p))) for p in pts)
        rows.append((order, fes.ndof, val, grad))
        assert val < TOL and grad < 1e-10, f"partition of unity violated for order {order}"
    return rows


def v0b_linear_field(mesh):
    a = np.array([0.3, -1.2, 0.7])
    rows = []
    for order in (1, 2):
        fes = ngs.H1(mesh, order=order)
        gf = ngs.GridFunction(fes)
        gf.Set(0.4 + a[0] * ngs.x + a[1] * ngs.y + a[2] * ngs.z)
        pts = sample_points(mesh, seed=1)
        err_v = max(abs(gf(p) - (0.4 + a @ np.array(mesh_point_xyz(mesh, p)))) for p in pts)
        err_g = max(np.max(np.abs(np.array(ngs.Grad(gf)(p)) - a)) for p in pts)
        rows.append((order, err_v, err_g))
        assert err_g < 1e-10, f"linear-field gradient not reproduced for order {order}"
    return rows


def mesh_point_xyz(mesh, mp):
    return ngs.CF((ngs.x, ngs.y, ngs.z))(mp)


def v0c_volume_and_area():
    mesh = unit_cube_mesh(0.4)
    Fa = np.array([[1.10, 0.05, 0.00], [0.02, 0.95, 0.04], [0.00, 0.03, 1.07]])
    fes = ngs.VectorH1(mesh, order=1)
    disp = ngs.GridFunction(fes)
    H = Fa - np.eye(3)
    disp.Set(ngs.CF(tuple(H[i, 0] * ngs.x + H[i, 1] * ngs.y + H[i, 2] * ngs.z for i in range(3))))
    vol0 = ngs.Integrate(ngs.CF(1.0), mesh)
    area0 = ngs.Integrate(ngs.CF(1.0), mesh, ngs.BND, definedon=mesh.Boundaries("right"))
    mesh.SetDeformation(disp)
    vol1 = ngs.Integrate(ngs.CF(1.0), mesh)
    area1 = ngs.Integrate(ngs.CF(1.0), mesh, ngs.BND, definedon=mesh.Boundaries("right"))
    mesh.UnsetDeformation()
    N = np.array([1.0, 0.0, 0.0])
    area_nanson = np.linalg.det(Fa) * np.linalg.norm(np.linalg.inv(Fa).T @ N)
    errs = (abs(vol0 - 1.0), abs(area0 - 1.0), abs(vol1 - np.linalg.det(Fa)), abs(area1 - area_nanson))
    assert max(errs) < 1e-12, "volume/area quadrature check failed"
    return {"V_ref": vol0, "A_ref": area0, "V_def": vol1, "detFa": float(np.linalg.det(Fa)),
            "A_def": area1, "A_nanson": float(area_nanson), "max_error": float(max(errs))}


def rigid_modes(fes, mesh):
    """Columns: three translations and three infinitesimal rotations."""
    vecs = []
    for cf in [ngs.CF((1, 0, 0)), ngs.CF((0, 1, 0)), ngs.CF((0, 0, 1)),
               ngs.CF((-ngs.y, ngs.x, 0)), ngs.CF((0, -ngs.z, ngs.y)), ngs.CF((ngs.z, 0, -ngs.x))]:
        g = ngs.GridFunction(fes)
        g.Set(cf)
        vecs.append(g.vec.FV().NumPy().copy())
    return np.array(vecs).T


def stiffness_spectrum(mesh, model, order, int_order, context=None):
    fes = ngs.VectorH1(mesh, order=order)
    gfu = ngs.GridFunction(fes)  # u = 0  ->  F = I
    _, K, _, _ = assemble_system(mesh, fes, gfu, model, int_order, context=context)
    Kd = np.array(K.ToDense()) if hasattr(K, "ToDense") else _to_dense(K, fes.ndof)
    Kd = 0.5 * (Kd + Kd.T)
    lam, vec = np.linalg.eigh(Kd)
    thresh = 1e-9 * np.max(np.abs(lam))
    nzero = int(np.sum(np.abs(lam) < thresh))
    R = rigid_modes(fes, mesh)
    rigid_residual = float(np.linalg.norm(Kd @ R) / (np.linalg.norm(Kd) * np.linalg.norm(R)))
    # every rigid mode must lie in the numerical null space
    null = vec[:, np.abs(lam) < thresh]
    proj = null @ (null.T @ R)
    span_err = float(np.linalg.norm(R - proj) / np.linalg.norm(R))
    nonzero = np.abs(lam[np.abs(lam) >= thresh])
    return {"ndof": fes.ndof, "n_zero": nzero, "n_negative": int(np.sum(lam < -thresh)), "smallest_nonzero": float(nonzero.min()) if nonzero.size else 0.0,
            "largest": float(np.abs(lam).max()), "rigid_residual": rigid_residual, "rigid_span_error": span_err,
            "threshold": float(thresh)}


def _to_dense(K, n):
    D = np.zeros((n, n))
    e = K.CreateColVector()
    y = K.CreateColVector()
    for j in range(n):
        e[:] = 0.0
        e[j] = 1.0
        y.data = K * e
        D[:, j] = y.FV().NumPy()
    return D


def load_rve_model(path):
    from src.picnn import HyperelasticNN
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = HyperelasticNN(n_ctx=1).double()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def main(ckpt=None, rve_ckpt=None, vf=0.05):
    print("V0a/V0b on an unstructured unit-cube tetrahedral mesh")
    mesh = unit_cube_mesh()
    print(f"  mesh: {mesh.ne} tetrahedra, {mesh.nv} vertices")
    for order, ndof, val, grad in v0a_partition_of_unity(mesh):
        print(f"  V0a order {order}: ndof={ndof:4d}  max|sum N_i - 1|={val:.3e}  max|sum grad N_i|={grad:.3e}")
    for order, ev, eg in v0b_linear_field(mesh):
        print(f"  V0b order {order}: max value error={ev:.3e}  max gradient error={eg:.3e}")
    r = v0c_volume_and_area()
    print("V0c affine mesh deformation")
    print(f"  reference volume {r['V_ref']:.15f}, reference face area {r['A_ref']:.15f}")
    print(f"  deformed volume  {r['V_def']:.15f} vs det(Fa) {r['detFa']:.15f}")
    print(f"  deformed area    {r['A_def']:.15f} vs Nanson {r['A_nanson']:.15f}")
    print(f"  max error {r['max_error']:.3e}")

    models = {"analytical neo-Hookean": (AnalyticalNeoHooke(), None, True)}
    if rve_ckpt:
        models["RVE pICNN (vf=%.2f)" % vf] = (load_rve_model(rve_ckpt), vf, True)
    if ckpt:
        from src.train import load_model
        models["Phase 1 pICNN (legacy)"] = (load_model(ckpt), None, False)
    print("V0d/V0e unconstrained element stiffness at F = I (production assemble_system)")
    print(f"  {'element':6s} {'order':>5s} {'quad':>4s} {'model':24s} {'ndof':>4s} {'#zero':>5s} "
          f"{'#neg':>4s} {'min|lam|>0':>11s} {'max|lam|':>9s} {'|K R|':>9s} {'span err':>9s}")
    cases = [("tet", 1, 2), ("tet", 2, 4), ("hex", 1, 2), ("hex", 2, 4)]
    for kind, order, q in cases:
        em = single_element_mesh(kind)
        for name, (model, ctx, must_pass) in models.items():
            s = stiffness_spectrum(em, model, order, q, context=ctx)
            print(f"  {kind:6s} {order:5d} {q:4d} {name:24s} {s['ndof']:4d} {s['n_zero']:5d} {s['n_negative']:4d} "
                  f"{s['smallest_nonzero']:11.3e} {s['largest']:9.3e} {s['rigid_residual']:9.2e} {s['rigid_span_error']:9.2e}")
            if not must_pass:
                continue
            assert s["n_zero"] == 6, f"{kind} order {order}: expected 6 zero eigenvalues, found {s['n_zero']}"
            assert s["rigid_span_error"] < 1e-8, "null space does not span the rigid-body modes"
    em = single_element_mesh("hex")
    s = stiffness_spectrum(em, models["analytical neo-Hookean"][0], 1, 0)  # IRS order 0 = one point
    print(f"  hex        1    0 {'analytical neo-Hookean':24s} {s['ndof']:4d} {s['n_zero']:5d} {s['n_negative']:4d} "
          f"{s['smallest_nonzero']:11.3e} {s['largest']:9.3e} {s['rigid_residual']:9.2e} {s['rigid_span_error']:9.2e}"
          "   <- one-point rule")
    assert s["n_zero"] == 18, f"one-point hexahedron: expected 18 zero eigenvalues, found {s['n_zero']}"
    print("V0 PASS: shape functions, quadrature, rigid-body modes and hourglass detection verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=None, help="optional legacy Phase 1 checkpoint (reported only)")
    parser.add_argument("--rve-ckpt", default=None, help="optional RVE-trained pICNN checkpoint (asserted)")
    parser.add_argument("--vf", type=float, default=0.05, help="volume-fraction context for --rve-ckpt")
    a = parser.parse_args()
    main(a.ckpt, a.rve_ckpt, a.vf)
