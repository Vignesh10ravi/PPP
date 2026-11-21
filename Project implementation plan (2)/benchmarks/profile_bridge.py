"""Computation-time study of the macroscopic PyTorch--NGSolve bridge.

Solves the first load increment (axial stretch 5e-4) of the clamped-cube
macro problem with the RVE-trained pICNN and prints the built-in stage
timings of src.ngsolve_umat (TIMINGS), per Newton iteration.

Run:  python -m benchmarks.profile_bridge --checkpoint <frozen.pt> \
          [--maxh 0.25] [--order 1] [--threads 1] [--output timings.json]
"""
import argparse
import json
import os
import time

import torch
from netgen.occ import Box, OCCGeometry, Pnt, X
from ngsolve import CF, Mesh, x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--maxh", type=float, default=0.25)
    p.add_argument("--order", type=int, default=1)
    p.add_argument("--integration-order", type=int, default=4)
    p.add_argument("--stretch", type=float, default=5e-4)
    p.add_argument("--volume-fraction", type=float, default=0.05)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    os.environ["NGS_NUM_THREADS"] = str(a.threads)
    torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.float64)

    from src import ngsolve_umat as um
    from src.picnn import HyperelasticNN

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    model = HyperelasticNN(n_ctx=1).double()
    model.load_state_dict(ck["state_dict"])
    model.eval()

    box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    box.faces.Min(X).name = "left"
    box.faces.Max(X).name = "right"
    mesh = Mesh(OCCGeometry(box).GenerateMesh(maxh=a.maxh))
    bc = CF((a.stretch * x, 0, 0))

    um.reset_timings()
    t0 = time.perf_counter()
    gfu, history = um.solve(mesh, model, bc_cf=bc, dirichlet="left|right", order=a.order,
                            context=a.volume_fraction, initial_cf=bc,
                            integration_order=a.integration_order, merit_function="energy",
                            max_iter=60, tol=1e-8)
    wall = time.perf_counter() - t0
    t = um.get_timings()
    n_assembly = t["2 pICNN energy, stress, tangent (AD)"][1]
    from ngsolve.comp import IntegrationRuleSpace
    nq = IntegrationRuleSpace(mesh, order=a.integration_order).ndof
    print(f"mesh: {mesh.ne} elements, order {a.order}, quadrature points {nq}, "
          f"Newton iterations {len(history) - 1}, assemblies {n_assembly}")
    print(f"{'stage':42s} {'total s':>9s} {'calls':>6s} {'ms/call':>9s} {'share':>6s}")
    total_timed = sum(v[0] for v in t.values())
    for k in sorted(t):
        s, c = t[k]
        print(f"{k:42s} {s:9.3f} {c:6d} {1e3 * s / c:9.2f} {100 * s / wall:5.1f}%")
    print(f"{'other (Python control, line search, BCs)':42s} {wall - total_timed:9.3f}")
    print(f"{'total wall time':42s} {wall:9.3f}")
    if a.output:
        json.dump({"elements": mesh.ne, "order": a.order, "quadrature_points": nq,
                   "newton_iterations": len(history) - 1, "wall_seconds": wall,
                   "stages": {k: {"seconds": v[0], "calls": v[1]} for k, v in t.items()},
                   "threads": a.threads}, open(a.output, "w"), indent=1)


if __name__ == "__main__":
    main()
