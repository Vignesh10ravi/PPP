"""Export RVE and macro mesh surfaces for the setup figures (run with PPP code on PYTHONPATH)."""
import json, sys, numpy as np
from ngsolve import Mesh, BND
from netgen.occ import Box, OCCGeometry, Pnt
from src.rve import make_periodic_particle_cell_mesh

def surf(mesh):
    P = np.array([v.point for v in mesh.vertices])
    out = {}
    for el in mesh.Elements(BND):
        out.setdefault(el.mat, []).append([v.nr for v in el.vertices])
    return P.tolist(), out

def tets(mesh):
    return [[v.nr for v in el.vertices] for el in mesh.Elements()]

r = (3 * 0.05 / (4 * np.pi)) ** (1 / 3)
rve = make_periodic_particle_cell_mesh(r, maxh=0.22)
P, S = surf(rve)
macro = Mesh(OCCGeometry(Box(Pnt(0, 0, 0), Pnt(1, 1, 1))).GenerateMesh(maxh=0.25))
Pm, Sm = surf(macro)
json.dump({"rve": {"points": P, "faces": S, "tets": tets(rve), "radius": r},
           "macro": {"points": Pm, "faces": Sm, "ne": macro.ne}},
          open(sys.argv[1], "w"))
print(rve.ne, len(P), list(S), macro.ne, list(Sm))
