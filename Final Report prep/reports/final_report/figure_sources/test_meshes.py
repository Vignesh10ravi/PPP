import json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from ngsolve import Mesh, BND
from netgen.occ import Box, Pnt, OCCGeometry, X
from tests import verify_fe_basics as fb
def surf(mesh):
    V = np.array([v.point for v in mesh.vertices])
    T = [[v.nr for v in el.vertices] for el in mesh.Elements(BND)]
    return {"V": V.tolist(), "T": T, "ne": mesh.ne}
out = {}
for name, h in (("v4", 0.4), ("v56", 0.3)):
    box = Box(Pnt(0, 0, 0), Pnt(1, 1, 1)); box.faces.Min(X).name = "left"; box.faces.Max(X).name = "right"
    out[name] = surf(Mesh(OCCGeometry(box).GenerateMesh(maxh=h)))
for k in ("tet", "hex"):
    m = fb.single_element_mesh(k)
    out[k] = {"V": [list(v.point) for v in m.vertices], "ne": m.ne}
json.dump(out, open(sys.argv[2], "w")); print({k: v.get("ne") for k, v in out.items()})
