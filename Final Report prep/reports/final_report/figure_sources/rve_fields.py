"""Sample deformed RVE fields on the mid-plane z=0.5 for two load cases (vf=0.10)."""
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from ngsolve import Norm, sqrt, InnerProduct, Trace, Id
from src.free_particle import solve_free_rotating_particle
from src.rve import make_periodic_particle_cell_mesh
from src.rve_dataset import particle_radius
from benchmarks.rve_volume_fraction_map import _polar_rotation
vf = 0.10; r = particle_radius(vf)
mesh = make_periodic_particle_cell_mesh(radius=r, maxh=0.22)
cases = {"uniaxial_5pct": np.diag([1.05, 1.0, 1.0]),
         "shear_5pct": np.array([[1, 0.05, 0], [0, 1, 0], [0, 0, 1.0]])}
n = 141; xs = np.linspace(0.0, 1.0, n)
out = {"vf": vf, "radius": r, "x": xs.tolist(), "cases": {}}
for name, F in cases.items():
    st = solve_free_rotating_particle(mesh, F, initial_rotation=_polar_rotation(F), mu=1.0,
                                      bulk_modulus=1e4, displacement_order=2, integration_order=10,
                                      verbose=False)
    w = st.solution.components[0]; p = st.solution.components[1]
    Fcf = st.deformation_gradient
    C = Fcf.trans * Fcf
    J = sqrt(1e-30 + (Fcf[0,0]*(Fcf[1,1]*Fcf[2,2]-Fcf[1,2]*Fcf[2,1]) - Fcf[0,1]*(Fcf[1,0]*Fcf[2,2]-Fcf[1,2]*Fcf[2,0]) + Fcf[0,2]*(Fcf[1,0]*Fcf[2,1]-Fcf[1,1]*Fcf[2,0]))**2)
    Cbar = J**(-2/3) * C
    dev = sqrt(InnerProduct(Cbar - Trace(Cbar)/3*Id(3), Cbar - Trace(Cbar)/3*Id(3)))
    W = np.full((n, n, 3), np.nan); P = np.full((n, n), np.nan); D = np.full((n, n), np.nan)
    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            if (x-0.5)**2 + (y-0.5)**2 < (r*1.0005)**2: continue
            mp = mesh(min(max(x,1e-9),1-1e-9), min(max(y,1e-9),1-1e-9), 0.5)
            try:
                W[j, i] = w(mp); P[j, i] = p(mp); D[j, i] = dev(mp)
            except Exception:
                pass
    out["cases"][name] = {"F": F.tolist(), "rotation": np.asarray(st.rotation).tolist(),
                          "w": W.tolist(), "p": P.tolist(), "dev": D.tolist()}
    print(name, "done; |w|max=", np.nanmax(np.linalg.norm(W, axis=2)), "p range", np.nanmin(P), np.nanmax(P), flush=True)
json.dump(out, open(sys.argv[2], "w"))
