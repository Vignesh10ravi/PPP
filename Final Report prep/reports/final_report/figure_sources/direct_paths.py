"""Direct-RVE validation of the frozen pICNN along the five loading paths."""
import json, sys, time
from argparse import Namespace
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
from benchmarks.rve_inverse_design import _evaluate_direct_rve, _load_model
from benchmarks.rve_volume_fraction_map import _model_values, _polar_rotation
from src.rve import make_periodic_particle_cell_mesh
from src.rve_dataset import particle_radius
torch.set_num_threads(1)
repo = Path(sys.argv[1]).parent
ckpt = repo / "rve_phase7_augmented_finetune12/training/partial_icnn_phase6_phase7_finetuned.pt"
out = Path(sys.argv[2]); q = int(sys.argv[3]); modes_sel = sys.argv[4].split(",")
amps = [float(a) for a in sys.argv[5].split(",")]
vfs = [float(v) for v in sys.argv[6].split(",")]
model, _, sha = _load_model(ckpt)
def Fof(mode, a):
    I = np.eye(3)
    return {"uniaxial_tension": np.diag([1+a,1,1]), "uniaxial_compression": np.diag([1-a,1,1]),
            "simple_shear": I + a*np.outer([1,0,0],[0,1,0]),
            "equibiaxial_tension": np.diag([1+a,1+a,1]),
            "volumetric_expansion": (1+a)**(1/3)*I}[mode]
res = json.load(open(out)) if out.exists() else []
done = {(r["mode"], r["vf"], r["amp"]) for r in res}
for vf in vfs:
    args = Namespace(maxh=0.22, mu=1.0, bulk_modulus=1e4, displacement_order=2, integration_order=q,
                     microscopic_jacobian_floor=1e-8, response="energy", i=0, j=0, rve_verbose=False)
    args.mesh = make_periodic_particle_cell_mesh(radius=particle_radius(vf), maxh=0.22)
    for mode in modes_sel:
        for a in amps:
            if (mode, vf, a) in done: continue
            F = Fof(mode, a); args.initial_rotation = _polar_rotation(F)
            t = time.time()
            try:
                d = _evaluate_direct_rve(vf, F, args); ok = True
            except Exception as e:
                print("FAIL", mode, vf, a, e, flush=True); continue
            e, P = _model_values(model, F, vf)
            res.append({"mode": mode, "vf": vf, "amp": a, "F": F.tolist(),
                        "energy_rve": float(d["energy"]), "stress_rve": np.asarray(d["stress"]).tolist(),
                        "energy_nn": float(e), "stress_nn": P.detach().numpy().tolist(),
                        "rotation": np.asarray(d.get("rotation", np.eye(3))).tolist() if "rotation" in d else None,
                        "seconds": time.time() - t})
            json.dump(res, open(out, "w"))
            print(f"{mode:22s} vf={vf} a={a}: W_rve={d['energy']:.5g} W_nn={float(e):.5g} ({time.time()-t:.1f}s)", flush=True)
