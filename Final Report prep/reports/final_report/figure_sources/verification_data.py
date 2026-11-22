"""Extract plotting data for the verification figures by calling the test modules."""
import json, sys
from pathlib import Path
import numpy as np, torch
root = Path(sys.argv[1]); sys.path.insert(0, str(root)); out = Path(sys.argv[2])
import os; os.chdir(root)
import ngsolve as ngs
from tests import verify_fe_basics as fb
from tests import verify_stress_control as sc
from src.ngsolve_umat import assemble_system
from src.train import load_model
from src.constitutive import stress_and_tangent
from src.neohooke import P_neohooke, psi_neohooke, sample_F
P1 = "checkpoints/phase1_validated.pt"
RVE = str(root.parent / "rve_phase7_augmented_finetune12/training/partial_icnn_phase6_phase7_finetuned.pt")
res = {}
# ---- V0d/V0e: element stiffness spectra at F = I
def spec(mesh, model, order, q, ctx=None):
    fes = ngs.VectorH1(mesh, order=order); gfu = ngs.GridFunction(fes)
    _, K, _, _ = assemble_system(mesh, fes, gfu, model, q, context=ctx)
    Kd = np.array(K.ToDense()) if hasattr(K, "ToDense") else fb._to_dense(K, fes.ndof)
    lam = np.linalg.eigvalsh(0.5 * (Kd + Kd.T)); return (lam / np.abs(lam).max()).tolist()
models = {"neo-Hookean (analytical)": (fb.AnalyticalNeoHooke(), None),
          "RVE pICNN": (fb.load_rve_model(RVE), 0.05), "Phase 1 pICNN": (load_model(P1), None)}
res["v0"] = {}
for kind, order, q in (("tet", 1, 2), ("hex", 1, 2), ("hex", 2, 4)):
    em = fb.single_element_mesh(kind)
    for name, (m, ctx) in models.items():
        res["v0"][f"{kind}{order}|{name}"] = spec(em, m, order, q, ctx)
res["v0"]["hex1_onepoint|neo-Hookean (analytical)"] = spec(fb.single_element_mesh("hex"), models["neo-Hookean (analytical)"][0], 1, 0)
# ---- V2b: Phase 1 network against analytical neo-Hooke on 256 validation states
model = load_model(P1); g = torch.Generator().manual_seed(2026)
F = sample_F(256, max_strain=0.2, generator=g).double(); ref = model.reference(F.dtype, F.device)
with torch.no_grad():
    pe, Pe, pn = psi_neohooke(F), P_neohooke(F), model.energy(F, ref=ref)
Pn, _ = stress_and_tangent(model, F)
res["v2b"] = {"psi_exact": pe.tolist(), "psi_nn": pn.detach().tolist(),
              "P_exact": Pe.reshape(-1, 9).tolist(), "P_nn": Pn.detach().reshape(-1, 9).tolist(),
              "J": torch.det(F).tolist()}
# ---- V8: uniaxial stress-controlled response, reference curves and Phase 1 network
def P_net(Fm):
    Ft = torch.as_tensor(Fm, dtype=torch.float64).unsqueeze(0)
    P, _ = stress_and_tangent(model, Ft); return P.detach().numpy()[0]
ts = np.linspace(0.0, 0.6, 25)
res["v8"] = {"t": ts.tolist()}
for key, fun in (("svk", sc.P_svk), ("neohooke", sc.P_nh), ("phase1", P_net)):
    ab = [sc.uniaxial_reference(fun, t) if t > 0 else np.array([1.0, 1.0]) for t in ts]
    res["v8"][key] = np.array(ab).tolist()
json.dump(res, open(out, "w")); print("written", out)
