"""Per-record diagnostics for figures: frozen pICNN vs direct RVE on the Phase 7
independent test split, plus the training-set determinant distribution.

python -I prep_record_data.py <picnn dir> <frozen.pt> <phase7.npz> <training.npz> <out_dir>
"""
import csv
import json
import sys

import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
from picnn import HyperelasticNN  # noqa: E402

torch.set_default_dtype(torch.float64)
ck = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
model = HyperelasticNN(n_ctx=1)
model.load_state_dict(ck["state_dict"])
model.eval()
out = sys.argv[5]

d = np.load(sys.argv[3])
keep = (d["split"] == "test") & (np.linalg.norm(d["deformation_gradient"] - np.eye(3), axis=(1, 2)) > 1e-12)
F, vf, P_ref = d["deformation_gradient"][keep], d["volume_fraction"][keep], d["stress"][keep]
P = np.empty_like(P_ref)
for v in np.unique(vf):
    idx = np.where(vf == v)[0]
    c = torch.full((len(idx), 1), float(v))
    Ft = torch.tensor(F[idx], requires_grad=True)
    e = model.energy(Ft, c, ref=model.reference(torch.float64, "cpu", c[:1]))
    (g,) = torch.autograd.grad(e.sum(), Ft)
    P[idx] = g.numpy()


def split(F, P):
    J = np.linalg.det(F)
    s = np.einsum("niK,njK->nij", P, F) / J[:, None, None]
    p = np.trace(s, axis1=1, axis2=2) / 3
    return J, p, s - p[:, None, None] * np.eye(3)


J, p_ref, s_ref = split(F, P_ref)
_, p, s = split(F, P)
with open(f"{out}/phase7_record_errors.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["J", "vf", "p_ref", "p_err_abs", "dev_ref_norm", "dev_err_norm"])
    for i in range(len(J)):
        w.writerow([J[i], vf[i], p_ref[i], abs(p[i] - p_ref[i]),
                    np.linalg.norm(s_ref[i]), np.linalg.norm(s[i] - s_ref[i])])

t = np.load(sys.argv[4])
Jt = np.linalg.det(t["deformation_gradient"])
hist, edges = np.histogram(Jt, bins=np.linspace(0.5, 1.85, 28))
json.dump({"counts": hist.tolist(), "edges": edges.tolist(), "n": int(len(Jt)),
           "energy_rms": float(np.sqrt(np.mean(t["energy"] ** 2))),
           "stress_rms": float(np.sqrt(np.mean(t["stress"] ** 2))),
           "normalization": {k: float(v) for k, v in ck["normalization"].items() if k in ("energy_scale", "stress_scale")},
           "best_validation": {k: float(v) for k, v in ck["best_validation"].items()}},
          open(f"{out}/training_J_histogram.json", "w"), indent=1)
print("records", len(J))
