"""Test loss (same definition as the training loss, eq. (3.x)) and parity data.

python -I test_loss_and_parity.py <dir with picnn.py> <out_dir> name=<ckpt.pt> ... -- set=<dataset.npz> ...
Writes <out_dir>/test_losses.json and <out_dir>/parity_<set>.csv (first checkpoint).
"""
import csv
import json
import sys

import numpy as np
import torch
from torch.func import hessian

sys.path.insert(0, sys.argv[1])
from picnn import HyperelasticNN  # noqa: E402

torch.set_default_dtype(torch.float64)
out = sys.argv[2]
sep = sys.argv.index("--")
ckpts = dict(a.split("=", 1) for a in sys.argv[3:sep])
sets = dict(a.split("=", 1) for a in sys.argv[sep + 1:])


def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=True)
    m = HyperelasticNN(n_ctx=1)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def evaluate(model, F, vf):
    psi = np.empty(len(F)); P = np.empty((len(F), 3, 3)); A = np.empty((len(F), 9, 9))
    for v in np.unique(vf):
        idx = np.where(vf == v)[0]
        c = torch.full((len(idx), 1), float(v))
        ref = model.reference(torch.float64, "cpu", c[:1])
        Ft = torch.tensor(F[idx], requires_grad=True)
        e = model.energy(Ft, c, ref=ref)
        (g,) = torch.autograd.grad(e.sum(), Ft)
        psi[idx] = e.detach().numpy(); P[idx] = g.numpy()
        c1 = c[:1]
        for j, i in enumerate(idx):
            A[i] = hessian(lambda Fm: model.energy(Fm.unsqueeze(0), c1, ref=ref).squeeze())(
                torch.tensor(F[i])).reshape(9, 9).detach().numpy()
    return psi, P, A


def cauchy_split(F, P):
    J = np.linalg.det(F)
    s = np.einsum("niK,njK->nij", P, F) / J[:, None, None]
    p = np.trace(s, axis1=1, axis2=2) / 3
    return p, s - p[:, None, None] * np.eye(3)


res = {}
first = list(ckpts)[0]
for sname, spath in sets.items():
    d = np.load(spath)
    keep = np.linalg.norm(d["deformation_gradient"] - np.eye(3), axis=(1, 2)) > 1e-12
    if "split" in d:
        keep &= d["split"] == "test"
    F, vf = d["deformation_gradient"][keep], d["volume_fraction"][keep]
    psi_r, P_r, A_r = d["energy"][keep], d["stress"][keep], d["tangent"][keep].reshape(-1, 9, 9)
    res[sname] = {"n": int(keep.sum())}
    for cname, cpath in ckpts.items():
        model, ck = load(cpath)
        n = ck["normalization"]; sE, sP = float(n["energy_scale"]), float(n["stress_scale"])
        w = ck["training_config"]
        psi, P, A = evaluate(model, F, vf)
        lE = float(np.mean(((psi - psi_r) / sE) ** 2))
        lP = float(np.mean(((P - P_r) / sP) ** 2))
        lA = float(np.mean(np.linalg.norm(A - A_r, axis=(1, 2)) ** 2 / np.linalg.norm(A_r, axis=(1, 2)) ** 2))
        res[sname][cname] = {"energy_loss_normalized": lE, "stress_loss_normalized": lP,
                             "tangent_relative_frobenius_mse": lA,
                             "loss_energy_plus_stress": lE + w.get("stress_weight", 1.0) * lP,
                             "objective": lE + w.get("stress_weight", 1.0) * lP + w.get("tangent_weight", 0.1) * lA}
        if cname == first:
            p_r, s_r = cauchy_split(F, P_r); p, s = cauchy_split(F, P)
            with open(f"{out}/parity_{sname}.csv", "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["J", "vf", "psi_ref", "psi", "p_ref", "p", "dev_ref", "dev", "A_relerr"])
                for i in range(len(F)):
                    wr.writerow([np.linalg.det(F[i]), vf[i], psi_r[i], psi[i], p_r[i], p[i],
                                 np.linalg.norm(s_r[i]), np.linalg.norm(s[i]),
                                 np.linalg.norm(A[i] - A_r[i]) / np.linalg.norm(A_r[i])])
json.dump(res, open(f"{out}/test_losses.json", "w"), indent=1)
print(json.dumps(res, indent=1))
