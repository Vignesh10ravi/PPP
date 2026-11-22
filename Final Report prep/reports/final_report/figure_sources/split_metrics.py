"""Volumetric/deviatoric split of surrogate errors against stored direct-RVE labels.

Usage: python -I split_metrics.py <dir with picnn.py> <out.json> <ckpt_name>=<ckpt.pt> ... -- <set_name>=<dataset.npz> ...
For every non-identity record the Cauchy stress sigma = P F^T / J is split into
pressure p = tr(sigma)/3 and deviator s = sigma - p I.  Errors are reported for
both parts separately, grouped by loading regime, together with the
training-scale NRMSE used in the original report for comparison.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
from picnn import HyperelasticNN  # noqa: E402

torch.set_default_dtype(torch.float64)
out_path = sys.argv[2]
sep = sys.argv.index("--")
ckpts = dict(a.split("=", 1) for a in sys.argv[3:sep])
sets = dict(a.split("=", 1) for a in sys.argv[sep + 1:])


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = HyperelasticNN(n_ctx=1)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck["normalization"]


def predict(model, F, vf):
    psi = np.empty(len(F))
    P = np.empty((len(F), 3, 3))
    for v in np.unique(vf):
        idx = np.where(vf == v)[0]
        c = torch.full((len(idx), 1), float(v))
        Ft = torch.tensor(F[idx], requires_grad=True)
        ref = model.reference(torch.float64, "cpu", c[:1])
        e = model.energy(Ft, c, ref=ref)
        (g,) = torch.autograd.grad(e.sum(), Ft)
        psi[idx] = e.detach().numpy()
        P[idx] = g.numpy()
    return psi, P


def split(F, P):
    J = np.linalg.det(F)
    sig = np.einsum("niK,njK->nij", P, F) / J[:, None, None]
    p = np.trace(sig, axis1=1, axis2=2) / 3.0
    s = sig - p[:, None, None] * np.eye(3)
    return J, p, s


def regime(J, F):
    C = np.einsum("nki,nkj->nij", F, F)
    I1b = np.trace(C, axis1=1, axis2=2) * J ** (-2.0 / 3.0)
    lab = np.where(J < 0.7, "compression J<0.7",
          np.where(J > 1.4, "expansion J>1.4",
          np.where(np.abs(J - 1) < 0.02, "near-isochoric |J-1|<0.02", "intermediate")))
    return lab, I1b


report = {}
for sname, spath in sets.items():
    d = np.load(spath)
    F = d["deformation_gradient"]
    keep = np.linalg.norm(F - np.eye(3), axis=(1, 2)) > 1e-12
    if "solver_status" in d:
        keep &= np.array([str(x).lower() in ("ok", "success", "converged", "complete", "1", "true") or "ok" in str(x).lower() for x in d["solver_status"]])
    if "split" in d and sname != "shear_final_holdout":
        keep &= d["split"] == "test"
    F = F[keep]
    vf = d["volume_fraction"][keep]
    psi_ref = d["energy"][keep]
    P_ref = d["stress"][keep]
    J, p_ref, s_ref = split(F, P_ref)
    lab, I1b = regime(J, F)
    report[sname] = {"n_records": int(keep.sum()), "models": {}}
    for cname, cpath in ckpts.items():
        model, norm = load(cpath)
        psi, P = predict(model, F, vf)
        _, p, s = split(F, P)
        es, ss = float(norm["energy_scale"]), float(norm["stress_scale"])
        dev_ref = np.linalg.norm(s_ref, axis=(1, 2))
        dev_err = np.linalg.norm(s - s_ref, axis=(1, 2))
        res = {
            "energy_nrmse_training_scale": float(np.sqrt(np.mean((psi - psi_ref) ** 2)) / es),
            "stress_nrmse_training_scale": float(np.sqrt(np.mean((P - P_ref) ** 2)) / ss),
            "groups": {},
        }
        for g in ["ALL"] + sorted(set(lab)):
            m = np.ones(len(J), bool) if g == "ALL" else lab == g
            if not m.any():
                continue
            rel_p = np.abs(p[m] - p_ref[m]) / np.maximum(np.abs(p_ref[m]), 1e-12)
            rel_s = dev_err[m] / np.maximum(dev_ref[m], 1e-12)
            res["groups"][g] = {
                "n": int(m.sum()),
                "pressure_rel_err_median_pct": float(100 * np.median(rel_p)),
                "deviatoric_rel_err_median_pct": float(100 * np.median(rel_s)),
                "deviatoric_rel_err_mean_pct": float(100 * np.mean(rel_s)),
                "deviatoric_rel_err_p90_pct": float(100 * np.percentile(rel_s, 90)),
                "rms_pressure_ref": float(np.sqrt(np.mean(p_ref[m] ** 2))),
                "rms_deviator_norm_ref": float(np.sqrt(np.mean(dev_ref[m] ** 2))),
                "rms_deviator_error": float(np.sqrt(np.mean(dev_err[m] ** 2))),
                "deviator_error_over_stress_scale": float(np.sqrt(np.mean(dev_err[m] ** 2)) / ss),
            }
        report[sname]["models"][cname] = res

# small-strain moduli at F = I for each checkpoint
from torch.func import hessian  # noqa: E402

report["small_strain_moduli"] = {}
I = torch.eye(3)
V = np.eye(3).reshape(9) / np.sqrt(3)
for cname, cpath in ckpts.items():
    model, _ = load(cpath)
    rows = {}
    for v in (0.02, 0.05, 0.10):
        c = torch.tensor([[v]])
        A = hessian(lambda Fm: model.energy(Fm.unsqueeze(0), c).squeeze())(I).reshape(9, 9).detach().numpy()
        rows[f"{v:.2f}"] = {"mu_eff": float(A[1, 1]), "K_eff": float(V @ A @ V / 3.0)}
    report["small_strain_moduli"][cname] = rows

json.dump(report, open(out_path, "w"), indent=2)
print(json.dumps(report, indent=1))
