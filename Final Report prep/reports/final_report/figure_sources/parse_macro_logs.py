"""Parse macro-FE run.log files into per-load-step records (stretch, energy, min J, Newton iterations)."""
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = sys.argv[3:]
head = re.compile(r"order=(\d+), integration order=(\d+)")
mh = re.compile(r"maxh=([0-9.]+)")
step = re.compile(r"\[load continuation\] attempt=\d+, accepted_step=(\d+), increment=[^,]+, axial_stretch=([0-9.eE+-]+)")
it = re.compile(r"iter\s+(\d+)\s+\|R_free\| = ([0-9.eE+-]+)\s+min\(J_q\) = ([0-9.eE+-]+)\s+energy = ([0-9.eE+-]+)")
conv = re.compile(r"\[solve\] Converged in (\d+) iterations")
out = {}
for r in runs:
    log = (root / r / "run.log").read_text(errors="ignore").splitlines()
    info = {"order": None, "integration_order": None, "maxh": None, "steps": []}
    cur = None
    for line in log:
        if info["order"] is None:
            m = head.search(line)
            if m:
                info["order"], info["integration_order"] = int(m.group(1)), int(m.group(2))
                info["maxh"] = float(mh.search(line).group(1))
        m = step.search(line)
        if m:
            cur = {"stretch": float(m.group(2)), "res": [], "minJ": None, "energy": None, "converged": False}
            continue
        m = it.search(line)
        if m and cur is not None:
            cur["res"].append(float(m.group(2)))
            cur["minJ"], cur["energy"] = float(m.group(3)), float(m.group(4))
            continue
        m = conv.search(line)
        if m and cur is not None:
            cur["converged"] = True
            cur["iterations"] = int(m.group(1))
            info["steps"].append(cur)
            cur = None
    out[r] = info
    s = info["steps"]
    print(r, "order", info["order"], "h", info["maxh"], "accepted", len(s),
          "final stretch", s[-1]["stretch"] if s else None, "final energy", s[-1]["energy"] if s else None)
json.dump(out, open(sys.argv[2], "w"), indent=1)
