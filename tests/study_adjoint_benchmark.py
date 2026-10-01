"""Memory/time: unrolled reverse AD vs checkpointed discrete adjoint (one case per process for clean peaks).
    PYTHONPATH=src:tests python tests/study_adjoint_benchmark.py <method unrolled|adjoint> <cpu|cuda> <elemSize> <T> [K]
Prints one JSON line."""
import json
import resource
import sys
import time

import torch

method, dev, es, T = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
K = int(sys.argv[5]) if len(sys.argv) > 5 else None
torch.set_default_device(dev)
torch.set_default_dtype(torch.float64)
from adjoint import DropStepper, checkpointed_adjoint
from dropGrad import nominal, evaluate, radii, placeholder_J
from latticeSetup import example3, drop_rig, n_design_rows
from material import project_material

MAT = project_material()
RIG = dict(law={'kind': 'crit_damped', 'F_ref': 2000.0, 'delta_ref': 0.2}, start='co_moving', gap0=0.1, energy_J=0.75)
SYM = 'Midplane'
lat = example3(elemSize=es)
x0 = torch.full((n_design_rows(SYM, lat), 1), 0.8)
rig0, sched = nominal(lat, x0, SYM, MAT, T=T, **RIG)
if dev == 'cuda':
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6            # GB (ru_maxrss in kB on Linux)
x = x0.clone().requires_grad_(True)
t0 = time.time()
if method == 'unrolled':
    J = evaluate(lat, x, SYM, MAT, rig0, sched, **RIG)
    g, = torch.autograd.grad(J, x)
    J = J.item()
else:
    kw = {k: v for k, v in RIG.items() if k != 'law'}
    rig = drop_rig(lat, radii(x, SYM, lat), rig0['law'], mat=MAT, freeze=rig0, **kw)
    J, (g,) = checkpointed_adjoint(DropStepper(rig, sched, placeholder_J), [x], K=K)
if dev == 'cuda':
    torch.cuda.synchronize()
wall = time.time() - t0
out = dict(method=method, device=dev, elemSize=es, T_ms=T * 1e3, steps=len(sched), ne=int(lat['connr'].shape[0]),
           K=K or round(len(sched) ** 0.5), wall_s=round(wall, 1), J=J, g_norm=g.norm().item(), g=g.flatten()[:3].tolist(),
           peak_rss_GB=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), rss_before_GB=round(rss0, 2))
if dev == 'cuda':
    out['peak_gpu_GB'] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
print(json.dumps(out), flush=True)
