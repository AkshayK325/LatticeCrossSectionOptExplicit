"""One-off: freeze legacy outputs of optBeam.ipynb (example 3fz) into legacy_3fz.json.

Run from the repo root BEFORE changing src/ or the notebook:
    python tests/data/make_legacy_ref.py
It execs notebook cells 9-10 so propPredict/shapePredict are the original notebook versions.
"""
import json
import sys
import time
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from legacy_setup import load_vae, build_3fz, volume_max  # noqa: E402

net = load_vae()
ns = {'torch': torch, 'np': np, 'plt': plt, 'splineNet': net, 'add_thickness': False}
nb = json.load(open(ROOT / 'optBeam.ipynb'))
for i in (9, 10):
    exec(''.join(nb['cells'][i]['source']), ns)

ref = {}
x_probe = torch.tensor(np.random.default_rng(0).uniform(size=(6, 4)), dtype=torch.float64)
ref['probe_x'] = x_probe.tolist()
ref['propPredict'] = ns['propPredict'](x_probe).tolist()
cp, th = ns['shapePredict'](x_probe)
ref['shapePredict_cp'] = cp.tolist()
ref['shapePredict_theta'] = th.tolist()

for section in ('circle', 'z'):
    opt, x0 = build_3fz(section, ns['propPredict'])
    t = time.perf_counter()
    J, dJ = opt.objectiveSE(x0, Jac=True)
    vc, dvc = opt.VolumeCon(x0, con_value=0.5 * volume_max(opt))
    ref[section] = {'nEle': int(opt.frameFE.numEle), 'ndof': int(opt.frameFE.ndof),
                    'J': float(np.asarray(J).item()), 'dJ': dJ.reshape(-1).tolist(),
                    'vc': float(vc.item()), 'dvc': dvc.reshape(-1).tolist(),
                    'time_obj_grad_s': time.perf_counter() - t}

# short MMA run, circle, as the optimizer regression
opt, x0 = build_3fz('circle')
setup = {'objective': 'SE', 'constraints': {'Volume': 0.5 * volume_max(opt)}}
t = time.perf_counter()
xopt, fun = opt.optimizerRun(setup, x0.copy(), algorithm='MMA',
                             options={'maxiter': 10, 'disp': False, 'move_limit': 0.5, 'maxfun': 500,
                                      'kkttol': 1e-5, 'miniter': 10})
ref['mma_circle'] = {'obj_hist': [float(np.asarray(v).item()) for v in opt.objective_values],
                     'fun': float(np.asarray(fun).item()), 'xopt': xopt.reshape(-1).tolist(),
                     'time_s': time.perf_counter() - t}

json.dump(ref, open(ROOT / 'tests' / 'data' / 'legacy_3fz.json', 'w'))
print({k: (v['J'], v['nEle'], v['time_obj_grad_s']) for k, v in ref.items() if k in ('circle', 'z')})
print('MMA hist', ref['mma_circle']['obj_hist'], ref['mma_circle']['time_s'])
