"""Smoke / sanity run of the helmet crown patch (shell + curved C4 liner + free head + hemispherical anvil).
Usage: python -u tests/study_helmet_smoke.py [T_ms] [r_mm]   (CPU, one run). Reports to run_status.log."""
import sys
import time

import numpy as np
import torch

import runStatus as rs
from latticeSetup import example3, n_design_rows, element_values
from material import project_material
from helmetRig import helmet_rig, run_helmet
import jointRule

torch.set_default_dtype(torch.float64)
T = float(sys.argv[1]) * 1e-3 if len(sys.argv) > 1 else 0.5e-3
r0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
rs.start(f'helmet smoke T={T*1e3:.2f} ms r={r0}')
lat = example3(elemSize=3.0)
r = torch.full((len(lat['connr']),), r0)
rig = helmet_rig(lat, r, project_material(), joint=dict(jointRule.PRESETS['R5_C4']))
fr = rig['fr']
print('nodes', fr.N, 'beam elems', fr.ne, 'shell tris', rig['shell'].nt, 'head mass [t]', rig['M'],
      'E0 [J]', 0.5 * (rig['M'] + rig['m'].sum().item() - rig['M']) * 4.8e3**2 * 1e-3, flush=True)
print('shell area', rig['shell'].area.sum().item(), 'min area', rig['shell'].area.min().item(), flush=True)
t0 = time.time()
h = run_helmet(rig, T=T, record_every=20)
print('steps', len(h['dt']), 'dt min/max', min(h['dt']), max(h['dt']), 'wall s', time.time() - t0, flush=True)
for k in range(0, len(h['t']), max(1, len(h['t']) // 15)):
    print(f"t={h['t'][k]*1e3:.3f} ms  a_head={h['a_head_g'][k]:8.1f} g  F_anv={h['F_anv'][k]:9.2f} N  KE={h['KE'][k]:7.3f}  "
          f"u_head={h['u_head'][k]:7.3f}  u_crown={h['u_crown'][k]:7.3f}  g_min={h['g_min'][k]:.4f}", flush=True)
