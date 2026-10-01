"""Mesh convergence of the drop rig (EPU 45, r = 0.8, 0.75 J, 6 m/s, co-moving + crit-damped, lattice contact).

Not a unit test (minutes per case). Writes a numbered figure + JSON to ResultsFig/explicit/.
    PYTHONPATH=src:tests python tests/study_mesh_convergence.py [rotscale]
"""
import json
import sys
import time
import pathlib

import numpy as np
import torch

from latticeSetup import example3, drop_rig, run_drop, jump_metrics
from material import project_material
from plotFrame import numbered

torch.set_default_dtype(torch.float64)
OUT = pathlib.Path(__file__).resolve().parents[1] / 'ResultsFig' / 'explicit'
MAT = project_material()
SPEC = {'kind': 'crit_damped', 'F_ref': 2000.0, 'delta_ref': 0.2}

cases = [(3.0, True), (1.5, True), (1.0, True), (0.75, True)]
if 'rotscale' in sys.argv:
    cases += [(0.75, False)]
res, hist = [], {}
for es, rot in cases:
    lat = example3(elemSize=es)
    rig = drop_rig(lat, torch.full((lat['connr'].shape[0],), 0.8), SPEC, MAT, start='co_moving', gap0=0.1,
                   energy_J=0.75)
    if not rot:                                   # physical rotary inertia (no scaling) for this case
        from explicitFE import lumped_mass, adaptive_dt
        _, j = lumped_mass(rig['fr'], rig['A'], rig['I'], MAT['rho'], rot_scale=False)
        j[rig['imp']] = 1.0
        rig['j'][:] = j
    t0 = time.time()
    h = run_drop(rig, T=8e-3)
    E = h['E_tot']
    jm = jump_metrics(h['F_imp'])
    r = {'elemSize': es, 'rot_scale': rot, 'n_el': int(lat['connr'].shape[0]), 'steps': len(h['dt']),
         'dt_min': float(h['dt'].min()), 'stroke_max': float(-h['z_plate'].min()), 'F_imp_peak': float(jm['peak']),
         'F_bot_peak': float(np.abs(h['F_bottom']).max()), 'E_diss': float(h['E_diss_material'][-1]),
         'v_imp_end': float(h['v_imp'][-1] / 1e3), 'dE': float(np.abs(E - E[0]).max() / E[0]),
         'g_self_min': float(np.min(h['g_self_min'])), 'g_plate_min': float(np.min(h['g_plate_min'])),
         'n_initial_overlap': rig['self_contact'].n_initial_overlap, 'runtime_s': time.time() - t0}
    res.append(r)
    hist[(es, rot)] = h
    print(json.dumps(r), flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for (es, rot), h in hist.items():
    lab = f'elemSize {es} mm' + ('' if rot else ' (no rot. scaling)')
    ax[0].plot(h['t'] * 1e3, h['F_imp'], label=lab)
    ax[1].plot(-h['z_plate'], h['F_imp'], label=lab)
    ax[2].plot(h['t'] * 1e3, h['E_diss_material'] / 1e3, label=lab)
ax[0].set(xlabel='t [ms]', ylabel='impactor force [N]', title='force history')
ax[1].set(xlabel='top-plate stroke [mm]', ylabel='impactor force [N]', title='force–stroke')
ax[2].set(xlabel='t [ms]', ylabel='material dissipation [J]', title='dissipation')
for a in ax:
    a.legend(fontsize=7)
fig.suptitle('Mesh convergence: EPU 45, r = 0.8 mm, 0.75 J, 6 m/s')
fig.tight_layout()
fig.savefig(numbered(OUT, 'mesh_convergence.png'), dpi=130)
json.dump(res, open(numbered(OUT, 'mesh_convergence.json'), 'w'), indent=1)
