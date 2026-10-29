"""Sizing study S0 (notes/0_1 register): uniform strut radius sweep on the one-cell-thick 3x3 helmet patch, 4.8 m/s, 21 J.
Reports peak head decel, head stroke at peak, energy absorbed by the liner at the end, bottoming. One run per radius, sequential."""
import sys, time, json
import numpy as np, torch
import runStatus as rs
from latticeSetup import example3
from material import project_material
from helmetRig import helmet_rig, run_helmet
import jointRule
torch.set_default_dtype(torch.float64)
NZ = int(sys.argv[1]); radii = [float(a) for a in sys.argv[2:]] or [0.8, 1.1]
T = 3.5e-3 if NZ == 1 else 6e-3
rs.start('helmet sizing S0')
lat = example3(elemSize=3.0, nz=NZ)
out = {}
for i, r0 in enumerate(radii):
    rs.update(f'r={r0}', i, len(radii))
    rig = helmet_rig(lat, torch.full((len(lat['connr']),), r0), project_material(), joint=dict(jointRule.PRESETS['R5_C4']))
    t0 = time.time()
    h = run_helmet(rig, T=T, record_every=20)
    a, u = h['a_head_g'], h['u_head']
    k = int(np.argmax(a))
    sc = float(np.array(h['u_head']).max() - np.array(h['u_crown'])[np.argmax(u)]) if len(u) else 0
    out[r0] = dict(peak_g=float(a.max()), t_peak_ms=float(h['t'][k] * 1e3), stroke_at_peak=float(u[k] - h['u_crown'][k]),
                   v_end=float(h['v_head'][-1]), KE_end_J=float(h['KE'][-1] * 1e-3), steps=len(h['dt']), wall=time.time() - t0,
                   F_anv_max=float(h['F_anv'].max()), a_mean_g=float(a[a > 0.1 * a.max()].mean()))
    print(r0, out[r0], flush=True)
json.dump(out, open('abaqus/results/helmet_sizing_S0_nz%d.json' % NZ + '', 'w'), indent=1)
