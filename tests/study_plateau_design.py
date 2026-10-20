"""Design for an ideal-absorber force-stroke curve: fast rise 0 -> F_PLATEAU over D_RISE, then constant F_PLATEAU.
Minimises the normalised F-delta MSE (loading branch) with exact adjoint gradients; 3 Midplane group radii.
    [F_PLATEAU=120] PYTHONPATH=src:tests python tests/study_plateau_design.py [cuda|cpu] [mma|gcmma|lbfgsb] [r0,r1,r2 start] [maxiter]
Writes (after the run finishes) a numbered figure + json in ResultsFig/explicit/ and notes/9_PlateauDesign_<opt>.md."""
import datetime
import json
import pathlib
import sys
import time

import numpy as np
import torch

dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
OPT = sys.argv[2] if len(sys.argv) > 2 else 'mma'
torch.set_default_device(dev)
torch.set_default_dtype(torch.float64)
from latticeSetup import example3, symmetry_groups, n_design_rows
from material import project_material
from objectives import make_terms, fd_mse_outer, plateau_target
from optimizeDrop import value_and_grad, response_curve
from utilFuncs import custom_minimize
from scipy.optimize import Bounds

ROOT = pathlib.Path(__file__).resolve().parents[1]
SYM, T = 'Midplane', 3.5e-3
import os
F_PLATEAU = float(os.environ.get('F_PLATEAU', 80.0))  # [N] plateau force (env F_PLATEAU); 0.75 J at 80 N needs 9.4 mm
D_RISE, D_END = 0.3, 12.0                           # [mm] rise length, target end
R_RANGE = (0.4, 1.2)
to_x = lambda r: [(v - R_RANGE[0]) / (R_RANGE[1] - R_RANGE[0]) for v in r]
to_r = lambda x: [R_RANGE[0] + (R_RANGE[1] - R_RANGE[0]) * v for v in x]
lat = example3(elemSize=3.0)
groups = symmetry_groups(SYM, lat)
nrow = n_design_rows(SYM, lat)


def rows(gv):
    x = torch.zeros(nrow, 1)
    for g, val in zip(groups, gv):
        x[torch.as_tensor(g)] = float(val)
    return x


base = dict(lat=lat, symType=SYM, mat=project_material(), T=T,
            rig_kw=dict(law={'kind': 'crit_damped', 'F_ref': 2000.0, 'delta_ref': 0.2}, start='co_moving', gap0=0.1,
                        energy_J=0.75), x_range=R_RANGE)
# Joint stiffening: default = validated closed-form rule jointRule.PRESETS['R5_C4'] (env JOINT=R5|R5_C4|R5_all22|none);
# JOINT_K=3.5 selects the old constant-k zone model (kept for comparison with earlier notes/9 runs).
JOINT_K = float(os.environ.get('JOINT_K', 0)) or None
JOINT = 'k' if JOINT_K else os.environ.get('JOINT', 'R5_C4')
if JOINT_K:
    base['rig_kw']['joint'] = {'k': JOINT_K}
elif JOINT != 'none':
    import jointRule
    base['rig_kw']['joint'] = dict(jointRule.PRESETS[JOINT])
target = plateau_target(F_PLATEAU, D_RISE, D_END)
spec = dict(base, terms=lambda rig: make_terms(rig, target), outer=lambda rig: fd_mse_outer(target))
g0 = [float(v) for v in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0.80, 0.80, 0.80]
MAXIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
TAG = (f'{F_PLATEAU:.0f}N_' if F_PLATEAU != 80.0 else '') + (f'joint{JOINT_K:g}_' if JOINT_K else (f'{JOINT}_' if JOINT not in ('R5_C4',) and JOINT != 'none' else ('bare_' if JOINT == 'none' else ''))) + OPT + ('' if len(sys.argv) <= 3 else '_from_' + '_'.join(f'{v:.3f}' for v in g0))

hist = {'J': [], 'g': [], 'time': []}
memo, J0 = {}, {}


def obj(xv):
    """Normalised x in [0, 1] (denormalised inside value_and_grad); J scaled by J(x0)."""
    xv = np.asarray(xv, dtype=float).reshape(-1)
    key = tuple(np.round(xv, 15))
    if key in memo:
        return memo[key]
    t0 = time.time()
    J, gx = value_and_grad(rows(xv), spec)
    gg = np.array([gx[torch.as_tensor(g)].sum().item() for g in groups])
    J0.setdefault('v', J)
    hist['J'].append(J)
    hist['g'].append(to_r(xv))
    hist['time'].append(time.time() - t0)
    print(f'iter {len(hist["J"])}: J = {J:.4e} (J/J0 {J / J0["v"]:.3e}), radii = {np.round(to_r(xv), 4)}, '
          f'dJ/dx = {gg}, {hist["time"][-1]:.0f}s', flush=True)
    memo[key] = (np.array([J / J0['v']]), (gg / J0['v']).reshape(-1, 1))
    return memo[key]


def no_constraint(x):
    return -np.ones((1, 1)), np.zeros((1, np.asarray(x).size))


t_start = time.time()
x0 = np.array(to_x(g0))
if OPT == 'mma':
    res = custom_minimize(obj, x0=x0, bounds=Bounds([0.0] * 3, [1.0] * 3), method='MMA', constraintCall=no_constraint,
                          options={'maxiter': MAXIT or 30, 'miniter': 5, 'kkttol': 1e-6, 'move_limit': 0.5, 'disp': True,
                                   'mma_variant': 'standard', 'move_schedule': 'cos', 'move_period': 10})
    x_opt = res.x.reshape(-1)
elif OPT == 'gcmma':
    from optimizeDrop import value_only
    vcall = lambda xv: (np.array([value_only(rows(np.asarray(xv).reshape(-1)), spec) / J0['v']]), -np.ones((1, 1)))
    res = custom_minimize(obj, x0=x0, bounds=Bounds([0.0] * 3, [1.0] * 3), method='GCMMA', constraintCall=no_constraint,
                          options={'maxiter': MAXIT or 15, 'miniter': 3, 'kkttol': 1e-6, 'disp': True, 'value_call': vcall,
                                   'move_limit': 0.5, 'move_schedule': 'cos'})
    x_opt = res.x.reshape(-1)
else:
    from scipy.optimize import minimize
    res = minimize(lambda xv: (lambda f, g: (f[0], g.reshape(-1)))(*obj(xv)), x0=x0, jac=True, method='L-BFGS-B',
                   bounds=[(0.0, 1.0)] * 3, options={'maxiter': 25, 'ftol': 1e-10, 'gtol': 1e-8})
    x_opt = res.x
# best evaluated design (MMA iterates are not monotone)
g_opt = np.array(hist['g'][int(np.argmin(hist['J']))])
wall = time.time() - t_start

# ---------------------------------------------------------------- report (after the run)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plotFrame import numbered


def metrics(d, F):
    """Loading-branch energy [J], peak force [N], max stroke [mm], crush efficiency E / (F_max d_max),
    plateau uniformity (mean / max force after the rise, up to max stroke)."""
    d, F = d.cpu().numpy(), F.cpu().numpy()
    k = int(np.argmax(d))
    E = np.trapezoid(F[:k + 1], d[:k + 1]) * 1e-3
    pl = F[:k + 1][d[:k + 1] > D_RISE]
    return dict(E_J=E, F_max=F.max(), d_max=d[k], eff=E * 1e3 / (F.max() * d[k]), uniform=pl.mean() / pl.max())


d0, F0, _ = response_curve(rows(to_x(g0)), base)
d1, F1, _ = response_curve(rows(to_x(g_opt)), base)
m0, m1 = metrics(d0, F0), metrics(d1, F1)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].plot(*[t.cpu() for t in target], 'k-', lw=2.5, label=f'target: {F_PLATEAU:.0f} N plateau')
ax[0].plot(d0.cpu(), F0.cpu(), 'C0--', label=f'initial {g0}')
ax[0].plot(d1.cpu(), F1.cpu(), 'C3-', label=f'optimized {np.round(g_opt, 3).tolist()}')
ax[0].set(xlabel='top-plate stroke [mm]', ylabel='impactor force [N]', title='force–stroke (full drop)',
          xlim=(0, max(d0.max().item(), d1.max().item()) * 1.05))
ax[0].legend(fontsize=8)
ax[1].semilogy(range(1, len(hist['J']) + 1), np.array(hist['J']) / hist['J'][0], 'o-')
ax[1].set(xlabel='function evaluation', ylabel='J / J0', title='objective (F–δ MSE)')
G = np.array(hist['g'])
for k in range(3):
    ax[2].plot(range(1, len(G) + 1), G[:, k], 'o-', label=f'group {k}')
ax[2].set(xlabel='function evaluation', ylabel='radius [mm]', title='design')
ax[2].legend(fontsize=8)
fig.tight_layout()
fp = numbered(ROOT / 'ResultsFig' / 'explicit', f'plateau_design_{TAG}.png')
fig.savefig(fp, dpi=130)
json.dump(dict(F_plateau=F_PLATEAU, d_rise=D_RISE, g0=g0, g_opt=g_opt.tolist(), hist=hist, wall_s=wall,
               metrics0={k: float(v) for k, v in m0.items()}, metrics1={k: float(v) for k, v in m1.items()}),
          open(numbered(ROOT / 'ResultsFig' / 'explicit', f'plateau_design_{TAG}.json'), 'w'), indent=1)
row = lambda name, m, g: (f'| {name} | {g[0]:.3f} / {g[1]:.3f} / {g[2]:.3f} | {m["F_max"]:.1f} | {m["d_max"]:.2f} | '
                          f'{m["E_J"]:.3f} | {m["eff"]:.2f} | {m["uniform"]:.2f} |')
md = [
    f'# 9 — Plateau design: {OPT} + exact adjoint gradients toward an ideal-absorber curve', '',
    f'Generated automatically by `tests/study_plateau_design.py` after the run finished ({datetime.datetime.now():%Y-%m-%d %H:%M}, device {dev}).', '',
    f'- Rig: EPU 45 drop, 0.75 J, 6 m/s, 3 mm mesh, {T*1e3:.1f} ms, fibre visco-hyperelastic beams, line + plate contact, μ = 0.5.',
    f'- Target: linear rise 0 → {F_PLATEAU:.0f} N over {D_RISE} mm, then {F_PLATEAU:.0f} N constant (to {D_END} mm). '
    f'0.75 J at {F_PLATEAU:.0f} N needs {750 / F_PLATEAU:.1f} mm of stroke.',
    f'- Objective: normalised MSE of the impactor force–stroke loading branch against the target. '
    f'Design: Midplane symmetry, 3 group radii, normalised x in [0, 1] (radius = {R_RANGE[0]} + {R_RANGE[1]-R_RANGE[0]:.1f} x).',
    f'- Optimizer: {OPT}; gradients: checkpointed discrete adjoint (exact). Start {g0}. Contact exclusion: smooth (C1) unless noted. '
    f'Joint stiffening: {"k = %g (geometric zones, latticeSetup.joint_factors)" % JOINT_K if JOINT_K else ("none (bare beams)" if JOINT == "none" else "closed-form rule jointRule.PRESETS[%r] (smooth, differentiable; notes/18)" % JOINT)}.', '',
    f'![plateau](../ResultsFig/explicit/{fp.name})', '',
    '| design | radii [mm] | peak F [N] | max stroke [mm] | loading energy [J] | crush efficiency | plateau mean/max |',
    '|---|---|---|---|---|---|---|',
    row('initial', m0, g0), row('optimized (best evaluated)', m1, g_opt), '',
    'Crush efficiency = loading energy / (peak force × max stroke); 1 for an ideal square-wave absorber.', '',
    f'Objective {hist["J"][0]:.3e} → {min(hist["J"]):.3e} (×{min(hist["J"]) / hist["J"][0]:.2e}) in {len(hist["J"])} '
    f'evaluations ({np.mean(hist["time"]):.0f} s each), total {wall/60:.1f} min.',
]
(ROOT / 'notes' / f'9_PlateauDesign_{TAG}.md').write_text('\n'.join(md) + '\n')
print('report:', fp, ROOT / 'notes' / f'9_PlateauDesign_{TAG}.md')
