"""End-to-end check: recover a known design from its F-delta curve with MMA + exact adjoint gradients.
    PYTHONPATH=src:tests python tests/study_twin_recovery.py [cuda|cpu]
Writes (after the run finishes) a numbered figure in ResultsFig/explicit/ and notes/8_TwinRecovery.md."""
import datetime
import json
import pathlib
import sys
import time

import numpy as np
import torch

dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
OPT = sys.argv[2] if len(sys.argv) > 2 else 'gcmma'      # 'gcmma' (default), 'mma' (standard MMA), 'lbfgsb'
torch.set_default_device(dev)
torch.set_default_dtype(torch.float64)
from latticeSetup import example3, symmetry_groups, n_design_rows
from material import project_material
from objectives import make_terms, fd_mse_outer
from optimizeDrop import value_and_grad, response_curve, loading_branch
from utilFuncs import custom_minimize
from scipy.optimize import Bounds

ROOT = pathlib.Path(__file__).resolve().parents[1]
SYM, T = 'Midplane', 3e-3
R_RANGE = (0.4, 1.2)                     # radius = min + (max - min) * x, x in [0, 1] (OptimizeFrame convention)
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
g_true = [0.70, 0.90, 0.80]                 # radii [mm]
g0 = [0.80, 0.80, 0.80]
d_t, F_t, _ = response_curve(rows(to_x(g_true)), base)
target = tuple(t.cpu() if False else t for t in loading_branch(d_t, F_t))
spec = dict(base, terms=lambda rig: make_terms(rig, target), outer=lambda rig: fd_mse_outer(target))

hist = {'J': [], 'g': [], 'time': []}
memo, J0 = {}, {}


def obj(xv):
    """MMA works on normalised x in [0,1]; denormalisation happens inside value_and_grad (x_range). J is scaled by
    its value at the start design (OptimizeFrame obj0 idea) so MMA's constants see an O(1) objective."""
    xv = np.asarray(xv, dtype=float).reshape(-1)                                # MMA passes (n,) or (n, 1)
    key = tuple(np.round(xv, 15))
    if key in memo:
        return memo[key]
    t0 = time.time()
    J, gx = value_and_grad(rows(xv), spec)
    gg = np.array([gx[torch.as_tensor(g)].sum().item() for g in groups])      # chain rule rows -> groups
    J0.setdefault('v', J)
    hist['J'].append(J)
    hist['g'].append(to_r(xv))
    hist['time'].append(time.time() - t0)
    print(f'iter {len(hist["J"])}: J = {J:.4e} (J/J0 {J / J0["v"]:.3e}), radii = {np.round(to_r(xv), 4)}, '
          f'dJ/dx = {gg}, {hist["time"][-1]:.0f}s', flush=True)
    memo[key] = (np.array([J / J0['v']]), (gg / J0['v']).reshape(-1, 1))
    return memo[key]


def no_constraint(x):
    """Inactive constraint in the shapes MMA expects: value (1,1), gradient (1,n)."""
    return -np.ones((1, 1)), np.zeros((1, np.asarray(x).size))


t_start = time.time()
if OPT == 'gcmma':
    from optimizeDrop import value_only

    def vcall(xv):                                   # value-only evaluations for GCMMA's inner loop
        xv = np.asarray(xv, dtype=float).reshape(-1)
        J = value_only(rows(xv), spec)
        print(f'   inner value: radii {np.round(to_r(xv), 4)}, J/J0 {J / J0["v"]:.3e}', flush=True)
        return np.array([J / J0['v']]), -np.ones((1, 1))
    res = custom_minimize(obj, x0=np.array(to_x(g0)).reshape(-1, 1), bounds=Bounds([0.0] * 3, [1.0] * 3),
                          method='GCMMA', constraintCall=no_constraint,
                          options={'maxiter': 20, 'miniter': 3, 'kkttol': 1e-6, 'disp': True, 'value_call': vcall})
    x_opt = res.x.reshape(-1)
elif OPT == 'mma':
    res = custom_minimize(obj, x0=np.array(to_x(g0)).reshape(-1, 1), bounds=Bounds([0.0] * 3, [1.0] * 3), method='MMA',
                          constraintCall=no_constraint,
                          options={'maxiter': 20, 'miniter': 5, 'kkttol': 1e-6, 'move_limit': 0.2, 'disp': True,
                                   'mma_variant': 'standard'})
    x_opt = res.x.reshape(-1)
else:
    from scipy.optimize import minimize
    res = minimize(lambda xv: (lambda f, g: (f[0], g.reshape(-1)))(*obj(xv)), x0=np.array(to_x(g0)), jac=True,
                   method='L-BFGS-B', bounds=[(0.0, 1.0)] * 3, options={'maxiter': 25, 'ftol': 1e-10, 'gtol': 1e-8})
    x_opt = res.x
g_opt = np.array(to_r(x_opt))
wall = time.time() - t_start

# ---------------------------------------------------------------- report (after the run)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plotFrame import numbered

d0, F0, _ = response_curve(rows(to_x(g0)), base)
d1, F1, _ = response_curve(rows(to_x(g_opt)), base)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].plot(d_t.cpu(), F_t.cpu(), 'k-', lw=2.5, label=f'target (groups {g_true})')
ax[0].plot(d0.cpu(), F0.cpu(), 'C0--', label=f'initial {g0}')
ax[0].plot(d1.cpu(), F1.cpu(), 'C3-', label=f'optimized {np.round(g_opt, 3).tolist()}')
ax[0].set(xlabel='top-plate stroke [mm]', ylabel='impactor force [N]', title='force–stroke (full drop, loading + unloading)')
ax[0].legend(fontsize=8)
ax[1].semilogy(range(1, len(hist['J']) + 1), hist['J'], 'o-')
ax[1].set(xlabel='function evaluation', ylabel='F–δ MSE (normalised)', title='objective')
G = np.array(hist['g'])
for k in range(3):
    ax[2].plot(range(1, len(G) + 1), G[:, k], 'o-', label=f'group {k}')
    ax[2].axhline(g_true[k], color=f'C{k}', ls=':', lw=1)
ax[2].set(xlabel='function evaluation', ylabel='radius [mm]', title='design (dotted: true)')
ax[2].legend(fontsize=8)
fig.tight_layout()
fp = numbered(ROOT / 'ResultsFig' / 'explicit', f'twin_recovery_fd_mse_{OPT}.png')
fig.savefig(fp, dpi=130)
json.dump(dict(g_true=g_true, g0=g0, g_opt=g_opt.tolist(), hist=hist, wall_s=wall),
          open(numbered(ROOT / 'ResultsFig' / 'explicit', f'twin_recovery_fd_mse_{OPT}.json'), 'w'), indent=1)
err = np.abs(np.array(g_opt) - np.array(g_true))
md = [
    '# 8 — Twin recovery: MMA + exact adjoint gradients recover a design from its F–δ curve', '',
    f'Generated automatically by `tests/study_twin_recovery.py` after the run finished ({datetime.datetime.now():%Y-%m-%d %H:%M}, device {dev}).', '',
    f'- Rig: EPU 45 drop, 0.75 J, 6 m/s, 3 mm mesh, {T*1e3:.0f} ms, fibre visco-hyperelastic beams, line + plate contact, μ = 0.5.',
    f'- Design: Midplane symmetry, 3 group radii; MMA on normalised x in [0, 1] (radius = {R_RANGE[0]} + {R_RANGE[1]-R_RANGE[0]:.1f} x, denormalised inside the objective), J scaled by J(x0). Target curve from the design {g_true}; start {g0}.',
    '- Objective: normalised MSE between the impactor force–stroke loading branch and the target.',
    f'- Gradients: checkpointed discrete adjoint (exact). Optimizer: {OPT} on normalised x in [0, 1].', '',
    f'![twin](../ResultsFig/explicit/{fp.name})', '',
    '| | group 0 | group 1 | group 2 |', '|---|---|---|---|',
    f'| true | {g_true[0]} | {g_true[1]} | {g_true[2]} |',
    f'| start | {g0[0]} | {g0[1]} | {g0[2]} |',
    f'| optimized | {g_opt[0]:.4f} | {g_opt[1]:.4f} | {g_opt[2]:.4f} |',
    f'| abs. error | {err[0]:.1e} | {err[1]:.1e} | {err[2]:.1e} |', '',
    f'Objective {hist["J"][0]:.3e} → {min(hist["J"]):.3e} in {len(hist["J"])} evaluations '
    f'({np.mean(hist["time"]):.0f} s each: nominal run + adjoint), total {wall/60:.1f} min.',
]
(ROOT / 'notes' / '8_TwinRecovery.md').write_text('\n'.join(md) + '\n')
print('report:', fp, ROOT / 'notes' / '8_TwinRecovery.md')
