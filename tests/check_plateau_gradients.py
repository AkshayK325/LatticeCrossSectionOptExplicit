"""Gradient verification on the ACTUAL plateau-design problem (study_plateau_design.py: example3 lattice, Midplane, ViscoHyper
EPU 45, full drop T, F-delta MSE objective) with a chosen joint model.
Compares for the frozen-schedule objective J(x): checkpointed adjoint  vs  unrolled reverse AD  vs  central finite differences
(per-group directions and random directions; all nrow design rows are covered by the AD/adjoint comparison).
    JOINT=R5_C4 PYTHONPATH=src:tests python -u tests/check_plateau_gradients.py [cuda|cpu] [T_full] [T_unrolled]
adjoint vs FD at the full drop T_full; adjoint vs unrolled AD vs FD on a truncated drop T_unrolled (unrolled AD stores the whole graph).
Writes abaqus/results/plateau_gradient_check_<JOINT>.md"""
import os
import pathlib
import sys

import numpy as np
import torch

import runStatus as rs

dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
T = float(sys.argv[2]) if len(sys.argv) > 2 else 3.5e-3
TU = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0e-3
torch.set_default_device(dev)
torch.set_default_dtype(torch.float64)
import jointRule
from adjoint import DropStepper, checkpointed_adjoint
from dropGrad import nominal, radii
import resource
resource.setrlimit(resource.RLIMIT_AS, (24 << 30, resource.RLIM_INFINITY)) if dev == 'cpu' else None
from latticeSetup import example3, symmetry_groups, n_design_rows, drop_rig
from material import project_material
from objectives import make_terms, fd_mse_outer, plateau_target

JOINT = os.environ.get('JOINT', 'R5_C4')
SYM, RR = 'Midplane', (0.4, 1.2)
lat = example3(elemSize=3.0)
groups = symmetry_groups(SYM, lat)
nrow = n_design_rows(SYM, lat)
mat = project_material()
target = plateau_target(float(os.environ.get('F_PLATEAU', 80.0)), 0.3, 12.0)
rig_kw = dict(law={'kind': 'crit_damped', 'F_ref': 2000.0, 'delta_ref': 0.2}, start='co_moving', gap0=0.1, energy_J=0.75)
if os.environ.get('JOINT_K'):
    rig_kw['joint'] = {'k': float(os.environ['JOINT_K'])}
elif JOINT != 'none':
    rig_kw['joint'] = dict(jointRule.PRESETS[JOINT])
if os.environ.get('COURANT'):
    rig_kw['courant'] = float(os.environ['COURANT'])
den = lambda x: RR[0] + (RR[1] - RR[0]) * x


def rows(gv):
    x = torch.zeros(nrow, 1)
    for g, v in zip(groups, gv):
        x[torch.as_tensor(g)] = float(v)
    return x



def build(x, rig0, sched):
    kw = {k: v for k, v in rig_kw.items() if k != 'law'}
    rig = drop_rig(lat, radii(den(x), SYM, lat), rig0['law'], mat=mat, freeze=rig0, **kw)  # courant only matters for the nominal schedule
    return rig, DropStepper(rig, sched, make_terms(rig, target))


def adjoint(x0, rig0, sched):
    x = x0.clone().requires_grad_(True)
    with torch.enable_grad():
        rig, st = build(x, rig0, sched)
    J, (g,) = checkpointed_adjoint(st, [x], outer=fd_mse_outer(target))
    return J, g


def unrolled(x0, rig0, sched):
    x = x0.clone().requires_grad_(True)
    with torch.enable_grad():
        rig, st = build(x, rig0, sched)
        z, S = st.initial_state(), 0.0
        for n in range(len(sched)):
            z, L = st.step(z, n)
            S = S + L
        J = fd_mse_outer(target)(S, None)
        g, = torch.autograd.grad(J, x)
    return J.item(), g


def fdJ(x, rig0, sched):
    with torch.no_grad():
        rig, st = build(x, rig0, sched)
        z, S = st.initial_state(), 0.0
        for n in range(len(sched)):
            z, L = st.step(z, n)
            S = S + L
        return fd_mse_outer(target)(S, None).item()


HS = (1e-5, 1e-4, 1e-3)
rs.start(f'plateau gradient check, joint {JOINT}')
rng = np.random.default_rng(0)
out = [f'# Plateau-problem gradient check — joint `{JOINT}`', '',
       'Actual plateau-design problem: example3(elemSize=3) lattice, Midplane, EPU 45 ViscoHyper, F-delta MSE objective '
       '(80 N plateau), design = 3 group radii mapped to all rows; frozen-schedule objective (nominal adaptive dt replayed). '
       'adjoint = checkpointed discrete adjoint; AD = unrolled reverse autograd; FD = central differences, h = 1e-5 '
       '(normalised x). dir = unit directions of the 3 groups and 1 random direction over all design rows.', '',
       '| T [ms] | start (3 groups) | J | adjoint vs AD (max rel) | dir | adjoint·d | ' + ' | '.join(f'FD h={h:g} (rel err)' for h in HS) + ' |',
       '|---|---|---|---|---|---|' + '---|' * len(HS)]
cases = [(T, [0.5, 0.5, 0.5], False), (T, [0.15, 0.55, 0.9], False), (TU, [0.5, 0.5, 0.5], True), (TU, [0.15, 0.55, 0.9], True)]
if os.environ.get('STARTS'):          # diagnostic: STARTS="0.15,0.55,0.9" ONLY_GROUPS=1 -> no unrolled, group directions only
    cases = [(T, [float(v) for v in os.environ['STARTS'].split(',')], False)]
for ci, (Tc, gv, unr) in enumerate(cases):
    x0 = rows(gv)
    with torch.no_grad():
        rig0, sched = nominal(lat, den(x0), SYM, mat, T=Tc, **rig_kw)
    rs.update(f'case {ci}: T={Tc:g} ({len(sched)} steps)', ci, len(cases))
    J, g = adjoint(x0, rig0, sched)
    if unr:
        _, ga = unrolled(x0, rig0, sched)
        cmp_ = f'{((g - ga).abs().max() / ga.abs().max()).item():.1e}'
    else:
        cmp_ = 'n/a (full drop)'
    dirs = [(f'group {k}', torch.zeros(nrow, 1).index_put_((torch.as_tensor(groups[k]),), torch.ones(len(groups[k]), 1))) for k in range(3)]
    for k in range(0 if os.environ.get('ONLY_GROUPS') else 1):
        d = torch.as_tensor(rng.standard_normal((nrow, 1)))
        dirs.append((f'random {k}', d / d.norm()))
    for name, d in dirs:
        ad = (g * d).sum().item()
        fds = []
        for h in HS:
            fds.append((fdJ(x0 + h * d, rig0, sched) - fdJ(x0 - h * d, rig0, sched)) / (2 * h))
        errs = [abs(ad - f) / max(abs(f), 1e-300) for f in fds]
        out.append(f'| {Tc * 1e3:g} | {gv} | {J:.4e} | {cmp_} | {name} | {ad:.6e} | ' + ' | '.join(f'{f:.6e} ({e:.1e})' for f, e in zip(fds, errs)) + ' |')
        print(out[-1], flush=True)
TAGP = os.environ.get('TAGP', '')
path = pathlib.Path(__file__).resolve().parents[1] / 'abaqus' / 'results' / f'plateau_gradient_check_{JOINT}{TAGP}.md'
path.write_text('\n'.join(out) + '\n')
print('wrote', path)
