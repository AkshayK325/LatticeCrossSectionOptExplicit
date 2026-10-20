"""Plateau-design problem (F-delta MSE objective, 3 Midplane groups, R5_C4 preset as used by study_plateau_design.py):
checkpointed adjoint == unrolled reverse AD == central finite differences, for unequal group radii.
Short drop (T = 4e-4): the full-drop check is tests/check_plateau_gradients.py (see abaqus/results/plateau_gradient_check_R5_C4.md)."""
import numpy as np
import torch

import jointRule
from adjoint import DropStepper, checkpointed_adjoint
from dropGrad import nominal, radii
from latticeSetup import example3, drop_rig, n_design_rows, symmetry_groups
from material import project_material
from objectives import make_terms, fd_mse_outer, plateau_target

torch.set_default_dtype(torch.float64)
SYM, RR, T = 'Midplane', (0.4, 1.2), 4e-4
MAT = project_material()
RIG = dict(law={'kind': 'crit_damped', 'F_ref': 2000.0, 'delta_ref': 0.2}, start='co_moving', gap0=0.1, energy_J=0.75,
           joint=dict(jointRule.PRESETS['R5_C4']))


def _J(x, rig0, sched, target, lat, grad):
    kw = {k: v for k, v in RIG.items() if k != 'law'}
    rig = drop_rig(lat, radii(RR[0] + (RR[1] - RR[0]) * x, SYM, lat), rig0['law'], mat=MAT, freeze=rig0, **kw)
    st = DropStepper(rig, sched, make_terms(rig, target))
    if grad:
        return checkpointed_adjoint(st, [x], outer=fd_mse_outer(target))
    with torch.no_grad():
        z, S = st.initial_state(), 0.0
        for n in range(len(sched)):
            z, L = st.step(z, n)
            S = S + L
    return fd_mse_outer(target)(S, None).item()


def test_plateau_gradient_r5c4_adjoint_ad_fd():
    lat = example3(elemSize=3.0)
    groups = symmetry_groups(SYM, lat)
    x0 = torch.zeros(n_design_rows(SYM, lat), 1)
    for g, v in zip(groups, (0.15, 0.55, 0.9)):
        x0[torch.as_tensor(g)] = v
    target = plateau_target(80.0, 0.3, 12.0)
    with torch.no_grad():
        rig0, sched = nominal(lat, RR[0] + (RR[1] - RR[0]) * x0, SYM, MAT, T=T, **RIG)
    x = x0.clone().requires_grad_(True)
    J, (g_adj,) = _J(x, rig0, sched, target, lat, True)
    # unrolled reverse AD through the same step map
    xa = x0.clone().requires_grad_(True)
    kw = {k: v for k, v in RIG.items() if k != 'law'}
    rig = drop_rig(lat, radii(RR[0] + (RR[1] - RR[0]) * xa, SYM, lat), rig0['law'], mat=MAT, freeze=rig0, **kw)
    st = DropStepper(rig, sched, make_terms(rig, target))
    z, S = st.initial_state(), 0.0
    for n in range(len(sched)):
        z, L = st.step(z, n)
        S = S + L
    g_ad, = torch.autograd.grad(fd_mse_outer(target)(S, None), xa)
    assert torch.allclose(g_adj, g_ad, rtol=1e-9, atol=1e-12 * g_ad.abs().max().item()), (g_adj - g_ad).abs().max()
    rng = np.random.default_rng(1)
    dirs = [torch.zeros_like(x0).index_put_((torch.as_tensor(gr),), torch.ones(len(gr), 1)) for gr in groups]
    d = torch.as_tensor(rng.standard_normal(x0.shape)); dirs.append(d / d.norm())
    for d in dirs:
        h = 1e-5
        fd = (_J(x0 + h * d, rig0, sched, target, lat, False) - _J(x0 - h * d, rig0, sched, target, lat, False)) / (2 * h)
        assert abs((g_adj * d).sum().item() - fd) < 1e-5 * abs(fd) + 1e-9, ((g_adj * d).sum().item(), fd)
