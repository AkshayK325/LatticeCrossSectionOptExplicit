"""Design gradient with the closed-form joint rule (src/jointRule.py) in the loop: the checkpointed adjoint equals unrolled
reverse AD and central finite differences, i.e. the radii enter the stiffness factors through the differentiable rule."""
import torch

from adjoint import DropStepper, checkpointed_adjoint
from dropGrad import nominal, evaluate, radii, placeholder_J
from latticeSetup import example3, drop_rig, n_design_rows
from material import project_material

torch.set_default_dtype(torch.float64)
MAT = project_material()
import pytest

import jointRule

JOINTS = {'first': dict(mode='rule', a=1.2, p=4.0, zl=1.1, b=0.3, mu=0.8),
          # R5 of abaqus/results/rule_fit3/R5.json: rigid zone, negative angle exponent, neighbour-sum saturation
          'R5': dict(mode='rule', a=3.45, p=-0.77, pz=-0.77, zl=0.77, b=0.15, gamma=0.0, wsat=2.47),
          # the shipped smooth presets (kink-free, short-strut law for C4): the configuration used for design
          'R5_C4_preset': dict(jointRule.PRESETS['R5_C4'])}
RIG0 = dict(law={'kind': 'crit_damped', 'F_ref': 2000.0, 'delta_ref': 0.2}, start='co_moving', gap0=0.1, energy_J=0.75)
SYM = 'Midplane'


def _setup(RIG):
    lat = example3(elemSize=3.0)
    x0 = torch.full((n_design_rows(SYM, lat), 1), 0.8)
    x0[::2] = 0.85
    rig0, sched = nominal(lat, x0, SYM, MAT, T=4e-4, **RIG)
    return lat, x0, rig0, sched


@pytest.mark.parametrize('which', ['first', 'R5', 'R5_C4_preset'])
def test_rule_gradient_adjoint_ad_fd(which):
    RIG = dict(RIG0, joint=JOINTS[which])
    lat, x0, rig0, sched = _setup(RIG)
    kw = {k: v for k, v in RIG.items() if k != 'law'}
    x = x0.clone().requires_grad_(True)
    g_ad, = torch.autograd.grad(evaluate(lat, x, SYM, MAT, rig0, sched, **RIG), x)
    x2 = x0.clone().requires_grad_(True)
    rig = drop_rig(lat, radii(x2, SYM, lat), rig0['law'], mat=MAT, freeze=rig0, **kw)
    J, (g_adj,) = checkpointed_adjoint(DropStepper(rig, sched, placeholder_J), [x2], K=7)
    assert torch.allclose(g_adj, g_ad, rtol=1e-9, atol=1e-12 * g_ad.abs().max().item()), (g_adj - g_ad).abs().max()
    # central finite differences along one design row
    d = torch.zeros_like(x0); d[1] = 1.0; h = 1e-5
    with torch.no_grad():
        fp = evaluate(lat, x0 + h * d, SYM, MAT, rig0, sched, **RIG).item()
        fm = evaluate(lat, x0 - h * d, SYM, MAT, rig0, sched, **RIG).item()
    fd = (fp - fm) / (2 * h)
    assert abs((g_adj * d).sum().item() - fd) < 1e-6 * max(abs(fd), 1e-12) + 1e-9
    # the joint rule changes the response (the stiffening is really in the loop)
    rig_nj = dict(RIG); rig_nj.pop('joint')
    with torch.no_grad():
        j0 = evaluate(lat, x0, SYM, MAT, rig0, sched, **rig_nj).item()
        j1 = evaluate(lat, x0, SYM, MAT, rig0, sched, **RIG).item()
    assert abs(j1 - j0) > 1e-6 * abs(j0)
