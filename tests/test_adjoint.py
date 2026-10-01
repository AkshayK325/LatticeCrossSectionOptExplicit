"""Checkpointed discrete adjoint (src/adjoint.py) == unrolled reverse AD == FD, without unrolling."""
import pytest
import torch

from adjoint import DropStepper, forward_J, checkpointed_adjoint
from dropGrad import nominal, evaluate, radii, placeholder_J
from latticeSetup import example3, drop_rig, n_design_rows
from material import project_material

torch.set_default_dtype(torch.float64)
MAT = project_material()
RIG = dict(law={'kind': 'crit_damped', 'F_ref': 2000.0, 'delta_ref': 0.2}, start='co_moving', gap0=0.1,
           energy_J=0.75)
SYM = 'Midplane'


@pytest.fixture(scope='module')
def case():
    lat = example3(elemSize=3.0)
    x0 = torch.full((n_design_rows(SYM, lat), 1), 0.8)
    x0[::2] = 0.85                                                  # non-uniform design
    rig0, sched = nominal(lat, x0, SYM, MAT, T=4e-4, **RIG)
    return lat, x0, rig0, sched


def adjoint_rig(lat, x, rig0):
    kw = {k: v for k, v in RIG.items() if k != 'law'}
    return drop_rig(lat, radii(x, SYM, lat), rig0['law'], mat=MAT, freeze=rig0, **kw)


def test_stepper_reproduces_simulate(case):
    lat, x0, rig0, sched = case
    with torch.no_grad():
        J_sim = evaluate(lat, x0, SYM, MAT, rig0, sched, **RIG).item()
    J_step = forward_J(DropStepper(adjoint_rig(lat, x0, rig0), sched, placeholder_J))
    assert J_step == pytest.approx(J_sim, rel=1e-12)


@pytest.mark.parametrize('K', [1, 7, None, 10**6])
def test_adjoint_equals_unrolled_ad(case, K):
    lat, x0, rig0, sched = case
    x = x0.clone().requires_grad_(True)
    g_ad, = torch.autograd.grad(evaluate(lat, x, SYM, MAT, rig0, sched, **RIG), x)
    x2 = x0.clone().requires_grad_(True)
    J, (g_adj,) = checkpointed_adjoint(DropStepper(adjoint_rig(lat, x2, rig0), sched, placeholder_J), [x2], K=K)
    assert torch.allclose(g_adj, g_ad, rtol=1e-9, atol=1e-12 * g_ad.abs().max().item()), (g_adj - g_ad).abs().max()
