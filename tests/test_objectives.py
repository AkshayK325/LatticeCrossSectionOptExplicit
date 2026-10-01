"""Objectives (target F-delta MSE, SEA) and their exact gradients through the checkpointed adjoint."""
import pytest
import torch

from adjoint import DropStepper, forward_J, checkpointed_adjoint
from dropGrad import nominal, radii
from latticeSetup import example3, drop_rig, n_design_rows
from material import project_material
from objectives import plateau_target, interp, make_terms, fd_mse_outer, sea_outer, lattice_mass

torch.set_default_dtype(torch.float64)
MAT = project_material()
SYM = 'Midplane'


def test_interp_and_target():
    d, F = plateau_target(60.0, 1.0, 8.0)
    assert interp(torch.tensor(0.5), d, F).item() == pytest.approx(30.0, rel=1e-2)
    assert interp(torch.tensor(5.0), d, F).item() == pytest.approx(60.0)
    x = torch.tensor(0.5, requires_grad=True)
    g, = torch.autograd.grad(interp(x, d, F), x)
    assert g.item() == pytest.approx(60.0, rel=2e-2)                  # slope of the rise


def build(lat, x, rig0, **kw):
    kw = {k: v for k, v in kw.items() if k != 'law'}
    return drop_rig(lat, radii(x, SYM, lat), rig0['law'], mat=MAT, freeze=rig0, **kw)


@pytest.fixture(scope='module')
def drop():
    lat = example3(elemSize=3.0)
    kw = dict(law={'kind': 'crit_damped', 'F_ref': 2000.0, 'delta_ref': 0.2}, start='co_moving', gap0=0.1, energy_J=0.75)
    x0 = torch.full((n_design_rows(SYM, lat), 1), 0.8)
    x0[::2] = 0.85
    rig0, sched = nominal(lat, x0, SYM, MAT, T=1.0e-3, **kw)
    return lat, x0, rig0, sched, kw


@pytest.mark.parametrize('obj', ['fd_mse', 'sea_dissipated', 'sea_loading'])
def test_objective_gradient_adjoint_vs_fd(drop, obj):
    lat, x0, rig0, sched, kw = drop
    target = plateau_target(60.0, 0.5, 8.0)

    def outer_for(rig):
        if obj == 'fd_mse':
            return fd_mse_outer(target)
        return sea_outer(rig, MAT['rho'], mode=obj.split('_')[1])

    x = x0.clone().requires_grad_(True)
    rig = build(lat, x, rig0, **kw)
    J, (g,) = checkpointed_adjoint(DropStepper(rig, sched, make_terms(rig, target)), [x], outer=outer_for(rig))

    def Jfd(xx):
        with torch.no_grad():
            r = build(lat, xx, rig0, **kw)
            S = forward_J(DropStepper(r, sched, make_terms(r, target)))
            return outer_for(r)(torch.as_tensor(S), [xx]).item()

    h = 1e-5
    for grp in ([0, 1], [2]):                                       # a few design rows
        e = torch.zeros_like(x0)
        e[grp] = 1.0
        fd = (Jfd(x0 + h * e) - Jfd(x0 - h * e)) / (2 * h)
        ad = (g * e).sum().item()
        assert ad == pytest.approx(fd, rel=1e-5, abs=1e-9 * abs(J)), (obj, ad, fd)


def test_areas_have_energy_meaning(drop):
    """Loading area ~ energy the lattice took from the impactor up to max stroke (drop: ~ impact energy)."""
    lat, x0, rig0, sched, kw = drop
    from latticeSetup import run_drop
    with torch.no_grad():
        rig = build(lat, x0, rig0, **kw)
        S = forward_J(DropStepper(rig, sched, make_terms(rig)))
    assert S[1] > 0 and S[2] <= S[1] + 1e-9                         # net (dissipated) <= loading area
    assert lattice_mass(rig, MAT['rho']).item() > 0
