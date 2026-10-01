"""Day 5: design gradients through the drop (autograd) vs central finite differences.

3 mm mesh, short window, frozen rig constants and replayed dt schedule; placeholder J only (objective
deferred). Exercises the visco-hyperelastic fibre law, line contact + friction and plate contact.
"""
import time

import pytest
import torch

from dropGrad import nominal, evaluate
from latticeSetup import example3, symmetry_groups, n_design_rows
from material import project_material

torch.set_default_dtype(torch.float64)
MAT = project_material()
RIG = dict(law={'kind': 'crit_damped', 'F_ref': 2000.0, 'delta_ref': 0.2}, start='co_moving', gap0=0.1,
           energy_J=0.75)


@pytest.fixture(scope='module')
def setup():
    lat = example3(elemSize=3.0)
    return lat


# 1 ms windows: a full autograd graph of 2.5 ms of contact dynamics took 47 GB RAM (needs checkpointing)
@pytest.mark.parametrize('symType,T', [('OneValue', 1.0e-3), ('Midplane', 1.0e-3)])
def test_autograd_matches_fd(setup, symType, T):
    lat = setup
    groups = symmetry_groups(symType, lat)
    x0 = torch.full((n_design_rows(symType, lat), 1), 0.8)
    rig0, sched = nominal(lat, x0, symType, MAT, T=T, **RIG)
    x = x0.clone().requires_grad_(True)
    t0 = time.time()
    J = evaluate(lat, x, symType, MAT, rig0, sched, **RIG)
    g, = torch.autograd.grad(J, x)
    t_ad = time.time() - t0
    h = 1e-5
    for gr in groups[:3]:
        gr = torch.as_tensor(gr).reshape(-1)
        xp, xm = x0.clone(), x0.clone()
        xp[gr] += h
        xm[gr] -= h
        with torch.no_grad():
            fd = (evaluate(lat, xp, symType, MAT, rig0, sched, **RIG) - evaluate(lat, xm, symType, MAT, rig0, sched, **RIG)).item() / (2 * h)
        ad = g[gr].sum().item()
        print(f'{symType} group size {len(gr)}: AD {ad:.6e} FD {fd:.6e} rel {abs(ad - fd) / abs(fd):.1e} (AD+backward {t_ad:.0f}s, {len(sched)} steps)')
        assert ad == pytest.approx(fd, rel=1e-4), (ad, fd)
