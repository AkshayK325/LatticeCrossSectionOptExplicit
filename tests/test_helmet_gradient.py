"""Helmet patch (shell + curved C4 liner + free head + anvil): checkpointed adjoint = unrolled autograd = central FD for the
head-acceleration objective w.r.t. the Midplane group radii, with the R5_C4 joint rule, smooth penalty contacts."""
import torch

from adjoint import DropStepper, checkpointed_adjoint
from latticeSetup import example3, n_design_rows, element_values, symmetry_groups
from material import project_material
from helmetRig import helmet_rig
import jointRule

torch.set_default_dtype(torch.float64)
MAT = project_material()
SYM = 'Midplane'
T = 2.0e-4


def _J(inner0):
    def J_fn(zin, zout, h):
        return zout['a'][inner0, 2] ** 2 * h * 1e-6
    return J_fn


def _setup():
    lat = example3(elemSize=3.0)
    x0 = torch.full((n_design_rows(SYM, lat), 1), 0.8)
    for g, val in zip(symmetry_groups(SYM, lat), (0.75, 0.85, 0.65)):
        x0[torch.as_tensor(g)] = val
    kw = dict(joint=dict(jointRule.PRESETS['R5_C4']))
    r0 = element_values(x0, SYM, lat)[:, 0]
    rig0 = helmet_rig(lat, r0, MAT, **kw)
    sched, u, R, v = [], None, None, None
    from explicitSolve import simulate
    with torch.no_grad():
        h = simulate(rig0['fr'], rig0['beam_law'], rig0['m'], rig0['j'], rig0['dt_fn'], None, t_end=T, v0=rig0['v_init'],
                     fix_u=rig0['fix_u'], fix_r=rig0['fix_r'], contact=rig0['contact'], tie=rig0['tie'])
    return lat, x0, kw, rig0, [float(d) for d in h['dt']]


def _rig(lat, x, kw, rig0):
    r = element_values(x, SYM, lat)[:, 0]
    return helmet_rig(lat, r, MAT, freeze=rig0, **kw)


def test_helmet_adjoint_ad_fd():
    lat, x0, kw, rig0, sched = _setup()
    i0 = rig0['inner'][0].item()
    x = x0.clone().requires_grad_(True)
    st = DropStepper(_rig(lat, x, kw, rig0), sched, _J(i0))
    z, J = st.initial_state(), 0.0
    for n in range(len(sched)):
        z, L = st.step(z, n)
        J = J + L
    g_ad, = torch.autograd.grad(J, x)
    x2 = x0.clone().requires_grad_(True)
    st2 = DropStepper(_rig(lat, x2, kw, rig0), sched, _J(i0))
    J2, (g_adj,) = checkpointed_adjoint(st2, [x2], K=11)
    print('J', J.item(), J2, 'ad', g_ad.flatten().tolist(), 'adj', g_adj.flatten().tolist())
    assert abs(J.item() - J2) < 1e-10 * abs(J2)
    assert torch.allclose(g_adj, g_ad, rtol=1e-8, atol=1e-10 * g_ad.abs().max().item()), (g_adj - g_ad).abs().max()
    d = torch.zeros_like(x0); d[torch.as_tensor(symmetry_groups(SYM, lat)[1])] = 1.0; h = 1e-5
    def Jf(xx):
        s = DropStepper(_rig(lat, xx, kw, rig0), sched, _J(i0))
        zz, JJ = s.initial_state(), 0.0
        with torch.no_grad():
            for n in range(len(sched)):
                zz, L = s.step(zz, n)
                JJ = JJ + L
        return JJ.item()
    fd = (Jf(x0 + h * d) - Jf(x0 - h * d)) / (2 * h)
    ad = (g_adj * d).sum().item()
    print('FD', fd, 'adjoint', ad)
    assert abs(ad - fd) < 1e-5 * max(abs(fd), 1e-12) + 1e-9
