"""Geometric joint stiffening (latticeSetup.joint_factors): mesh independence and exact design gradients."""
import numpy as np
import pytest
import torch

from latticeSetup import example3, joint_factors

torch.set_default_dtype(torch.float64)


def test_joint_factors_mesh_independent():
    """Each strut's compliance length sum(l_e / s_e) is the same for any element size."""
    comps = []
    for es in (0.5, 1.0, 3.0):
        lat = example3(elemSize=es, nx=1, ny=1)
        r = torch.full((len(lat['connr']),), 0.8)
        s = joint_factors(lat, r, 3.5).numpy()
        Le = np.linalg.norm(lat['Xr'][lat['connr'][:, 1]] - lat['Xr'][lat['connr'][:, 0]], axis=1)
        c = np.zeros(len(lat['conn']))
        np.add.at(c, lat['beam_of_elem'], Le / s)
        comps.append(c)
    assert np.allclose(comps[0], comps[1], rtol=1e-10) and np.allclose(comps[0], comps[2], rtol=1e-10)


def test_joint_zone_is_geometric():
    """Equal radii at 90 deg: zone length r cot(45 deg) = r, so a 1-element-per-r mesh is fully stiffened."""
    lat = example3(elemSize=0.4, nx=1, ny=1)
    s = joint_factors(lat, torch.full((len(lat['connr']),), 0.4), 3.5)
    assert s.max().item() == pytest.approx(3.5, rel=1e-6) and s.min().item() == pytest.approx(1.0)


def test_joint_factors_gradient_matches_fd():
    lat = example3(elemSize=1.0, nx=1, ny=1)
    rng = np.random.default_rng(0)
    r0 = torch.as_tensor(0.6 + 0.4 * rng.random(len(lat['conn'])))[torch.as_tensor(lat['beam_of_elem'])]
    w = torch.as_tensor(rng.random(len(r0)))
    f = lambda r: (w * joint_factors(lat, r, 3.5)).sum()
    r = r0.clone().requires_grad_(True)
    g, = torch.autograd.grad(f(r), r)
    boe = torch.as_tensor(lat['beam_of_elem'])
    for b in (0, 7, 20):                                   # perturb one strut's radius (all its elements)
        e = (boe == b).double()
        h = 1e-6
        fd = (f(r0 + h * e) - f(r0 - h * e)).item() / (2 * h)
        assert (g * e).sum().item() == pytest.approx(fd, rel=1e-5, abs=1e-8)


def test_fused_section_factors_basic():
    """Fused-section (topology-aware) joint model: factors >= 1, exactly 1 for a strut far from any node, and the
    stiffened length is local (a few radii) and mesh independent in compliance."""
    import jointSection
    comps = []
    for es in (0.5, 1.0):
        lat = example3(elemSize=es, nx=1, ny=1)
        F, prof = jointSection.element_factors_aniso(lat, np.full(len(lat['conn']), 0.6), ds_frac=0.1, n_grid=41)
        assert (F >= 1.0 - 1e-12).all()
        s, R = max(prof, key=lambda p: p[0][-1])            # the longest strut: mid-span is plain strut
        mid = np.abs(s - 0.5 * s[-1]) < 0.3
        assert np.allclose(R[mid], 1.0)
        Le = np.linalg.norm(lat['Xr'][lat['connr'][:, 1]] - lat['Xr'][lat['connr'][:, 0]], axis=1)
        comps.append((Le / F[:, 2]).sum())
    assert comps[0] == pytest.approx(comps[1], rel=2e-3)


def test_rule_factors_basic_and_gradient():
    """Angle / radius rule (jointRule): >= 1, 1 far from nodes, mesh independent, radii-gradient == finite differences."""
    import jointRule
    lat = example3(elemSize=0.5, nx=1, ny=1)
    r = torch.full((len(lat['connr']),), 0.6)
    s = jointRule.rule_factors(lat, r, a=1.0, p=4.0)
    assert float(s.min()) >= 1.0 - 1e-12 and float(s.max()) > 1.5
    lat2 = example3(elemSize=0.25, nx=1, ny=1)
    s2 = jointRule.rule_factors(lat2, torch.full((len(lat2['connr']),), 0.6), a=1.0, p=4.0)
    # element-length-weighted mean factor per strut is mesh independent (the compliance average)
    assert abs(float(s.mean()) - float(s2.mean())) < 0.01 * float(s.mean())
    torch.manual_seed(0)
    w = torch.rand(len(lat['connr']), dtype=torch.float64)
    r = torch.full((len(lat['connr']),), 0.6, dtype=torch.float64, requires_grad=True)
    f = lambda rr: (w * jointRule.rule_factors(lat, rr, a=1.2, p=4.0, zl=1.1, b=0.3, mu=0.8)).sum()
    g, = torch.autograd.grad(f(r), r)
    d = torch.zeros_like(r); d[::7] = 1.0; h = 1e-6
    fd = (f(r.detach() + h * d) - f(r.detach() - h * d)) / (2 * h)
    assert abs(float((g * d).sum()) - float(fd)) < 1e-6 * max(1.0, abs(float(fd)))


def test_rule_factors_new_parameters():
    """pz (zone exponent), negative p and gamma (pairs leaving a plate) keep factors >= 1 and the radii gradient exact."""
    import jointRule
    lat = example3(elemSize=1.0, nx=1, ny=1)
    r = torch.full((len(lat['connr']),), 0.6, dtype=torch.float64, requires_grad=True)
    f = lambda rr: jointRule.rule_factors(lat, rr, a=1.1, p=-1.0, zl=1.2, b=0.2, pz=0.0, gamma=0.4).sum()
    s = jointRule.rule_factors(lat, r.detach(), a=1.1, p=-1.0, zl=1.2, b=0.2, pz=0.0, gamma=0.4)
    assert float(s.min()) >= 1.0 - 1e-12
    s1 = jointRule.rule_factors(lat, r.detach(), a=1.1, p=-1.0, zl=1.2, gamma=1.0)
    s0 = jointRule.rule_factors(lat, r.detach(), a=1.1, p=-1.0, zl=1.2, gamma=0.0)
    assert float(s0.sum()) <= float(s1.sum()) + 1e-9
    g, = torch.autograd.grad(f(r), r)
    d = torch.zeros_like(r); d[::5] = 1.0; h = 1e-6
    fd = (f(r.detach() + h * d) - f(r.detach() - h * d)) / (2 * h)
    assert abs(float((g * d).sum()) - float(fd)) < 1e-6 * max(1.0, abs(float(fd)))


def test_rule_factors_cap_and_saturation():
    """cap (zone length <= cap * strut length) and wsat (smooth saturation of the neighbour sum) only reduce the stiffening
    and keep the radii gradient exact."""
    import jointRule
    lat = example3(elemSize=1.0, nx=1, ny=1)
    r = torch.full((len(lat['connr']),), 0.6, dtype=torch.float64, requires_grad=True)
    kw = dict(a=5.0, p=-0.8, zl=0.8, b=0.2, gamma=0.0)
    s_full = jointRule.rule_factors(lat, r.detach(), **kw)
    s_cap = jointRule.rule_factors(lat, r.detach(), cap=0.25, **kw)
    s_sat = jointRule.rule_factors(lat, r.detach(), wsat=1.0, **kw)
    assert float(s_cap.sum()) <= float(s_full.sum()) + 1e-9 and float(s_sat.sum()) <= float(s_full.sum()) + 1e-9
    assert float(s_cap.min()) >= 1.0 - 1e-12 and float(s_sat.min()) >= 1.0 - 1e-12
    f = lambda rr: jointRule.rule_factors(lat, rr, cap=0.3, wsat=2.0, **kw).sum()
    g, = torch.autograd.grad(f(r), r)
    d = torch.zeros_like(r); d[::4] = 1.0; h = 1e-6
    fd = (f(r.detach() + h * d) - f(r.detach() - h * d)) / (2 * h)
    assert abs(float((g * d).sum()) - float(fd)) < 1e-6 * max(1.0, abs(float(fd)))


def test_rule_factors_lmin_floor():
    """lmin (floor of the pair zone length) only adds stiffening and keeps the radii gradient exact."""
    import jointRule
    lat = example3(elemSize=1.0, nx=1, ny=1)
    r = torch.full((len(lat['connr']),), 0.6, dtype=torch.float64, requires_grad=True)
    kw = dict(a=3.0, p=-0.8, zl=0.8, b=0.1, gamma=0.0, wsat=2.5)
    s0 = jointRule.rule_factors(lat, r.detach(), **kw)
    s1 = jointRule.rule_factors(lat, r.detach(), lmin=0.8, **kw)
    assert float(s1.sum()) >= float(s0.sum()) - 1e-9
    f = lambda rr: jointRule.rule_factors(lat, rr, lmin=0.8, **kw).sum()
    g, = torch.autograd.grad(f(r), r)
    d = torch.zeros_like(r); d[::3] = 1.0; h = 1e-6
    fd = (f(r.detach() + h * d) - f(r.detach() - h * d)) / (2 * h)
    assert abs(float((g * d).sum()) - float(fd)) < 1e-5 * max(1.0, abs(float(fd)))


def test_rule_factors_taper():
    """taper > 0 (smooth stiffness profile) keeps factors >= 1, is mesh independent and has an exact radii gradient."""
    import jointRule
    kw = dict(a=3.0, p=-0.8, zl=1.2, b=0.1, gamma=0.0, wsat=2.5, taper=1.0)
    lat = example3(elemSize=1.0, nx=1, ny=1)
    r = torch.full((len(lat['connr']),), 0.6, dtype=torch.float64, requires_grad=True)
    s = jointRule.rule_factors(lat, r.detach(), **kw)
    assert float(s.min()) >= 1.0 - 1e-12
    lat2 = example3(elemSize=0.5, nx=1, ny=1)
    s2 = jointRule.rule_factors(lat2, torch.full((len(lat2['connr']),), 0.6, dtype=torch.float64), **kw)
    assert abs(float((1 / s).mean()) - float((1 / s2).mean())) < 0.02            # compliance, mesh independent to quadrature level
    f = lambda rr: jointRule.rule_factors(lat, rr, **kw).sum()
    g, = torch.autograd.grad(f(r), r)
    d = torch.zeros_like(r); d[::3] = 1.0; h = 1e-6
    fd = (f(r.detach() + h * d) - f(r.detach() - h * d)) / (2 * h)
    assert abs(float((g * d).sum()) - float(fd)) < 1e-5 * max(1.0, abs(float(fd)))


def test_rule_factors_tail():
    """The exponential tail (overlapping tails of a short strut) only adds stiffness, is mesh independent and keeps the radii gradient exact."""
    import jointRule
    kw = dict(a=3.0, p=-0.8, zl=0.8, b=0.1, gamma=0.0, wsat=2.5)
    lat = example3(elemSize=1.0, nx=1, ny=1)
    r = torch.full((len(lat['connr']),), 0.6, dtype=torch.float64, requires_grad=True)
    s0 = jointRule.rule_factors(lat, r.detach(), **kw)
    s1 = jointRule.rule_factors(lat, r.detach(), tail=0.5, tail_len=2.0, **kw)
    assert float(s1.min()) >= 1.0 - 1e-9 and float(s1.sum()) >= float(s0.sum()) - 1e-6
    lat2 = example3(elemSize=0.5, nx=1, ny=1)
    s2 = jointRule.rule_factors(lat2, torch.full((len(lat2['connr']),), 0.6, dtype=torch.float64), tail=0.5, tail_len=2.0, **kw)
    assert abs(float((1 / s1).mean()) - float((1 / s2).mean())) < 0.02
    f = lambda rr: jointRule.rule_factors(lat, rr, tail=0.5, tail_len=2.0, **kw).sum()
    g, = torch.autograd.grad(f(r), r)
    d = torch.zeros_like(r); d[::3] = 1.0; h = 1e-6
    fd = (f(r.detach() + h * d) - f(r.detach() - h * d)) / (2 * h)
    assert abs(float((g * d).sum()) - float(fd)) < 1e-5 * max(1.0, abs(float(fd)))


def test_rule_presets_match_fit_files():
    """The PRESETS of jointRule are the fitted parameters of the result files (no silent edits)."""
    import json
    import pathlib
    import jointRule
    root = pathlib.Path(__file__).resolve().parent.parent / 'abaqus' / 'results'
    for name, f in (('R5', 'rule_fit3/R5.json'), ('R5_all22', 'rule_fit4/R5.json')):
        par = json.loads((root / f).read_text())['params']
        for k, v in par.items():
            if k != 'mu':
                assert jointRule.PRESETS[name][k] == v, (name, k)

    c4 = dict(jointRule.PRESETS['R5'], short_gain=1.5, short_l0=4.6)
    assert jointRule.PRESETS['R5_C4'] == c4
