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
