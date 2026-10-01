"""Day 4: capsule beam-to-beam contact (contact.SelfContact)."""
import math

import numpy as np
import pytest
import torch

from contact import SelfContact, potential_force, hertz_k_crossed_cylinders
from explicitFE import Frame, lumped_mass, stable_dt
from material import project_material

torch.set_default_dtype(torch.float64)
MAT = project_material()
R = 0.8


def crossed_beams(z_gap_centres, n_el=2, L=6.0, parallel=False):
    """Beam A along x at z=0, beam B along y (or x if parallel) at height z, crossing over the centre.
    The Frame (reference state) is always built separated (z = 2R + 1): pairs overlapping in the reference
    are excluded by design, so contact states are imposed as positions, not as the reference."""
    xa = np.stack([np.linspace(-L / 2, L / 2, n_el + 1), np.zeros(n_el + 1), np.zeros(n_el + 1)], 1)
    t = np.linspace(-L / 2, L / 2, n_el + 1)
    xb = np.stack([t, np.zeros(n_el + 1), np.full(n_el + 1, z_gap_centres)], 1) if parallel else \
        np.stack([np.zeros(n_el + 1), t, np.full(n_el + 1, z_gap_centres)], 1)
    X = np.vstack((xa, xb))
    ca = np.stack([np.arange(n_el), np.arange(1, n_el + 1)], 1)
    conn = np.vstack((ca, ca + n_el + 1))
    X_ref = X.copy()
    X_ref[n_el + 1:, 2] = 2 * R + 1.0
    fr = Frame(X_ref, conn)
    fr.X_pos = torch.as_tensor(X)                          # the requested (possibly touching) positions
    return fr


EPS = 0.04


def make(fr, r=R):
    beam = torch.arange(fr.ne) // (fr.ne // 2)                     # two beams of ne/2 elements each
    return SelfContact(fr, torch.full((fr.ne,), r), MAT['E0'], MAT['nu'], eps=EPS, beam_of_elem=beam)


def test_force_zero_until_touch_then_hertz():
    k = hertz_k_crossed_cylinders(torch.tensor(R), torch.tensor(R), MAT['E0'], MAT['nu']).item()
    for zc, expect in ((2 * R + 0.5, 0.0), (2 * R - 0.05, None), (2 * R - 0.2, k * 0.2**1.5)):
        fr = crossed_beams(zc)
        sc = make(fr)
        f, Phi, g = potential_force(sc.energy, fr.X_pos.clone())
        fz_b = f[fr.N // 2:, 2].sum().item()                      # on beam B (upper): pushed up
        assert torch.allclose(f.sum(0), torch.zeros(3), atol=1e-9)  # equal and opposite
        assert f[:, :2].abs().max() < 1e-9                        # normal is z
        if expect == 0.0:
            assert abs(fz_b) < 1e-12
        else:
            assert fz_b > 0
            if expect is not None:
                assert fz_b == pytest.approx(expect, rel=2e-2)      # deep: softplus -> exact Hertz


def test_force_is_minus_gradient_and_radius_derivative():
    fr = crossed_beams(2 * R - 0.1)
    X = fr.X_pos.clone()
    X[3, 0] += 0.3                                                 # make it asymmetric (off-centre contact)
    sc = make(fr)
    f, _, _ = potential_force(sc.energy, X)
    h = 1e-6
    for node, comp in ((1, 2), (4, 2), (4, 0), (3, 1)):
        Xp, Xm = X.clone(), X.clone()
        Xp[node, comp] += h
        Xm[node, comp] -= h
        fd = -(sc.energy(Xp)[0] - sc.energy(Xm)[0]).item() / (2 * h)
        assert f[node, comp].item() == pytest.approx(fd, rel=1e-5, abs=1e-6 * f.abs().max().item())
    # d Phi / d r (design sensitivity through the gap)
    r = torch.full((fr.ne,), R, requires_grad=True)
    beam = torch.arange(fr.ne) // (fr.ne // 2)
    sc2 = SelfContact(fr, r, MAT['E0'], MAT['nu'], eps=EPS, beam_of_elem=beam)
    Phi, _ = sc2.energy(X)
    dr, = torch.autograd.grad(Phi, r)
    for e in range(fr.ne):
        rp, rm = r.detach().clone(), r.detach().clone()
        rp[e] += h
        rm[e] -= h
        mk = lambda rr: SelfContact(fr, rr, MAT['E0'], MAT['nu'], eps=EPS, beam_of_elem=beam)
        fd = (mk(rp).energy(X)[0] - mk(rm).energy(X)[0]).item() / (2 * h)
        assert dr[e].item() == pytest.approx(fd, rel=1e-4, abs=1e-9)


def test_parallel_segments_no_nan():
    fr = crossed_beams(2 * R - 0.1, parallel=True)
    f, Phi, g = potential_force(make(fr).energy, fr.X_pos.clone())
    assert torch.isfinite(f).all() and Phi > 0
    assert f[:, :2].abs().max() < 1e-9 and f[fr.N // 2:, 2].sum() > 0


def test_lattice_pair_exclusion():
    from latticeSetup import example3
    lat = example3()
    fr = Frame(lat['Xr'], lat['connr'])
    sc = SelfContact(fr, torch.full((fr.ne,), R), MAT['E0'], MAT['nu'])
    ne = fr.ne
    assert len(sc.pairs) > 0.9 * ne * (ne - 1) / 2                 # nearly all pairs are candidates
    assert sc.n_initial_overlap == 0, sc.n_initial_overlap         # hop exclusion removes all joint overlaps


def test_free_beams_collide_energy_conserved():
    """Two free elastic beams (equal mass) collide crosswise: contact is conservative and frictionless, so
    z-momentum and energy are conserved, the surfaces never interpenetrate, and B hands its momentum to A."""
    from explicitSolve import simulate, strain_energy
    fr = crossed_beams(2 * R + 0.3, n_el=4)
    A = torch.full((fr.ne, 3), math.pi * R**2)
    I = torch.tensor([[math.pi * R**4 / 2, math.pi * R**4 / 4, math.pi * R**4 / 4]]).repeat(fr.ne, 1)
    law = lambda d: fr.elastic_law(d, A, I, MAT['E'], MAT['G'])
    m, j = lumped_mass(fr, A, I, MAT['rho'])
    sc = make(fr)
    v0 = torch.zeros(fr.N, 3)
    v0[fr.N // 2:, 2] = -2000.0                                     # beam B down at 2 m/s onto free beam A
    _, wb = stable_dt(fr, law, m, j)
    dt = 0.2 * 2 / math.sqrt(wb**2 + 4 * 300 / m.min().item())
    E, P, G = [], [], []
    a, b = slice(0, fr.N // 2), slice(fr.N // 2, fr.N)

    def cb(n, t, state, fint):
        u, v, R_, w = state
        ke = 0.5 * (m * (v**2).sum(1)).sum() + 0.5 * (j * (w**2).sum(1)).sum()
        Phi, g = sc.energy(fr.X + u)
        E.append((ke + strain_energy(fint[2], fint[3]) + Phi).item())
        P.append((m * v[:, 2]).sum().item())
        G.append(g.min().item() if len(g) else np.inf)
        return {'vA': (m[a] * v[a, 2]).sum().item() / m[a].sum().item(),
                'vB': (m[b] * v[b, 2]).sum().item() / m[b].sum().item()}

    with torch.no_grad():
        h = simulate(fr, law, m, j, dt, int(1e-3 / dt), v0=v0, u0=fr.X_pos - fr.X, callback=cb,
                     contact=lambda u, v: potential_force(sc.energy, fr.X + u)[0])
    E, P = np.array(E), np.array(P)
    assert np.abs(E / E[0] - 1).max() < 1e-2
    assert np.abs(P - P[0]).max() < 1e-9 * abs(P[0])
    assert min(G) > -0.05 * R                                       # surfaces do not interpenetrate
    assert h['vA'][-1] < -1500 and abs(h['vB'][-1]) < 500           # momentum handed over (equal masses)
