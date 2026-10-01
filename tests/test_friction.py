"""Coulomb friction (mu = 0.5) on contacts: contact.coulomb and LineContact friction."""
import numpy as np
import pytest
import torch

from contact import coulomb, LineContact, potential_force, _line_friction
from material import project_material
from test_beam_on_beam import crossed, RA, RB

torch.set_default_dtype(torch.float64)
MAT = project_material()
MU = 0.5


def test_coulomb_slip_and_stability_bound():
    Fn = torch.tensor([2.0, 2.0])
    vt = torch.tensor([[500.0, 0, 0], [0.0, 1e-3, 0]])            # sliding / nearly stuck
    m, dt = torch.tensor([1e-6, 1e-6]), 1e-6                      # v_eps = 2 mu Fn dt / m = 2 mm/s
    Ft = coulomb(Fn, vt, MU, m, dt)
    assert Ft[0, 0].item() == pytest.approx(-MU * 2.0, rel=1e-3)   # slip: |F| = mu Fn, opposite to vt
    c = (Ft[1].norm() / vt[1].norm()).item()                        # equivalent viscous coefficient near stick
    assert c <= m[1].item() / (2 * dt) * (1 + 1e-9)
    assert (Ft * vt).sum(1).le(0).all()                             # always dissipative


def test_line_friction_is_mu_times_normal_and_balanced():
    fr, r, beam = crossed()
    lc = LineContact(fr, r, beam, MAT['E0'], MAT['nu'], h_q=0.02, eps=0.002)
    X = fr.X.clone()
    X[fr.N // 2:, 2] = RA + RB - 0.03                               # pressed
    f_n, _, _ = potential_force(lc.energy, X)
    Fn_total = f_n[fr.N // 2:, 2].sum().item()
    v = torch.zeros(fr.N, 3)
    v[fr.N // 2:, 0] = 200.0                                        # B slides along +x over A
    m = torch.full((fr.N,), 1.0)                                    # heavy: Coulomb regime
    f, P = _line_friction(lc, X, v, MU, m, 1e-6)
    FtB = f[fr.N // 2:].sum(0)
    assert FtB[0].item() == pytest.approx(-MU * Fn_total, rel=2e-2)  # opposes slip, magnitude mu Fn
    assert abs(FtB[1].item()) < 1e-6 * abs(FtB[0].item()) + 1e-12 and abs(FtB[2].item()) < 1e-6 * abs(FtB[0].item()) + 1e-12
    assert torch.allclose(f.sum(0), torch.zeros(3), atol=1e-9)      # equal and opposite on A
    assert P == pytest.approx(MU * Fn_total * 200.0, rel=2e-2)       # dissipated power = mu Fn |v|
