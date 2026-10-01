"""EPU 45 visco-hyperelastic section law (explicitFE.ViscoHyperLaw) on a single bar element."""
import math

import numpy as np
import pytest
import torch

from explicitFE import Frame, ViscoHyperLaw
from material import epu45, yeoh_uniaxial

torch.set_default_dtype(torch.float64)
MAT = epu45()


def bar(L=3.0, r=1.0):
    fr = Frame(np.array([[0.0, 0, 0], [L, 0, 0]]), np.array([[0, 1]]))
    A = torch.full((1, 3), math.pi * r**2)
    I = torch.tensor([[math.pi * r**4 / 2, math.pi * r**4 / 4, math.pi * r**4 / 4]])
    return fr, ViscoHyperLaw(fr, A, I, MAT), A[0, 0].item()


def axial_d(fr, strain):
    d = torch.zeros(1, 6)
    d[0, 0] = strain * fr.L0[0]
    return d


def test_epu45_material_values():
    assert MAT['rho'] == 1.06e-9                                    # 1060 kg/m^3 in t/mm^3
    assert MAT['E'] == pytest.approx(6 * MAT['yeoh'][0])            # E_inf = 6 C10 = 14.4 MPa
    assert MAT['E0'] == pytest.approx(342.86, rel=1e-3)             # E_inf / (1 - 0.958)
    assert MAT['nu'] == 0.49


def test_long_term_axial_is_yeoh():
    fr, law, A = bar()
    for e in (-0.5, -0.2, 0.01, 0.5, 1.0):
        N = law.f_inf(axial_d(fr, e))[0, 0].item()
        assert N == pytest.approx(A * yeoh_uniaxial(torch.tensor(1 + e), MAT['yeoh']).item(), rel=1e-12)
    # small-strain slope = 6 C10 = E_inf
    assert law.f_inf(axial_d(fr, 1e-7))[0, 0].item() / (A * 1e-7) == pytest.approx(MAT['E'], rel=1e-5)


def test_step_strain_relaxes_from_E0_to_Einf():
    """Step strain at t=0 then held: N(t) = A eps [E_inf + (E0 - E_inf) exp(-t/tau)] (small strain)."""
    fr, law, A = bar()
    eps, tau, h = 1e-5, MAT['prony_tau'][0], 2e-6
    d = axial_d(fr, eps)
    # simulate() order: law(d) = force at the end of the step from the committed history, then commit(d)
    law.step_dt = 1e-12                                             # (near) instantaneous step
    assert law(d)[0, 0].item() / (A * eps) == pytest.approx(MAT['E0'], rel=1e-4)
    law.commit(d)
    law.step_dt = h
    t = 1e-12
    for _ in range(100):
        t += h
        exact = MAT['E'] + (MAT['E0'] - MAT['E']) * math.exp(-t / tau)
        assert law(d)[0, 0].item() / (A * eps) == pytest.approx(exact, rel=1e-4)   # Yeoh is O(eps) nonlinear
        law.commit(d)


def test_harmonic_loss_matches_loss_modulus():
    """Energy dissipated per cycle = pi E'' eps0^2 * volume, E'' = (E0 - Einf) w tau / (1 + (w tau)^2)."""
    fr, law, A = bar()
    eps0, f, tau = 1e-4, 1000.0, MAT['prony_tau'][0]
    w = 2 * math.pi * f
    n = 2000
    h = 1 / f / n
    law.step_dt = h
    W, d_prev, N_prev = 0.0, axial_d(fr, 0.0), 0.0
    for k in range(1, 3 * n + 1):                                   # 3 cycles, measure the last
        d = axial_d(fr, eps0 * math.sin(w * k * h))
        N = law(d)[0, 0].item()
        law.commit(d)
        if k > 2 * n:
            W += 0.5 * (N + N_prev) * (d[0, 0] - d_prev[0, 0]).item()
        d_prev, N_prev = d, N
    Epp = (MAT['E0'] - MAT['E']) * w * tau / (1 + (w * tau) ** 2)
    assert W == pytest.approx(math.pi * Epp * eps0**2 * A * fr.L0[0].item(), rel=2e-2)
