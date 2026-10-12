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


# ---------------------------------------------------------------- large-strain bending / torsion (fibre model)
def direct_section(fun, r, n=801):
    """Integrate fun(y, z) over a solid circle of radius r on a fine Cartesian grid (independent check)."""
    t = np.linspace(-r, r, n)
    Y, Z = np.meshgrid(t, t, indexing='ij')
    inside = Y**2 + Z**2 <= r * r
    dA = (t[1] - t[0]) ** 2
    return (fun(Y, Z) * inside).sum() * dA


def test_fibre_small_strain_equals_linear_timoshenko():
    fr, law, A = bar()
    g = torch.Generator().manual_seed(0)
    d = 1e-7 * torch.randn(1, 6, generator=g)
    f_lin = fr.elastic_law(d, law.A, law.I, MAT['E'], MAT['G'])
    assert torch.allclose(law.f_inf(d), f_lin, rtol=1e-4, atol=1e-12)


@pytest.mark.parametrize('kappa', [0.05, 0.2, 0.4])
def test_fibre_pure_bending_vs_direct_integration(kappa):
    """kappa r = 0.04..0.32 surface strain. Antisymmetric end rotations = constant curvature (no shear)."""
    fr, law, A = bar(r=1.0)
    L = fr.L0[0].item()
    d = torch.zeros(1, 6)
    d[0, 3], d[0, 5] = -kappa * L / 2, kappa * L / 2                # rz1, rz2 -> kappa_z
    f = law.f_inf(d)[0]
    C = MAT['yeoh']
    P = lambda lam: yeoh_uniaxial(torch.as_tensor(lam), C).numpy()
    M = direct_section(lambda y, z: P(1 - y * kappa) * (-y), 1.0)
    N = direct_section(lambda y, z: P(1 - y * kappa), 1.0)
    assert f[5].item() == pytest.approx(M, rel=1e-3)
    assert f[3].item() == pytest.approx(-M, rel=1e-3)
    # tension/compression asymmetry of Yeoh: pure bending carries a net axial force (fibre model only)
    assert abs(N) > 0 and law.fibre


@pytest.mark.parametrize('twist', [0.05, 0.2, 0.4])
def test_fibre_torsion_vs_direct_integration(twist):
    fr, law, A = bar(r=1.0)
    L = fr.L0[0].item()
    d = torch.zeros(1, 6)
    d[0, 1] = twist * L
    T = law.f_inf(d)[0, 1].item()
    C10, C20, C30 = MAT['yeoh']
    def tau(y, z):
        rho = np.sqrt(y * y + z * z)
        gam = twist * rho
        return 2 * gam * (C10 + 2 * C20 * gam**2 + 3 * C30 * gam**4) * rho
    T_direct = direct_section(tau, 1.0)
    # linear part uses G = E/(2(1+nu)) with nu = 0.49 vs Yeoh's 2 C10 (nu = 0.5): 0.7% at small strain
    assert T == pytest.approx(T_direct, rel=1e-2)


def _ramp_axial(law, fr, strain, t_ramp, t_hold, n=4000):
    """Drive one bar element through a constant-rate axial ramp and a hold; returns (t, N)."""
    law.reset()
    t = np.linspace(0, t_ramp + t_hold, n)
    h = t[1] - t[0]
    out = []
    with torch.no_grad():
        for k in range(1, n):
            law.step_dt = h
            d = axial_d(fr, strain * min(t[k] / t_ramp, 1.0))
            out.append(law(d)[0, 0].item())
            law.commit(d)
    return t[1:], np.array(out)


def _qlv_1d(strain, t_ramp, t_hold, measure, n=4000):
    """Independent 1D quasi-linear viscoelastic integral of the nominal or the Kirchhoff (= lambda P) stress."""
    g, tau = MAT['prony_g'][0], MAT['prony_tau'][0]
    beta = g / (1 - g)
    t = np.linspace(0, t_ramp + t_hold, n)
    h = t[1] - t[0]
    lam = 1 + strain * np.clip(t / t_ramp, 0, 1)
    P = yeoh_uniaxial(torch.tensor(lam), MAT['yeoh']).numpy()
    s_inf = P * lam if measure == 'kirchhoff' else P
    a, b = math.exp(-h / tau), (1 - math.exp(-h / tau)) / (h / tau)
    q = np.zeros(n)
    for k in range(1, n):
        q[k] = a * q[k - 1] + beta * b * (s_inf[k] - s_inf[k - 1])
    s = s_inf + q
    return t[1:], (s / lam if measure == 'kirchhoff' else s)[1:]


@pytest.mark.parametrize('relax', ['nominal', 'kirchhoff'])
def test_relax_measure_matches_1d_integral(relax):
    """30 % compression in 0.3 ms then a hold: the section law equals the independent 1D integral of its measure."""
    fr = Frame(np.array([[0.0, 0, 0], [3.0, 0, 0]]), np.array([[0, 1]]))
    A = torch.full((1, 3), math.pi)
    I = torch.tensor([[math.pi / 2, math.pi / 4, math.pi / 4]])
    law = ViscoHyperLaw(fr, A, I, MAT, relax=relax)
    t, N = _ramp_axial(law, fr, -0.3, 3e-4, 7e-4)
    _, P = _qlv_1d(-0.3, 3e-4, 7e-4, relax)
    assert np.abs(N / math.pi - P).max() < 0.01 * np.abs(P).max()


def test_relax_measures_agree_at_small_strain_and_long_term():
    fr = Frame(np.array([[0.0, 0, 0], [3.0, 0, 0]]), np.array([[0, 1]]))
    A = torch.full((1, 3), math.pi)
    I = torch.tensor([[math.pi / 2, math.pi / 4, math.pi / 4]])
    Ns = {r: _ramp_axial(ViscoHyperLaw(fr, A, I, MAT, relax=r), fr, -0.002, 3e-4, 2e-3)[1] for r in ('nominal', 'kirchhoff')}
    assert np.abs(Ns['nominal'] - Ns['kirchhoff']).max() < 0.005 * np.abs(Ns['nominal']).max()   # small strain
    Nl = {r: _ramp_axial(ViscoHyperLaw(fr, A, I, MAT, relax=r), fr, -0.3, 3e-4, 2e-3)[1][-1] for r in ('nominal', 'kirchhoff')}
    assert Nl['nominal'] == pytest.approx(Nl['kirchhoff'], rel=1e-3)                               # relaxed
