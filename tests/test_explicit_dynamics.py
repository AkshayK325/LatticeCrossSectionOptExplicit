"""Day 2/3 dynamic checks of explicitFE + explicitSolve on a TPU cantilever (E0 = 6.70 MPa).

Run as a script to write GIFs/plots to ResultsFig/explicit/.
"""
import math
import pathlib

import numpy as np
import torch

from explicitFE import Frame, lumped_mass, stable_dt
from explicitSolve import simulate, strain_energy
from material import tpu90a_impact

torch.set_default_dtype(torch.float64)
MAT = tpu90a_impact()
E, G, RHO = MAT['E'], MAT['G'], MAT['rho']
OUT = pathlib.Path(__file__).resolve().parents[1] / 'ResultsFig' / 'explicit'


def cantilever(L=30.0, n_el=20, r=1.0, rot_scale=True):
    X = np.stack([np.linspace(0, L, n_el + 1), np.zeros(n_el + 1), np.zeros(n_el + 1)], 1)
    conn = np.stack([np.arange(n_el), np.arange(1, n_el + 1)], 1)
    fr = Frame(X, conn)
    A = torch.full((n_el, 3), math.pi * r**2)
    I = torch.tensor([[math.pi * r**4 / 2, math.pi * r**4 / 4, math.pi * r**4 / 4]]).repeat(n_el, 1)
    law = lambda d: fr.elastic_law(d, A, I, E, G)
    m, j = lumped_mass(fr, A, I, RHO, rot_scale=rot_scale)
    dt, _ = stable_dt(fr, law, m, j, courant=0.8)
    fix = torch.zeros(fr.N, 3, dtype=torch.bool)
    fix[0] = True
    EI = E * math.pi * r**4 / 4
    return dict(fr=fr, A=A, I=I, law=law, m=m, j=j, dt=dt, fix=fix, L=L, EI=EI, rhoA=RHO * math.pi * r**2)


def static_tip_shape(c, w_tip):
    """EB static shape for a tip load (deflection in z, rotation about -y) scaled to tip deflection w_tip."""
    x = c['fr'].X[:, 0]
    L = c['L']
    w = w_tip * x**2 * (3 * L - x) / (2 * L**3)
    th = w_tip * 3 * x * (2 * L - x) / (2 * L**3)          # dw/dx
    u = torch.zeros(c['fr'].N, 3)
    u[:, 2] = w
    rot = torch.zeros(c['fr'].N, 3)
    rot[:, 1] = -th                                       # +z deflection = rotation about -y
    from explicitFE import rodrigues
    return u, rodrigues(rot)


def energy_cb(c):
    def cb(n, t, state, fint):
        u, v, R, w = state
        _, _, d, f_loc = fint[:4]
        ke = 0.5 * (c['m'] * (v**2).sum(1)).sum() + 0.5 * (c['j'] * (w**2).sum(1)).sum()
        return {'E': (ke + strain_energy(d, f_loc)).item(), 'tip': u[-1].clone()}
    return cb


def run_free_vibration(w_tip, periods=2.0, rot_scale=True, courant=0.8):
    c = cantilever(rot_scale=rot_scale)
    c['dt'] *= courant / 0.8
    w1 = 1.875104**2 * math.sqrt(c['EI'] / (c['rhoA'] * c['L']**4))
    n = int(periods * 2 * math.pi / w1 / c['dt'])
    u0, R0 = static_tip_shape(c, w_tip)
    with torch.no_grad():
        h = simulate(c['fr'], c['law'], c['m'], c['j'], c['dt'], n, u0=u0, R0=R0,
                     fix_u=c['fix'], fix_r=c['fix'], record_every=5, callback=energy_cb(c))
    return c, w1, h


def measured_omega(t, z):
    """From upward zero crossings of z - mean(z), linearly interpolated."""
    t, z = np.asarray(t), np.asarray(z) - np.mean(z)
    k = np.where((z[:-1] < 0) & (z[1:] >= 0))[0]
    tc = t[k] - z[k] * (t[k + 1] - t[k]) / (z[k + 1] - z[k])
    return 2 * math.pi / np.mean(np.diff(tc))


def test_first_frequency_matches_euler_bernoulli():
    _, w1, h = run_free_vibration(w_tip=0.05, periods=3.0)
    z = torch.stack(h['tip'])[:, 2].numpy()
    w = measured_omega(h['t'], z)
    assert abs(w - w1) / w1 < 0.02, (w, w1)


def test_energy_conserved_large_amplitude():
    """Tip 1/3 of L (geometrically nonlinear). Verlet energy error is a bounded O((w dt)^2) oscillation:
    ~5% at courant 0.8 on the high modes the release excites, ~0.3% at 0.2. Check the bound and no growth."""
    _, _, h = run_free_vibration(w_tip=10.0, periods=2.0, courant=0.2)
    err = np.abs(np.array(h['E']) / h['E'][0] - 1)
    assert err.max() < 1e-2, err.max()
    half = len(err) // 2
    assert err[half:].max() < 1.5 * err[:half].max(), (err[:half].max(), err[half:].max())


def run_tip_moment_curl(frac=0.5, n_ramp_periods=6, n_hold_periods=4):
    """Tip moment ramped to frac*2*pi*EI/L with damping (dynamic relaxation): beam curls into an arc."""
    c = cantilever()
    w1 = 1.875104**2 * math.sqrt(c['EI'] / (c['rhoA'] * c['L']**4))
    T1 = 2 * math.pi / w1
    n_ramp, n_hold = int(n_ramp_periods * T1 / c['dt']), int(n_hold_periods * T1 / c['dt'])
    Mmax = frac * 2 * math.pi * c['EI'] / c['L']
    tr = n_ramp * c['dt']
    zero = torch.zeros(c['fr'].N, 3)

    def f_ext(t):
        Mom = zero.clone()
        # follower-free (fixed direction) moment about -y bends the beam toward +z; global axis is fine
        # here because a pure end moment about a fixed axis keeps the beam in the x-z plane
        Mom[-1, 1] = -Mmax * min(t / tr, 1.0)
        return zero, Mom

    with torch.no_grad():
        h = simulate(c['fr'], c['law'], c['m'], c['j'], c['dt'], n_ramp + n_hold, f_ext=f_ext,
                     fix_u=c['fix'], fix_r=c['fix'], damping=0.5 * w1, record_every=50)
    kappa = Mmax / c['EI']
    tip_exact = np.array([math.sin(kappa * c['L']) / kappa, 0.0, (1 - math.cos(kappa * c['L'])) / kappa])
    tip = (c['fr'].X[-1] + h['u'][-1][-1]).numpy()
    return c, h, tip, tip_exact


def test_tip_moment_half_circle():
    c, _, tip, exact = run_tip_moment_curl(frac=0.5)
    assert np.linalg.norm(tip - exact) / c['L'] < 0.02, (tip, exact)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plotFrame import animate_frame, numbered

    OUT.mkdir(parents=True, exist_ok=True)

    c, w1, h = run_free_vibration(w_tip=10.0, periods=2.0)
    tip = torch.stack(h['tip']).numpy()
    Eh = np.array(h['E'])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(np.array(h['t']) * 1e3, tip[:, 2], label='tip z')
    ax[0].plot(np.array(h['t']) * 1e3, tip[:, 0], label='tip x (shortening)')
    ax[0].set(xlabel='t [ms]', ylabel='tip displacement [mm]', title='free vibration, tip 10 mm (L = 30)')
    ax[0].legend()
    ax[1].plot(np.array(h['t']) * 1e3, (Eh - Eh[0]) / Eh[0])
    ax[1].set(xlabel='t [ms]', ylabel='(E - E0)/E0', title='energy drift (no damping)')
    fig.tight_layout()
    fig.savefig(numbered(OUT, 'cantilever_free_vibration.png'), dpi=130)
    animate_frame(c['fr'], h['u'], numbered(OUT, 'cantilever_free_vibration.gif'), times=h['t'], elev=0, azim=-90,
                  title='TPU cantilever, free vibration')
    print('dt', c['dt'], 'max |dE|/E0', np.abs(Eh - Eh[0]).max() / Eh[0])

    _, w1s, hs = run_free_vibration(w_tip=0.05, periods=3.0)
    print('omega measured', measured_omega(hs['t'], torch.stack(hs['tip'])[:, 2].numpy()), 'EB', w1s)

    c, h, tip, exact = run_tip_moment_curl(frac=0.5)
    print('half circle tip', tip, 'exact', exact)
    animate_frame(c['fr'], h['u'], numbered(OUT, 'cantilever_tip_moment_half_circle.gif'), times=h['t'], elev=0, azim=-90,
                  title='tip moment -> half circle')
    c, h, tip, exact = run_tip_moment_curl(frac=1.0, n_ramp_periods=10, n_hold_periods=6)
    print('full circle tip', tip, 'exact', exact)
    animate_frame(c['fr'], h['u'], numbered(OUT, 'cantilever_tip_moment_full_circle.gif'), times=h['t'], elev=0, azim=-90,
                  title='tip moment -> full circle')
