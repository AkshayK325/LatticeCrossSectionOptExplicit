"""Day 3: flat-surface impact with smooth penalty contact (TPU_90A, E0 = 6.70 MPa).

Run as a script to write lattice impact GIFs/plots to ResultsFig/explicit/.
"""
import math
import pathlib
from functools import partial

import numpy as np
import torch

from contact import flat_surface, flat_surface_potential, node_radius
from explicitFE import Frame, lumped_mass, stable_dt, adaptive_dt
from explicitSolve import simulate, strain_energy
from material import tpu90a_impact

torch.set_default_dtype(torch.float64)
MAT = tpu90a_impact()
E, G, RHO = MAT['E'], MAT['G'], MAT['rho']
OUT = pathlib.Path(__file__).resolve().parents[1] / 'ResultsFig' / 'explicit'


def circle_sections(ne, r):
    r = torch.as_tensor(r, dtype=torch.float64).expand(ne)
    A = (math.pi * r**2)[:, None].expand(ne, 3).clone()
    I = torch.stack((math.pi * r**4 / 2, math.pi * r**4 / 4, math.pi * r**4 / 4), 1)
    return A, I


def build(X, conn, r, energy_J, v0_ms, kc_factor=10.0, eps_frac=0.02, gap0=0.05, top_tol=1e-6, courant=0.7):
    """Lattice (X, conn already refined) resting above z=0, bonded impactor plate on the top nodes.
    Impactor mass M = 2E/v^2. kc = kc_factor * mean element EA/L. eps = eps_frac * r_mean."""
    fr0 = Frame(X, conn)
    A, I = circle_sections(fr0.ne, r)
    rn = node_radius(fr0, A)
    X = X.copy()
    X[:, 2] += gap0 - (X[:, 2] - rn.numpy()).min()       # lowest beam surface gap0 above the plane
    fr = Frame(X, conn)
    law = lambda d: fr.elastic_law(d, A, I, E, G)
    m, j = lumped_mass(fr, A, I, RHO)
    top = torch.nonzero(fr.X[:, 2] > fr.X[:, 2].max() - top_tol).flatten()
    v0 = v0_ms * 1e3
    M = 2 * energy_J * 1e3 / v0**2                          # tonne
    m = m.index_add(0, top, torch.full((len(top),), M / len(top)))
    kc = kc_factor * (E * A[:, 0] / fr.L0).mean().item()
    eps = eps_frac * rn.mean().item()
    _, w_beam = stable_dt(fr, law, m, j)
    w_c = math.sqrt(kc / m.min().item())
    # beam and contact stiffness act on the same node: frequencies add in quadrature, not min(dt)
    dt_beam, dt_c = courant * 2 / w_beam, courant * 2 / w_c
    fix_r = torch.zeros(fr.N, 3, dtype=torch.bool)
    fix_r[top] = True
    dt_fn = adaptive_dt(fr, law, m, j, w_extra=w_c, courant=courant)
    return dict(fr=fr, A=A, I=I, law=law, m=m, j=j, rn=rn, top=top, probe=top[0].item(), M=M, dt_fn=dt_fn, v0=v0, kc=kc, eps=eps,
                dt=courant * 2 / math.sqrt(w_beam**2 + w_c**2), dt_beam=dt_beam, dt_c=dt_c, fix_r=fix_r,
                contact=lambda u, v: flat_surface(fr.X + u, rn, kc, eps))


def run(c, T, record_every=1):
    fr = c['fr']
    v0 = torch.zeros(fr.N, 3)
    v0[:, 2] = -c['v0']
    W = {'Wc': 0.0, 'P_prev': None, 't_prev': 0.0}

    def cb(n, t, state, fint):
        u, v, R, w = state
        _, _, d, f_loc, fc = fint
        P = (fc * v).sum().item()                            # power of contact forces
        if W['P_prev'] is not None:
            W['Wc'] += 0.5 * (P + W['P_prev']) * (t - W['t_prev'])
        W['P_prev'], W['t_prev'] = P, t
        ke = 0.5 * (c['m'] * (v**2).sum(1)).sum().item() + 0.5 * (c['j'] * (w**2).sum(1)).sum().item()
        U = strain_energy(d, f_loc).item()
        gap = (fr.X[:, 2] + u[:, 2] - c['rn']).min().item()
        p = c['probe']
        Phi = flat_surface_potential(fr.X + u, c['rn'], c['kc'], c['eps'])
        return {'KE': ke, 'U': U, 'Wc': W['Wc'], 'Phi': Phi, 'F_wall': fc[:, 2].sum().item(),
                'z_plate': u[p, 2].item(), 'v_plate': v[p, 2].item(), 'gap_min': gap,
                'rot_max': d[:, 1:].abs().max().item()}   # largest local (element-relative) rotation [rad]

    with torch.no_grad():
        h = simulate(fr, c['law'], c['m'], c['j'], c['dt_fn'], None, t_end=T, v0=v0, fix_r=c['fix_r'],
                     contact=c['contact'], tie=c['top'], record_every=record_every, callback=cb)
    h = {k: (np.array(v) if isinstance(v, list) else v) for k, v in h.items()}
    # contact work is only exact when recorded every step; coarser records under-resolve it
    assert record_every == 1
    h['E_tot'] = h['KE'] + h['U'] + h['Phi']                 # exact contact potential (Wc kept as cross-check)
    return h


def lattice_3fz(elemSize=3.0):
    from mesher import MeshFrame
    H = 0.5 * 25.4
    params = {'base': H, 'height': H, 'phi': 0.0, 'delta': 0.0, 'nx': 3, 'ny': 3, 'nz': 1, 'Name': 'C4', 'Shape': 'Square'}
    mf = MeshFrame()
    X, conn, _, _ = mf.generateCombined3DLattice(params)
    return mf.refineFrameMeshElemSize(X, conn, elemSize)


# ---------------------------------------------------------------- tests
def test_contact_force_is_smooth_penalty():
    x = torch.tensor([[0.0, 0, 2.0], [0, 0, 1.0], [0, 0, 0.9]])
    r = torch.ones(3)
    f = flat_surface(x, r, kc=100.0, eps=0.01)
    assert f[0, 2] < 1e-30 and abs(f[1, 2] - 100 * 0.01 * math.log(2)) < 1e-12
    assert abs(f[2, 2] - 100 * 0.1) < 1e-3                      # deep: kc * penetration
    assert (f[:, :2] == 0).all()


def test_free_beam_dropped_flat_rebounds_elastically():
    X = np.stack([np.linspace(0, 20, 11), np.zeros(11), np.zeros(11)], 1)
    conn = np.stack([np.arange(10), np.arange(1, 11)], 1)
    c = build(X, conn, r=1.0, energy_J=1e-3, v0_ms=1.0)
    c['top'], c['probe'] = None, 5            # free beam: no impactor plate (build put M on its nodes)
    c['fix_r'][:] = False
    h = run(c, T=1.5e-3)
    assert h['v_plate'][-1] > 0.9e3, h['v_plate'][-1]           # straight drop, uniform contact: ~elastic rebound
    err = np.abs(h['E_tot'] / h['E_tot'][0] - 1)
    # contact resolved in ~10 steps: O((w dt)^2) Verlet error, ~2% transient and ~0.4% net per contact
    # event at courant 0.7 (shrinks with dt^2)
    assert err.max() < 5e-2 and err[-1] < 1e-2, (err.max(), err[-1])


def test_lattice_impact_energy_balance_and_penetration():
    X, conn = lattice_3fz()
    c = build(X, conn, r=1.0, energy_J=1.0, v0_ms=6.0)
    h = run(c, T=8e-3)
    Eerr = np.abs(h['E_tot'] - h['E_tot'][0]).max() / h['E_tot'][0]
    assert Eerr < 2e-2, Eerr
    assert h['gap_min'].min() > -0.5 * 1.0, h['gap_min'].min()    # penetration < r/2
    assert h['v_plate'][-1] > 0, 'impactor did not rebound'


if __name__ == '__main__':
    import sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plotFrame import animate_frame

    OUT.mkdir(parents=True, exist_ok=True)
    X, conn = lattice_3fz()
    cases = [(1.0, 6.0), (1.0, 10.0), (5.0, 6.0), (5.0, 10.0)] if 'all' in sys.argv else [(1.0, 6.0), (5.0, 10.0)]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for Ej, vms in cases:
        c = build(X, conn, r=1.0, energy_J=Ej, v0_ms=vms)
        h = run(c, T=10e-3)
        t = h['t'] * 1e3
        lab = f'{Ej:g} J, {vms:g} m/s (M = {c["M"]*1e6:.0f} g)'
        ax[0].plot(t, h['F_wall'], label=lab)
        ax[1].plot(-h['z_plate'], h['F_wall'], label=lab)
        ax[2].plot(t, (h['E_tot'] - h['E_tot'][0]) / h['E_tot'][0], label=lab)
        print(lab, f"N={c['fr'].N} ne={c['fr'].ne} dt={c['dt']:.2e} (beam {c['dt_beam']:.2e}, contact {c['dt_c']:.2e})",
              f"steps={len(t)} peakF={h['F_wall'].max():.1f} N stroke={-h['z_plate'].min():.2f} mm",
              f"min gap={h['gap_min'].min():.3f} mm |dE|/E0={np.abs(h['E_tot']-h['E_tot'][0]).max()/h['E_tot'][0]:.2e}",
              f"v_end={h['v_plate'][-1]/1e3:.2f} m/s")
        plane = lambda a, k: a.plot_surface(*np.meshgrid(np.linspace(-2, 40, 2), np.linspace(-2, 40, 2)),
                                            np.zeros((2, 2)), color='0.5', alpha=0.3)
        animate_frame(c['fr'], h['u'], OUT / f'lattice_impact_{Ej:g}J_{vms:g}ms.gif', times=h['t'], elev=8, azim=-60,
                      title=f'{lab}', extra=plane, max_frames=60)
    ax[0].set(xlabel='t [ms]', ylabel='surface force [N]', title='force on flat surface')
    ax[1].set(xlabel='impactor stroke [mm]', ylabel='surface force [N]', title='force–stroke')
    ax[2].set(xlabel='t [ms]', ylabel='(E − E0)/E0', title='energy balance (KE + U + contact potential)')
    for a in ax:
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'lattice_impact_histories.png', dpi=130)
