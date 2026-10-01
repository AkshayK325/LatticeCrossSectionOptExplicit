"""Day 1 toy: 1D TPU column + impactor dropping onto a flat rigid wall.

Explicit central difference (velocity Verlet), softplus penalty contact, all torch float64.
Design variables: spring cross-section areas A_i. Checks energy conservation and autograd vs FD.
Run as a script to write figures/GIF to ResultsFig/explicit/toy_*.
"""
import math
import pathlib

import pytest
import torch
import torch.nn.functional as F

from material import tpu90a_impact

torch.set_default_dtype(torch.float64)
MAT = tpu90a_impact()
OUT = pathlib.Path(__file__).resolve().parents[1] / 'ResultsFig' / 'explicit'


def setup(n_el=20, H=20.0, A0=100.0, energy=2.0, v0=8000.0, gap0=0.5):
    """Units mm/N/s/t. Impactor mass M from coupon energy: M = 2E/v^2 (E in N*mm)."""
    M = 2 * energy * 1e3 / v0**2
    return dict(n_el=n_el, h=H / n_el, A=torch.full((n_el,), A0), M=M, v0=v0,
                z0=gap0 + torch.linspace(0, H, n_el + 1))


def contact(z, kc, eps):
    """Smooth penalty on gap g=z: force kc*eps*softplus(-g/eps) (>0 pushes up), potential integral of it."""
    return kc * eps * F.softplus(-z / eps)


def simulate(A, cfg, kc_factor=10.0, eps=0.01, T=8e-3, courant=0.5, keep_states=False):
    E, rho, h = MAT['E'], MAT['rho'], cfg['h']
    k = E * A / h                                   # spring stiffness per element
    m = torch.zeros(cfg['n_el'] + 1)
    m = m.index_add(0, torch.arange(cfg['n_el']), rho * A * h / 2).index_add(0, torch.arange(1, cfg['n_el'] + 1), rho * A * h / 2)
    m = m + torch.nn.functional.one_hot(torch.tensor(cfg['n_el']), cfg['n_el'] + 1) * cfg['M']  # impactor on top node
    # penalty and time grid come from the nominal design cfg['A'], not from A: they must stay fixed
    # under design perturbation, otherwise FD sees a different time grid than autograd
    k_nom = E * cfg['A'].max().item() / h
    kc = kc_factor * k_nom
    w_max = 2 * math.sqrt((2 * k_nom + kc) / (rho * cfg['A'].min().item() * h / 2))
    dt = courant * 2 / w_max
    n = int(T / dt)

    def force(z):
        stretch = z[1:] - z[:-1] - h
        f_s = k * stretch
        f = torch.zeros_like(z).index_add(0, torch.arange(1, z.numel()), -f_s).index_add(0, torch.arange(z.numel() - 1), f_s)
        fc = contact(z, kc, eps)
        return f + fc, (fc.sum(), f_s[-1])  # wall force, force carried into the impactor

    z, v = cfg['z0'].clone(), torch.full_like(cfg['z0'], -cfg['v0'])
    f, (Fw, Fi) = force(z)
    Fwall, Fimp, states = [Fw], [Fi], [z.detach().clone()]
    for _ in range(n):
        v = v + 0.5 * dt * f / m
        z = z + dt * v
        f, (Fw, Fi) = force(z)
        v = v + 0.5 * dt * f / m
        Fwall.append(Fw)
        Fimp.append(-Fi)
        if keep_states:
            states.append(z.detach().clone())
    energy = lambda z, v: 0.5 * (m * v**2).sum() + 0.5 * (k * (z[1:] - z[:-1] - h)**2).sum()
    return dict(F=torch.stack(Fwall), F_imp=torch.stack(Fimp), dt=dt, n=n, z=z, v=v, E0=energy(cfg['z0'], torch.full_like(z, -cfg['v0'])),
                E_end=energy(z, v), states=states, eps=eps, m=m)


def J_peak(res, p=8, key='F'):
    """Placeholder objective (smooth max of a force history) — the real objective is deferred (plan 3)."""
    return res[key].clamp_min(0).pow(p).mean().pow(1 / p)


def J_peak_imp(res):
    return J_peak(res, key='F_imp')


def fd_check(cfg, J=J_peak, idx=(0, 7, 19), rel=1e-6, **kw):
    A = cfg['A'].clone().requires_grad_(True)
    g = torch.autograd.grad(J(simulate(A, cfg, **kw)), A)[0]
    out = []
    for i in idx:
        dA = rel * cfg['A'][i].item()
        Ap, Am = cfg['A'].clone(), cfg['A'].clone()
        Ap[i] += dA
        Am[i] -= dA
        fd = (J(simulate(Ap, cfg, **kw)) - J(simulate(Am, cfg, **kw))).item() / (2 * dA)
        out.append((g[i].item(), fd))
    return out


# ---------------------------------------------------------------- tests
def test_energy_conserved_after_rebound():
    cfg = setup()
    r = simulate(cfg['A'], cfg)
    assert (r['z'] > 20 * r['eps']).all(), 'column has not left the wall; increase T'
    assert r['v'][-1] > 0, 'impactor did not rebound'
    assert abs(r['E_end'] - r['E0']) / r['E0'] < 1e-2


@pytest.mark.parametrize('J', [J_peak, J_peak_imp])
def test_gradient_matches_fd(J):
    cfg = setup()
    # vary areas so the gradient is not symmetric
    cfg['A'] = cfg['A'] * (1 + 0.3 * torch.sin(torch.arange(cfg['n_el']).double()))
    for ad, fd in fd_check(cfg, J=J):
        assert abs(ad - fd) <= 1e-5 * max(abs(fd), 1e-12), (ad, fd)


# ---------------------------------------------------------------- figures
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = setup()
    cfg['A'] = cfg['A'] * (1 + 0.3 * torch.sin(torch.arange(cfg['n_el']).double()))

    # 1. force history + energy
    r = simulate(cfg['A'], cfg, keep_states=True)
    t = torch.arange(r['n'] + 1) * r['dt'] * 1e3
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(t, r['F'].detach(), lw=0.5, label='wall (contact) force')
    ax[0].plot(t, r['F_imp'].detach(), lw=1.5, label='force on impactor (M·a)')
    ax[0].legend()
    ax[0].set(xlabel='t [ms]', ylabel='force [N]', title=f"2 J, 8 m/s, M={cfg['M']*1e6:.1f} g")
    zs = torch.stack(r['states'])
    ax[1].plot(t, zs[:, 0], label='bottom node')
    ax[1].plot(t, zs[:, -1], label='impactor')
    ax[1].axhline(0, color='k', lw=0.8)
    ax[1].set(xlabel='t [ms]', ylabel='z [mm]', title=f"|dE|/E0 = {abs(r['E_end']-r['E0']).item()/r['E0'].item():.1e}")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / 'toy_force_history.png', dpi=130)

    # 2. GIF of the column
    fig, ax = plt.subplots(figsize=(3, 5))
    stride = max(1, len(r['states']) // 120)
    def frame(i):
        ax.clear()
        z = r['states'][i * stride]
        ax.plot([0] * len(z), z, 'o-', ms=3)
        ax.plot([0], [z[-1]], 's', ms=12, color='C3')
        ax.axhline(0, color='k', lw=2)
        ax.set(xlim=(-1, 1), ylim=(-1, 22), xticks=[], title=f't = {i*stride*r["dt"]*1e3:.2f} ms')
    FuncAnimation(fig, frame, frames=len(r['states']) // stride).save(OUT / 'toy_impact.gif', writer=PillowWriter(fps=20))

    # 3. contact smoothing sweep: gradient value and FD agreement
    eps_list = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for J, name, ls in ((J_peak, 'wall force', '--'), (J_peak_imp, 'impactor force', '-')):
        for kcf in (1.0, 10.0, 100.0):
            gv, err = [], []
            for eps in eps_list:
                (ad, fd), = fd_check(cfg, J=J, idx=(0,), eps=eps, kc_factor=kcf)
                gv.append(ad)
                err.append(abs(ad - fd) / abs(fd))
            lab = f'{name}, k_c={kcf:g}·EA/h'
            ax[0 if name == 'wall force' else 1].semilogx(eps_list, gv, 'o' + ls, label=lab)
            ax[2].loglog(eps_list, err, 'o' + ls, label=lab)
            print(name, kcf, [(e, round(g, 5), f'{r:.1e}') for e, g, r in zip(eps_list, gv, err)])
    for a, nm in zip(ax[:2], ('wall force', 'impactor force')):
        a.set_yscale('symlog', linthresh=1e-2)
        a.set(xlabel='contact smoothing ε [mm]', ylabel='dJ/dA_0', title=f'placeholder J = smooth max of {nm}')
        a.legend(fontsize=7)
    ax[2].set(xlabel='contact smoothing ε [mm]', ylabel='|AD − FD| / |FD|', title='autograd vs central FD')
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'toy_eps_sweep.png', dpi=130)
    print('dt', r['dt'], 'steps', r['n'], 'dE/E0', (abs(r['E_end'] - r['E0']) / r['E0']).item())
