"""exampleCase-3 lattice + symmetry, contact laws, and the bonded-bottom / top-plate drop rig (ENERGY_J, 0.75 J).

Run as a script for the contact-option comparison figure and the drop GIF (ResultsFig/explicit/drop_*).
"""
import pathlib

import numpy as np
import pytest
import torch

from contact import calibrate
from latticeSetup import (SYM_TYPES, example3, symmetry_groups, n_design_rows, apply_symmetry, element_values,
                          drop_rig, run_drop, jump_metrics)
from material import project_material

torch.set_default_dtype(torch.float64)
MAT = project_material()   # EPU 45; elastic tests use its long-term E
OUT = pathlib.Path(__file__).resolve().parents[1] / 'ResultsFig' / 'explicit'
DEFAULT = {'kind': 'crit_damped', 'F_ref': 2000.0, 'delta_ref': 0.2, 'zeta': 1.0}
ENERGY_J = 0.75     # user choice 2026-10-01 (was 1 J)
R_DEFAULT = 0.8      # user choice 2026-10-01: r = 0.8 (d = 1.6 mm); r = 1.6 (d = 3.2) too large for this system


@pytest.fixture(scope='module')
def lat():
    return example3()


# ---------------------------------------------------------------- lattice + symmetry
def test_example3_matches_notebook_mesh(lat):
    from legacy_setup import build_3fz
    opt, _ = build_3fz('circle')
    assert np.allclose(lat['Xr'], opt.frameFE.nodeXY) and (lat['connr'] == opt.frameFE.connectivity).all()


@pytest.mark.parametrize('symType', SYM_TYPES)
def test_symmetry_matches_legacy_applySymm(lat, symType):
    from legacy_setup import build_3fz
    opt, _ = build_3fz('circle')
    opt.varSetup['symArray'] = symmetry_groups(symType, lat)
    x = torch.rand(n_design_rows(symType, lat), 1, generator=torch.Generator().manual_seed(0))
    assert torch.equal(apply_symmetry(x, opt.varSetup['symArray'], lat), opt.applySymm(x))
    assert element_values(x, symType, lat).shape == (lat['connr'].shape[0], 1)


# ---------------------------------------------------------------- contact laws
@pytest.mark.parametrize('kind', ['linear', 'hertz', 'hunt_crossley', 'crit_damped', 'barrier'])
def test_contact_laws_continuous_nonnegative(kind):
    law = calibrate(kind, 2000.0, 0.2, mu=1e-5)
    g = torch.linspace(-0.3 if kind != 'barrier' else 0.02, 1.5, 4001)   # barrier is singular at g -> 0 by design
    for gd in (-6000.0, 0.0, 3000.0):
        F = law(g, torch.full_like(g, gd))
        assert (F >= 0).all() and torch.isfinite(F).all()
        assert F[-1] < 1e-6 * 2000.0                              # separated: < 1e-6 F_ref (crit_damped's damping
                                                                  # onset decays as exp(-g/4eps), ~1e-8 F_ref here)
        assert (F.diff().abs().max() < 0.05 * F.abs().max() + 1e-9)   # no jump on a fine gap grid


def test_crit_damped_releases_exactly_on_separation():
    law = calibrate('crit_damped', 2000.0, 0.2, mu=1e-5)
    assert law(torch.tensor(1.0), torch.tensor(5000.0)) == 0.0   # opening, far apart: exactly zero
    assert law(torch.tensor(0.05), torch.tensor(-100.0)) > law(torch.tensor(0.05), torch.tensor(0.0))


# ---------------------------------------------------------------- drop rig
@pytest.fixture(scope='module')
def drop(lat):
    rig = drop_rig(lat, torch.full((lat['connr'].shape[0],), R_DEFAULT), DEFAULT, MAT, start='co_moving', gap0=0.1, energy_J=ENERGY_J)
    return rig, run_drop(rig, T=8e-3)


def test_drop_bottom_bonded_and_no_fly_off(drop):
    rig, h = drop
    assert h['u'][:, rig['bot']].abs().max() == 0                 # bonded to the rigid bottom plate
    assert h['z_plate'].max() < 3.0                               # top plate stays on the lattice (mm)
    assert h['v_imp'][-1] > 0                                     # impactor separates and goes up


def test_drop_no_crush_through(drop):
    """Top plate must stop above the compacted lattice (~2 layers of beams, 2*d), not pass the bottom plate."""
    rig, h = drop
    H = rig['fr'].X[rig['top'][0], 2].item()
    assert -h['z_plate'].min() < H - 2 * (2 * R_DEFAULT), (-h['z_plate'].min(), H)


def test_drop_contact_deformation_bounded(drop):
    """Beam-beam and beam-plate approach (Hertz indentation) stays a small fraction of the radius."""
    _, h = drop
    assert np.min(h['g_self_min']) > -0.15 * R_DEFAULT, np.min(h['g_self_min'])
    assert np.min(h['g_plate_min']) > -0.15 * R_DEFAULT, np.min(h['g_plate_min'])


def test_drop_energy_balance(drop):
    _, h = drop
    E = h['E_tot']
    assert np.abs(E - E[0]).max() / E[0] < 1e-2
    assert abs(E[0] - ENERGY_J * 1e3) < 1e-6                      # J -> N*mm


def test_drop_single_smooth_pulse(drop):
    _, h = drop
    jm = jump_metrics(h['F_imp'])
    assert jm['episodes'] == 1 or jm['chatter'] < 4, jm


def test_drop_epu45_absorbs_not_springs(drop):
    """EPU 45 (tan d ~ 2 at ~1 kHz) must absorb most of the impact energy; a spring-like lattice would return it."""
    rig, h = drop
    assert h['E_diss_material'][-1] > 0.5 * ENERGY_J * 1e3, h['E_diss_material'][-1]
    assert h['v_imp'][-1] < 0.5 * rig['v0'], h['v_imp'][-1]


def test_drop_force_has_no_jump_dt_convergence(lat):
    """A true discontinuity keeps its per-step size as dt -> 0; a continuous force's shrinks ~ dt."""
    jumps = []
    for c in (0.7, 0.35):
        rig = drop_rig(lat, torch.full((lat['connr'].shape[0],), R_DEFAULT), DEFAULT, MAT, start='co_moving',
                       gap0=0.1, courant=c, energy_J=ENERGY_J)
        jumps.append(np.abs(np.diff(run_drop(rig, T=3e-3)['F_imp'])).max())
    assert jumps[1] / jumps[0] < 0.6, jumps


# ---------------------------------------------------------------- figures
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plotFrame import animate_frame, numbered

    OUT.mkdir(parents=True, exist_ok=True)
    lat = example3()
    r = torch.full((lat['connr'].shape[0],), R_DEFAULT)
    base = {'F_ref': 2000.0, 'delta_ref': 0.2}
    variants = [('gap', {'kind': 'linear'}, 'A  gap · linear penalty'),
                ('gap', {'kind': 'hertz'}, 'B  gap · Hertz'),
                ('gap', {'kind': 'hunt_crossley', 'e': 0.3}, 'C  gap · Hunt–Crossley e=0.3'),
                ('gap', {'kind': 'barrier'}, 'D  gap · IPC barrier'),
                ('co_moving', {'kind': 'hertz'}, 'E  co-moving · Hertz'),
                ('co_moving', {'kind': 'crit_damped'}, 'F  co-moving · crit. damped (default)')]
    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    rows = []
    for i, (start, spec, lab) in enumerate(variants):
        rig = drop_rig(lat, r, {**base, **spec}, MAT, start=start, gap0=0.3 if start == 'gap' else 0.1,
                       energy_J=ENERGY_J)
        h = run_drop(rig, T=8e-3)
        t = h['t'] * 1e3
        a = ax.flat[i]
        a.plot(t, h['F_imp'], lw=1, label='impactor–plate force')
        a.plot(t, -h['F_bottom'], lw=1, label='bottom-plate reaction')
        jm = jump_metrics(h['F_imp'])
        E = h['E_tot']
        dE = np.abs(E - E[0]).max() / E[0]
        a.set(title=f'{lab}\npeak {jm["peak"]:.0f} N, episodes {jm["episodes"]}, chatter {jm["chatter"]:.1f}, |ΔE| {dE:.1e}',
              xlabel='t [ms]', ylabel='force [N]')
        a.title.set_fontsize(9)
        a.legend(fontsize=7)
        rows.append((lab, jm, dE, h['Wd'][-1], -h['z_plate'].min(), h['v_imp'][-1] / 1e3))
        print(f'{lab:40s} peak={jm["peak"]:6.0f} episodes={jm["episodes"]:2d} chatter={jm["chatter"]:5.2f} '
              f'|dE|={dE:.1e} Wd={h["Wd"][-1]:6.1f} stroke={-h["z_plate"].min():.2f} v_end={h["v_imp"][-1]/1e3:+.2f}', flush=True)
        if spec['kind'] == 'crit_damped':
            best = (rig, h)
    fig.tight_layout()
    fig.savefig(numbered(OUT, 'drop_contact_options.png'), dpi=120)

    rig, h = best
    fr = rig['fr']
    x0, x1 = fr.X[:-1, 0].min().item() - 2, fr.X[:-1, 0].max().item() + 2
    y0, y1 = fr.X[:-1, 1].min().item() - 2, fr.X[:-1, 1].max().item() + 2
    ztop0 = fr.X[rig['top'][0], 2].item()
    XX, YY = np.meshgrid([x0, x1], [y0, y1])

    def plates(ax, k):
        ax.plot_surface(XX, YY, np.zeros((2, 2)) - 0.5, color='0.4', alpha=0.4)              # bonded bottom plate
        zp = ztop0 + h['u'][k, rig['top'][0], 2].item()
        ax.plot_surface(XX, YY, np.full((2, 2), zp + 0.5), color='tab:orange', alpha=0.5)    # top plate
        zi = ztop0 + rig['gap0'] + h['z_imp'][k] + 1.0
        xi, yi = np.meshgrid([x0 + 8, x1 - 8], [y0 + 8, y1 - 8])
        ax.plot_surface(xi, yi, np.full((2, 2), zi), color='tab:red', alpha=0.7)              # impactor face

    U = h['u'].clone()
    U[:, -1] = 0                                                   # impactor drawn as a plate, not a node
    animate_frame(fr, U, numbered(OUT, f'drop_{ENERGY_J:g}J_6ms_r{R_DEFAULT:g}.gif'), times=h['t'], elev=10, azim=-60, extra=plates,
                  title=f'{ENERGY_J:g} J, 6 m/s, r = {R_DEFAULT:g} mm, bonded bottom', max_frames=70)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    t = h['t'] * 1e3
    ax[0].plot(t, h['F_imp'], label='impactor–plate')
    ax[0].plot(t, -h['F_bottom'], label='bottom reaction')
    ax[0].set(xlabel='t [ms]', ylabel='force [N]', title='forces')
    ax[0].legend()
    ax[1].plot(-h['z_plate'], h['F_imp'])
    ax[1].set(xlabel='top-plate stroke [mm]', ylabel='impactor force [N]', title='force–stroke')
    for k, lab in (('KE', 'kinetic'), ('U', 'stored (long-term strain)'), ('E_diss_material', 'material (Prony) dissipation'),
                   ('Phi', 'contact potential'), ('Wd', 'contact damping')):
        ax[2].plot(t, h[k] / 1e3, label=lab)
    ax[2].plot(t, h['E_tot'] / 1e3, 'k--', label='total')
    ax[2].set(xlabel='t [ms]', ylabel='energy [J]', title='energy budget')
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(numbered(OUT, f'drop_{ENERGY_J:g}J_6ms_r{R_DEFAULT:g}_histories.png'), dpi=130)
